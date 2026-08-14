#!/usr/bin/env python3
"""
13_qualitative_figure.py — the CT examples the manuscript is missing.

WHY THIS FIGURE IS NEEDED
-------------------------
The paper currently contains no CT images at all. For a medical imaging paper
that is unusual on its own, but here it is a specific evidential gap: the
Discussion asserts that epidural and subdural collections "share an extra-axial
location adjacent to the inner skull table and are distinguished primarily by
whether the collection crosses sutures and by concave-versus-biconvex
morphology --- features that are subtle at 5 mm slice thickness and frequently
ambiguous when both are present."

That is the mechanism the whole data-level argument rests on, and it is asserted
without a single image. A radiologist reviewer will want to see it; a machine
learning reviewer will want to see what the 3.9x confusion actually looks like.

WHAT IT PRODUCES
----------------
A three-row panel:

  Row 1  True epidural studies the model ranked confidently. The biconvex,
         suture-bounded shape the literature describes.
  Row 2  True subdural studies. The crescentic, suture-crossing shape.
  Row 3  The false positives that matter: SDH-positive studies the model
         scored highly for EDH. This row IS the 3.9x enrichment, shown rather
         than tabulated.

Selection is deterministic and stated: within each group, studies are ordered
by model score and the top-k taken, with the slice of maximum predicted
attention shown. No hand-picking. The script prints the study identifiers it
chose so the figure is reproducible and auditable.

USAGE
    python 13_qualitative_figure.py                    # 3x4 panel
    python 13_qualitative_figure.py --n 5 --window brain

NOTE ON PATIENT PRIVACY
-----------------------
RSNA slices are de-identified by the distributor and carry no burned-in
identifiers, but check the rendered figure before submission: this is the one
place in the paper where raw pixel data is reproduced.
"""
import argparse, json, glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ich_config import ROOT, MAN_DIR, MEMMAP_DIR, S2_DIR, ALL_COLS, PROFILE

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
EDH, SDH = 0, 4


def load_pooled(tpl="stage2_cond_fold{F}_conditional_" + PROFILE, folds=(0, 1, 2, 3, 4)):
    P, T, S = [], [], []
    for f in folds:
        npz = S2_DIR / tpl.replace("{F}", str(f)) / "val_predictions.npz"
        if not npz.exists():
            print(f"  missing {npz}"); continue
        d = np.load(npz, allow_pickle=True)
        P.append(d["probs"]); T.append(d["targets"]); S.append(d["study"].astype(str))
    return np.concatenate(P), np.concatenate(T), np.concatenate(S)


def slice_of_max_signal(idx_rows, mm, ch=0):
    """Pick the slice with the largest blood-density area, as a proxy for the
    most informative slice. Deterministic; no manual choice."""
    best, best_v = idx_rows[0], -1
    for r in idx_rows:
        a = np.asarray(mm[r][ch])
        v = float((a >= 159).sum())          # >= ~50 HU in the brain window
        if v > best_v:
            best, best_v = r, v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="examples per row")
    ap.add_argument("--channel", type=int, default=0,
                    help="0=brain 40/80, 1=subdural 80/200, 2=bone 600/2800")
    a = ap.parse_args()

    P, T, S = load_pooled()
    idx = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    idx = idx[idx["ok"]]
    meta = json.load(open(MEMMAP_DIR / "memmap_meta.json"))
    mm = np.memmap(MEMMAP_DIR / meta["dat"], dtype=np.uint8, mode="r",
                   shape=(meta["n"], meta["c"], meta["h"], meta["w"]))
    by_study = {s: g["row"].tolist() for s, g in idx.groupby("study", sort=False)}

    y_edh, y_sdh, p_edh = T[:, EDH].astype(int), T[:, SDH].astype(int), P[:, EDH]

    groups = [
        ("True epidural",
         np.where((y_edh == 1))[0], p_edh, True,
         "biconvex, does not cross sutures"),
        ("True subdural (no epidural)",
         np.where((y_sdh == 1) & (y_edh == 0))[0], P[:, SDH], True,
         "crescentic, crosses sutures"),
        ("Epidural false positives on subdural studies",
         np.where((y_edh == 0) & (y_sdh == 1))[0], p_edh, True,
         "the 3.9$\\times$ enrichment, shown"),
    ]

    fig, axes = plt.subplots(3, a.n, figsize=(2.3 * a.n, 7.4))
    chosen = {}
    for r, (title, pool, score, desc, caption) in enumerate(groups):
        order = pool[np.argsort(-score[pool])]
        picked = []
        for k in order:
            st = S[k]
            if st in by_study:
                picked.append((st, score[k]))
            if len(picked) == a.n:
                break
        chosen[title] = picked
        for c in range(a.n):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(picked):
                continue
            st, sc = picked[c]
            row = slice_of_max_signal(by_study[st], mm, a.channel)
            ax.imshow(np.asarray(mm[row][a.channel]), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"$p_{{\\mathrm{{EDH}}}}={sc:.2f}$", fontsize=8, pad=2)
        axes[r, 0].set_ylabel(title, fontsize=9)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values():
            sp.set_visible(False)

    fig.suptitle("Epidural and subdural hemorrhage on non-contrast CT, and the "
                 "confusion between them", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGS / "qualitative_edh_sdh.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")

    # provenance: which studies were shown, so the figure is auditable
    rec = {k: [{"study": s, "score": float(v)} for s, v in vv]
           for k, vv in chosen.items()}
    (RESULTS / "qualitative_figure_studies.json").write_text(json.dumps(rec, indent=2))
    print(f"[logged] {RESULTS/'qualitative_figure_studies.json'}")
    print("\nStudies shown (deterministic: highest-scoring within each group):")
    for k, vv in chosen.items():
        print(f"  {k}: " + ", ".join(s for s, _ in vv))
    print("\nCheck the rendered figure for burned-in identifiers before submission.")


if __name__ == "__main__":
    main()