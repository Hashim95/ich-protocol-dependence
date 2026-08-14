#!/usr/bin/env python3
"""
14_collect_for_paper.py — everything still missing from the manuscript, in one run.

Run this on ivision. It does four jobs:

  1. Rebuilds the qualitative EDH/SDH figure, correctly this time.
  2. Recovers wall-clock training cost per architecture from the run logs.
  3. Benchmarks inference latency (needed for the triage claim).
  4. Writes one JSON with every value the manuscript still marks [?].

WHAT WAS WRONG WITH THE FIRST FIGURE
------------------------------------
Slice selection used `ch0 >= 159`, intended to find blood-density voxels in the
brain window. But the skull saturates that window identically to acute blood ---
which is precisely why the paper's weak-knowledge mask uses a CONJUNCTION with
the bone channel. Selecting on ch0 alone therefore maximised bone, and returned
the skull base for every study: the one part of the head where extra-axial
collections are least visible.

Two further bugs: every panel was labelled p_EDH even in the subdural row, where
the selection score was the subdural probability; and the row labels collided.

The fix here does not use a heuristic at all. Stage-1 wrote per-slice
predictions with image_id and study. For each chosen study we display the slice
the model itself scored highest for the class in question. That is both more
defensible ("the slice driving the study-level prediction") and immune to the
bone confound.

USAGE
    python 14_collect_for_paper.py                  # everything
    python 14_collect_for_paper.py --skip-latency   # if the GPU is busy
    python 14_collect_for_paper.py --only figure
"""
import argparse, glob, json, os, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ich_config import (ROOT, MAN_DIR, MEMMAP_DIR, S1_DIR, S2_DIR, ALL_COLS,
                        TRAIN_RES, DEVICE, AMP_DTYPE, PROFILE, setup_hardware)

RESULTS = ROOT / "results"; FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
EDH, SDH = 0, 4
OUT = {}


# ============================================================ 1. FIGURE
def build_figure(n=4):
    print("\n[1/4] qualitative figure")

    # -- study-level predictions, to choose which studies to show ------------
    P, T, S = [], [], []
    for f in range(5):
        p = S2_DIR / f"stage2_cond_fold{f}_conditional_{PROFILE}" / "val_predictions.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        P.append(d["probs"]); T.append(d["targets"]); S.append(d["study"].astype(str))
    if not P:
        print("   no Stage-2 predictions found; skipping"); return
    P, T, S = np.concatenate(P), np.concatenate(T), np.concatenate(S)

    # -- slice-level predictions, to choose WHICH SLICE of each study --------
    sl = {}
    for f in range(5):
        p = S1_DIR / f"stage1_fold{f}_logit_adjusted_{PROFILE}" / "val_predictions.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        st = d["study"].astype(str); pr = d["probs"]
        iid = d["image_id"].astype(str) if "image_id" in d.files else None
        for k in range(len(st)):
            sl.setdefault(st[k], []).append((pr[k], iid[k] if iid is not None else None))
    has_slice_preds = len(sl) > 0
    print(f"   slice-level predictions available for {len(sl):,} studies"
          if has_slice_preds else
          "   WARNING: no slice-level predictions; falling back to blood-mask heuristic")

    idx = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    idx = idx[idx["ok"]]
    meta = json.load(open(MEMMAP_DIR / "memmap_meta.json"))
    mm = np.memmap(MEMMAP_DIR / meta["dat"], dtype=np.uint8, mode="r",
                   shape=(meta["n"], meta["c"], meta["h"], meta["w"]))
    rows_by_study, ids_by_study = {}, {}
    for s, g in idx.groupby("study", sort=False):
        g = g.sort_values("z", kind="stable")
        rows_by_study[s] = g["row"].tolist()
        ids_by_study[s] = g["image_id"].astype(str).tolist() if "image_id" in g else None

    def pick_slice(study, cls):
        """Slice the model scored highest for `cls`; blood-mask fallback."""
        rows = rows_by_study[study]
        if has_slice_preds and study in sl and ids_by_study[study]:
            byid = {i: p for p, i in sl[study] if i is not None}
            best, bestv = rows[0], -1.0
            for r, iid in zip(rows, ids_by_study[study]):
                v = float(byid.get(iid, [0] * 6)[cls]) if iid in byid else -1.0
                if v > bestv:
                    best, bestv = r, v
            if bestv >= 0:
                return best
        # fallback: TRUE blood mask -- blood-density AND not bone
        best, bestv = rows[0], -1
        for r in rows:
            a = np.asarray(mm[r])
            v = int(((a[0] >= 159) & (a[2] <= 110)).sum())
            if v > bestv:
                best, bestv = r, v
        return best

    y_edh, y_sdh = T[:, EDH].astype(int), T[:, SDH].astype(int)
    pe = P[:, EDH]

    # Row 4: cases at the decision boundary. Showing only p=1.00 examples makes
    # the class boundary look cleaner than it is; the near-threshold band is
    # where the ceiling actually lives, so it belongs in the figure.
    band = np.where((pe >= 0.4) & (pe <= 0.6))[0]
    # order by |p - 0.5| so the most ambiguous come first, not the most confident
    amb_score = -np.abs(pe - 0.5)

    groups = [
        ("True epidural", np.where(y_edh == 1)[0], P[:, EDH], EDH,
         r"$p_{\mathrm{EDH}}$", None),
        ("True subdural,\nno epidural", np.where((y_sdh == 1) & (y_edh == 0))[0],
         P[:, SDH], SDH, r"$p_{\mathrm{SDH}}$", None),
        ("Epidural false positives\non subdural studies",
         np.where((y_edh == 0) & (y_sdh == 1))[0], P[:, EDH], EDH,
         r"$p_{\mathrm{EDH}}$", None),
        ("Near-threshold,\n$p_{\mathrm{EDH}}\\in[0.4,0.6]$", band, amb_score, EDH,
         r"$p_{\mathrm{EDH}}$", pe),
    ]
    print(f"   {len(band)} studies in the near-threshold band "
          f"(EDH positives among them: {int(y_edh[band].sum())})")

    fig, axes = plt.subplots(4, n, figsize=(2.5 * n, 10.8))
    chosen = {}
    for r, (title, pool, score, cls, sym, disp) in enumerate(groups):
        order = pool[np.argsort(-score[pool])]
        picked = []
        for k in order:
            if S[k] in rows_by_study:
                picked.append((S[k], float(disp[k] if disp is not None else score[k]),
                               int(y_edh[k])))
            if len(picked) == n:
                break
        chosen[title.replace("\n", " ")] = [(a_, b_) for a_, b_, _ in picked]
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c >= len(picked):
                ax.axis("off"); continue
            st, sc, lab = picked[c]
            ax.imshow(np.asarray(mm[pick_slice(st, cls)][0]), cmap="gray",
                      vmin=0, vmax=255)
            tag = "" if r < 3 else ("  EDH+" if lab else "  EDH$-$")
            ax.set_title(f"{sym}$\\,=\\,{sc:.2f}${tag}", fontsize=8, pad=3)
        axes[r, 0].set_ylabel(title, fontsize=8.5, labelpad=8)

    fig.suptitle("Epidural and subdural hemorrhage, and the confusion between them",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.02, 0, 1, 0.975])
    out = FIGS / "qualitative_edh_sdh.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [figure] {out}")
    (RESULTS / "qualitative_figure_studies.json").write_text(
        json.dumps({k: [{"study": s, "score": v} for s, v in vv]
                    for k, vv in chosen.items()}, indent=2))
    OUT["figure_studies"] = {k: [s for s, _ in v] for k, v in chosen.items()}
    OUT["figure_slice_selection"] = ("model-scored slice" if has_slice_preds
                                     else "blood-mask fallback")
    print("\n   INSPECT THE FIGURE BEFORE USING IT. Every panel should show brain")
    print("   parenchyma at the level of the lateral ventricles or above, not the")
    print("   skull base. If the rows still look like skull base, tell me and I")
    print("   will restrict selection by slice position instead.")


# ============================================================ 2. WALL-CLOCK
def collect_timings():
    print("\n[2/4] training wall-clock from history.csv and logs")
    specs = [("ConvNeXt-tiny", f"stage1_fold*_logit_adjusted_{PROFILE}"),
             ("ResNet50", f"stage1_fold*_logit_adjusted_resnet50_{PROFILE}"),
             ("EfficientNet-B4", f"stage1_fold*_logit_adjusted_tfefficientnetb4_{PROFILE}"),
             ("DenseNet-161", f"stage1_fold*_logit_adjusted_densenet161_{PROFILE}"),
             ("focal", f"stage1_fold*_focal_{PROFILE}"),
             ("weighted_bce", f"stage1_fold*_weighted_bce_{PROFILE}")]
    t = {}
    for name, pat in specs:
        secs, folds = [], 0
        for d in sorted(glob.glob(str(S1_DIR / pat))):
            h = Path(d) / "history.csv"
            if not h.exists():
                continue
            folds += 1
            df = pd.read_csv(h)
            if "epoch_seconds" in df.columns:
                secs.append(df["epoch_seconds"].sum())
        if folds:
            t[name] = dict(folds=folds,
                           hours_per_fold=round(np.mean(secs) / 3600, 2) if secs else None,
                           total_hours=round(sum(secs) / 3600, 2) if secs else None)
    # epoch times are printed to stdout, not always stored -- try the logs
    logpat = os.path.expanduser("~/s1_*.log")
    log_secs = {}
    for lg in glob.glob(logpat):
        v = [int(m) for m in re.findall(r"\((\d+)s\)", open(lg, errors="ignore").read())]
        if v:
            log_secs[os.path.basename(lg)] = round(sum(v) / 3600, 2)
    OUT["training_hours"] = t
    OUT["training_hours_from_logs"] = log_secs
    for k, v in t.items():
        print(f"   {k:18} folds={v['folds']}  {v['total_hours']} h total")
    if log_secs:
        print(f"   recovered from {len(log_secs)} log files:")
        for k, v in sorted(log_secs.items()):
            print(f"     {k}: {v} h")
    if not any(v.get("total_hours") for v in t.values()) and not log_secs:
        print("   NOT RECOVERABLE from history.csv (no epoch_seconds column) or logs.")
        print("   Table S5 timing rows will have to stay [?] or be filled from memory.")


# ============================================================ 3. LATENCY
def bench_latency(reps=30):
    print("\n[3/4] inference latency")
    import torch, timm
    setup_hardware()
    ck = S1_DIR / f"stage1_fold0_logit_adjusted_{PROFILE}" / "best.pt"
    if not ck.exists():
        print("   no checkpoint; skipping"); return
    d = torch.load(ck, map_location="cpu", weights_only=False)
    m = timm.create_model(d["cfg"]["backbone"], pretrained=False, num_classes=6)
    sd = {k.replace("net.", "", 1): v for k, v in d["model"].items() if k.startswith("net.")}
    m.load_state_dict(sd, strict=False)
    m = m.to(DEVICE).eval().to(memory_format=torch.channels_last)

    idx = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    n_sl = int(idx[idx["ok"]].groupby("study").size().median())
    meta = json.load(open(MEMMAP_DIR / "memmap_meta.json"))
    mm = np.memmap(MEMMAP_DIR / meta["dat"], dtype=np.uint8, mode="r",
                   shape=(meta["n"], meta["c"], meta["h"], meta["w"]))

    x = torch.from_numpy(np.asarray(mm[:32])).to(DEVICE).float().div_(255.)
    x = x.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        for _ in range(5):
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                m(x)
        torch.cuda.synchronize()
        per_slice_ms = (time.perf_counter() - t0) / (reps * 32) * 1000

    OUT["latency"] = dict(
        median_slices_per_study=n_sl,
        per_slice_ms=round(per_slice_ms, 2),
        per_study_single_s=round(per_slice_ms * n_sl / 1000, 2),
        per_study_ensemble5_s=round(per_slice_ms * n_sl * 5 / 1000, 2),
        batch=32, resolution=TRAIN_RES, precision=str(AMP_DTYPE), device=str(DEVICE))
    print(f"   {per_slice_ms:.2f} ms/slice, median {n_sl} slices/study")
    print(f"   -> {per_slice_ms*n_sl/1000:.2f} s/study single, "
          f"{per_slice_ms*n_sl*5/1000:.2f} s/study 5-model ensemble")


# ============================================================ 4. MISC
def collect_misc():
    print("\n[4/4] remaining values")
    p = MEMMAP_DIR / f"slices_{meta_h()}.dat" if False else None
    d = MEMMAP_DIR / "memmap_meta.json"
    if d.exists():
        m = json.load(open(d))
        gb = m["n"] * m["c"] * m["h"] * m["w"] / 1e9
        OUT["memmap"] = dict(slices=m["n"], resolution=m["h"], size_gb=round(gb, 1))
        print(f"   memmap: {m['n']:,} slices, {gb:.0f} GB")
    sm = MAN_DIR / "split_meta.json"
    if sm.exists():
        OUT["split"] = json.load(open(sm))
        print(f"   split hash: {OUT['split']['assignment_sha256_16']}")
    # human comparator already in the reference set
    OUT["notes"] = {
        "human_baseline": "Angkurawaranon et al. 2023 report resident EDH "
                          "sensitivity 0.71 and model 0.72 on their cohort; "
                          "usable as a human comparator sentence in Sec 5.5.",
        "preprocessing_time": "not logged; fill from memory or rerun 00_preprocess",
    }


def meta_h():
    return json.load(open(MEMMAP_DIR / "memmap_meta.json"))["h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["figure", "timings", "latency", "misc"])
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()

    if a.only in (None, "figure"):   build_figure(a.n)
    if a.only in (None, "timings"):  collect_timings()
    if a.only in (None, "latency") and not a.skip_latency:
        try: bench_latency()
        except Exception as e: print(f"   latency failed: {type(e).__name__}: {e}")
    if a.only in (None, "misc"):     collect_misc()

    out = RESULTS / "paper_collected_values.json"
    out.write_text(json.dumps(OUT, indent=2, default=str))
    print(f"\n[written] {out}")
    print("\nPaste that JSON back and I will fill Table S5 and the remaining [?].")


if __name__ == "__main__":
    main()