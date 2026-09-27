#!/usr/bin/env python3
"""
dump_stage2_attention.py — extract the per-class attention weights from a
trained Stage-2 conditional head, for the Fig-3(a) profile.

WHY THIS EXISTS
---------------
03c_conditional_subtype.py saves `val_predictions.npz` with probs, targets and
study ids — but not the attention weights. The Fig-3 caption claims the
epidural map "concentrates on a small number of sections rather than pooling
uniformly". That is a factual claim about the trained model, so the profile
drawn in the figure has to come from the model, not from plausible-looking
numbers typed into the TikZ source.

This script re-imports CondHead and StudyDS from 03c rather than
re-implementing them, so the weights it reports are the ones the head actually
used. If 03c's pooling changes, this follows automatically.

USAGE
-----
    python dump_stage2_attention.py \
        --features stage1_fold0_logit_adjusted_workstation \
        --fold 0 \
        --study ID_xxxxxxxx          # same study as the Fig-3 filmstrip

Omit --study and it picks the highest-scoring epidural-positive validation
study, which is the one worth drawing.

It prints a 12-value list on stdout. Paste that into the \\icoAttn{...} call in
figures/fig3_condhead_modern.tex, and record the study id in the caption or the
provenance file so the panel is traceable.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ich_config import MAN_DIR, S2_DIR, PROFILE, DEVICE, EDH_I


def load_03c():
    """Import 03c by path — the leading digit makes it un-importable by name."""
    path = Path("03c_conditional_subtype.py")
    if not path.exists():
        sys.exit("run this from the directory holding 03c_conditional_subtype.py")
    spec = importlib.util.spec_from_file_location("cond03c", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]          # 03c parses args only inside main(); be safe
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--method", default="conditional")
    ap.add_argument("--study", default=None)
    ap.add_argument("--from-provenance", default=None,
                    help="path to assets_provenance.json written by "
                         "make_figure_assets.py; takes the filmstrip study from "
                         "it so the profile and the strip describe the same scan")
    ap.add_argument("--cls", type=int, default=None,
                    help="attention map index; defaults to EDH_I from ich_config")
    ap.add_argument("--all-studies", action="store_true",
                    help="compute the peak-to-mean ratio for EVERY "
                         "class-positive validation study, report the "
                         "distribution, and emit the MEDIAN study rather than "
                         "an arbitrary one. Use this for anything that goes in "
                         "a paper: a single hand-picked map is an anecdote.")
    ap.add_argument("--max-terminal", type=float, default=25.0,
                    help="in --all-studies, exclude candidates whose last bin "
                         "exceeds this share of the peak. The terminal sections "
                         "are often near-empty, so a study with weight there is "
                         "unrepresentative even if its concentration is median.")
    ap.add_argument("--cells", type=int, default=16,
                    help="cells in the drawn strip; the 32 weights are pooled "
                         "down to this many so the bars stay legible. Use a "
                         "divisor of seq_len (8, 16) -- see the binning note.")
    ap.add_argument("--out", default=None,
                    help="figures/assets directory of the manuscript; the "
                         "profile is written there as attn_edh.tex for the "
                         "figures to \\input, so nothing is pasted by hand")
    a = ap.parse_args()
    cls = EDH_I if a.cls is None else a.cls

    if a.from_provenance:
        prov = json.loads(Path(a.from_provenance).expanduser().read_text())
        a.study = prov["filmstrip_study"]
        print(f"study taken from provenance: {a.study}")

    m = load_03c()

    # locate the run directory exactly as 03c names it
    ft = a.features.replace(f"stage1_fold{a.fold}_", "").replace(f"_{PROFILE}", "")
    suffix = "" if ft == "logit_adjusted" else f"_{ft}"
    run = S2_DIR / f"stage2_cond_fold{a.fold}_{a.method}{suffix}_{PROFILE}"
    ckpt = run / "best.pt"
    if not ckpt.exists():
        sys.exit(f"no checkpoint at {ckpt} — train fold {a.fold} first")

    feat, labels, study, z, fd = m.load_features(a.features)
    folds = pd.read_parquet(MAN_DIR / "folds.parquet")
    fo = dict(zip(folds["study"], folds["fold"]))

    order = np.argsort(z, kind="stable")
    s2r = {}
    for r in order:
        s2r.setdefault(study[r], []).append(int(r))

    st = np.array([s for s in s2r if s in fo])
    va = st[np.array([fo[s] for s in st]) == a.fold]

    # priors are only needed to build the module; the checkpoint overwrites them
    model = m.CondHead(fd, [0.5] * 6, method=a.method).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
    model.eval()

    ds = m.StudyDS(va, s2r, feat, labels, m.CFG["seq_len"])

    # choose the study to draw
    if a.study is not None:
        if a.study not in set(va):
            sys.exit(f"{a.study} is not in fold {a.fold}'s validation split")
        pick = int(np.where(va == a.study)[0][0])
    else:
        pos = [i for i in range(len(va))
               if labels[s2r[va[i]]].max(0)[cls] > 0]
        if not pos:
            sys.exit(f"no class-{cls}-positive study in fold {a.fold} validation")
        pick = pos[0]
    chosen = va[pick]

    def weights_for(i):
        """Binned, peak-normalised attention for validation study index i."""
        f, _y, msk = ds[i]
        with torch.no_grad():
            f = f.unsqueeze(0).to(DEVICE)
            msk = msk.unsqueeze(0).to(DEVICE)
            al = model.attn(f).masked_fill(msk.unsqueeze(-1) == 0, -1e4)
            aw = torch.softmax(al, dim=1)      # (1, seq, 6), softmax over sequence
        raw = aw[0, :, cls].cpu().numpy().astype(np.float64)
        # np.array_split, never reshape: seq_len is not always divisible by
        # --cells, and truncating silently drops the superior sections.
        binned = np.array([b.mean() for b in np.array_split(raw, a.cells)])
        return 100.0 * binned / (binned.max() + 1e-12)

    # ------------------------------------------------------------------ #
    # population mode
    # ------------------------------------------------------------------ #
    if a.all_studies:
        pos_idx = [i for i in range(len(va))
                   if labels[s2r[va[i]]].max(0)[cls] > 0]
        if not pos_idx:
            sys.exit(f"no class-{cls}-positive study in fold {a.fold} validation")
        rows = []
        for i in pos_idx:
            w_i = weights_for(i)
            rows.append(dict(study=str(va[i]),
                             ratio=float(w_i.max() / w_i.mean()),
                             terminal=float(w_i[-1]),      # last bin, 0-100
                             leading=float(w_i[0]),
                             weights=[float(v) for v in w_i]))
        ratios = np.array([r["ratio"] for r in rows])
        term   = np.array([r["terminal"] for r in rows])
        med    = float(np.median(ratios))

        # Pick the study that is typical on BOTH axes. Selecting on ratio alone
        # can land on a study that is median in concentration yet sits in the
        # tail for terminal-section weight -- which is exactly what happened
        # with ID_0ffdf20f08 (ratio 8.23, the median, but terminal bin 61 when
        # the median is 0 and only 3 of 57 studies exceed 50).
        elig = [r for r in rows if r["terminal"] <= a.max_terminal]
        if elig:
            e_ratios = np.array([r["ratio"] for r in elig])
            pick_row = elig[int(np.argmin(np.abs(e_ratios - med)))]
        else:
            pick_row = rows[int(np.argmin(np.abs(ratios - med)))]
            print("  [warning] no study under the terminal-weight threshold; "
                  "falling back to ratio alone")

        print(f"\nclass-{cls}-positive validation studies : {len(rows)}")
        print(f"peak/mean ratio  median {med:.2f}   "
              f"IQR [{np.percentile(ratios,25):.2f}, {np.percentile(ratios,75):.2f}]   "
              f"range [{ratios.min():.2f}, {ratios.max():.2f}]")
        print(f"studies with ratio < 2 (near-uniform)   : "
              f"{int((ratios < 2).sum())} / {len(rows)}")
        print(f"terminal-bin weight, median             : {np.median(term):.0f}"
              f"  (0 = none, 100 = equal to the peak)")
        print(f"studies with terminal bin > 50 of peak  : "
              f"{int((term > 50).sum())} / {len(rows)}"
              "   <- attention on the most superior sections")
        print(f"\neligible after terminal filter (<= {a.max_terminal:.0f}) : "
              f"{len(elig)} / {len(rows)}")
        print(f"study drawn in Fig 3(a) : {pick_row['study']}  "
              f"(ratio {pick_row['ratio']:.2f} vs median {med:.2f}, "
              f"terminal {pick_row['terminal']:.0f})")

        if a.out:
            adir = Path(a.out).expanduser(); adir.mkdir(parents=True, exist_ok=True)
            cells_str = ",".join(f"{v:.0f}" for v in pick_row["weights"])
            (adir / "attn_edh.tex").write_text(
                "% AUTO-GENERATED by dump_stage2_attention.py --all-studies\n"
                f"% median-ratio study of {len(rows)} class-{cls}-positive "
                f"fold-{a.fold} validation studies\n"
                f"\\renewcommand{{\\attnEDHlist}}{{{cells_str}}}\n"
                f"\\renewcommand{{\\attnEDHstudy}}{{{pick_row['study']}}}\n"
                f"\\renewcommand{{\\attnEDHratio}}{{{pick_row['ratio']:.1f}}}\n"
                f"\\renewcommand{{\\attnEDHmedian}}{{{med:.1f}}}\n"
                f"\\renewcommand{{\\attnEDHnstudies}}{{{len(rows)}}}\n"
                f"\\renewcommand{{\\attnEDHcells}}{{{a.cells}}}\n")
            (adir / "attn_study.txt").write_text(pick_row["study"] + "\n")
            (adir / "attn_summary.json").write_text(json.dumps(
                dict(fold=a.fold, cls=cls, n=len(rows), median_ratio=med,
                     iqr=[float(np.percentile(ratios,25)),
                          float(np.percentile(ratios,75))],
                     median_terminal=float(np.median(term)),
                     n_terminal_gt50=int((term > 50).sum()),
                     max_terminal_filter=a.max_terminal,
                     n_eligible=len(elig),
                     chosen=pick_row["study"], per_study=rows), indent=2))
            print(f"\nwrote {adir/'attn_edh.tex'}, attn_study.txt, attn_summary.json")
            print("Run make_figure_assets.py --study-file "
                  f"{adir/'attn_study.txt'} so the filmstrip matches.")
        return

    w = weights_for(pick)

    cells = ",".join(f"{v:.0f}" for v in w)

    print(f"\nstudy      : {chosen}")
    print(f"class      : {cls}  (0 = epidural under ALL_COLS)")
    print(f"seq_len    : {m.CFG['seq_len']}  pooled to {a.cells} cells")
    print(f"max/mean   : {w.max()/w.mean():.2f}x  "
          f"(a flat map would give 1.00x — this is the 'concentrates' claim)")
    ratio = float(w.max() / w.mean())

    # Write the profile as a LaTeX snippet the figures \input. This removes the
    # copy-paste step entirely and means the drawn profile cannot drift from
    # the checkpoint it came from.
    if a.out:
        adir = Path(a.out).expanduser()
        adir.mkdir(parents=True, exist_ok=True)
        tex = adir / "attn_edh.tex"
        tex.write_text(
            "% AUTO-GENERATED by dump_stage2_attention.py -- do not edit.\n"
            f"% study {chosen}, class {cls}, fold {a.fold}, {a.cells} cells\n"
            f"\\renewcommand{{\\attnEDHlist}}{{{cells}}}\n"
            f"\\renewcommand{{\\attnEDHstudy}}{{{chosen}}}\n"
            f"\\renewcommand{{\\attnEDHratio}}{{{ratio:.1f}}}\n"
            f"\\renewcommand{{\\attnEDHcells}}{{{a.cells}}}\n")
        print(f"\nwrote {tex}  -- the figures pick this up automatically")
    else:
        print(f"\n\\icoAttn{{{cells}}}{{icAmber}}{{...}}\n")

    out = run / "attention_fig3.json"
    out.write_text(json.dumps(dict(study=str(chosen), cls=cls, fold=a.fold,
                                   cells=a.cells, max_over_mean=ratio,
                                   weights=[float(v) for v in w]), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()