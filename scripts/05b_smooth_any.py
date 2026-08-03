#!/usr/bin/env python3
"""
05b_smooth_any.py — JFM-equivalent sequence smoothing for ANY prediction file.

WHY THIS EXISTS
---------------
WsGSA's published avg F1 of 0.746 INCLUDES their JFM smoother; their own
ablation attributes +3.9% to it, so WsGSA alone is approximately 0.707. Their
EDH F1 of 0.467 is very likely the smoothed figure too, and cannot be
decomposed from the published tables. Our re-implementation omits JFM by design.

Comparing our unsmoothed pipeline against their smoothed numbers understates us
and is a confound a reviewer will find. This script applies equivalent smoothing
to BOTH prediction sets so the comparison is smoothed-vs-smoothed.

WHY NOT 05_sequence_smooth.py
-----------------------------
That script hardcodes stage1_runs/ and requires a `z` array in the npz.
`10_wsgsa_baseline.py` saved only probs/targets/study — no z, no image_id — so
there is no slice ordering to smooth along. This version takes an explicit path
and RECONSTRUCTS z from memmap_index.parquet when it is absent, using the same
deterministic fold selection the evaluation DataLoader used (inner join on
folds.parquet, filter to the fold, shuffle=False, so row order is preserved).

The reconstruction is verified: it aborts if the recovered study sequence does
not match the study array stored in the npz. A silent misalignment would smooth
across the wrong slices and corrupt the comparison.

METHOD
------
Per study, order slices by geometric z, then replace each slice probability with
a Gaussian-weighted moving average over its neighbours. ICH is spatially
continuous — a hemorrhage on slice N is very likely present on N±1 — so isolated
single-slice predictions are usually errors. Standard post-processing, not a
novel contribution; the point is only that both methods receive it.

USAGE
-----
    # ours (has z already)
    python 05b_smooth_any.py --npz "$ICH_ROOT/stage1_runs/stage1_fold0_logit_adjusted_workstation/val_predictions.npz" --fold 0

    # WsGSA (z reconstructed)
    python 05b_smooth_any.py --npz "$ICH_ROOT/wsgsa_runs/wsgsa_fold0/val_predictions.npz" --fold 0

    # sweep the smoothing strength
    python 05b_smooth_any.py --npz ... --fold 0 --window 7 --sigma 2.0

Writes <name>_smoothed.npz beside the input and prints slice-level metrics at
threshold 0.5 before and after, so the smoothing delta is explicit.
"""
import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

from ich_config import MAN_DIR, MEMMAP_DIR, ALL_COLS


def gaussian_weights(window, sigma):
    half = window // 2
    x = np.arange(-half, half + 1)
    w = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return w / w.sum()


def reconstruct_z(study_arr, fold):
    """Recover per-slice z for a prediction file that did not store it.

    Rebuilds the exact validation dataframe the evaluation loader used:
    memmap_index -> keep ok -> inner join folds.parquet -> filter to `fold`.
    Order is preserved because the val DataLoader ran with shuffle=False.
    """
    idx = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    folds = pd.read_parquet(MAN_DIR / "folds.parquet")
    idx = idx[idx["ok"]].merge(folds[["study", "fold"]], on="study", how="inner")
    va = idx[idx.fold == fold]
    if len(va) != len(study_arr):
        raise SystemExit(
            f"\nCannot reconstruct z: rebuilt validation set has {len(va):,} rows "
            f"but the npz has {len(study_arr):,}.\n"
            f"The prediction file was not produced from this fold/split.")
    rebuilt = va["study"].values.astype(str)
    n_mismatch = int((rebuilt != study_arr.astype(str)).sum())
    if n_mismatch:
        raise SystemExit(
            f"\nCannot reconstruct z: {n_mismatch:,} study IDs differ between the "
            f"rebuilt order and the npz.\nSmoothing across misaligned slices would "
            f"silently corrupt the result. Re-run inference saving z explicitly.")
    print(f"  reconstructed z for {len(va):,} slices (order verified)")
    return va["z"].values.astype(np.float32)


def smooth(probs, study, z, window, sigma):
    w = gaussian_weights(window, sigma)
    half = window // 2
    out = probs.copy()
    order = np.lexsort((z, study))          # group by study, ascending z within
    bounds = np.flatnonzero(np.r_[True, study[order][1:] != study[order][:-1],
                                  True])
    n_studies = len(bounds) - 1
    for b in range(n_studies):
        rows = order[bounds[b]:bounds[b + 1]]
        if len(rows) < 2:
            continue
        seq = probs[rows]                                   # (n_slices, n_cls)
        pad = np.pad(seq, ((half, half), (0, 0)), mode="edge")
        sm = np.empty_like(seq)
        for k in range(len(seq)):
            sm[k] = (pad[k:k + window] * w[:, None]).sum(0)
        out[rows] = sm
    print(f"  smoothed {n_studies:,} studies (window={window}, sigma={sigma})")
    return out


def slice_metrics(P, Y, label):
    rows = []
    for i, c in enumerate(ALL_COLS):
        y, p = Y[:, i].astype(int), P[:, i]
        if y.sum() == 0:
            continue
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        se = tp / max(tp + fn, 1); pr = tp / max(tp + fp, 1)
        rows.append(dict(cls=c, pos=int(y.sum()),
                         AUC=float(roc_auc_score(y, p)),
                         AP=float(average_precision_score(y, p)),
                         sensitivity=se, precision=pr,
                         F1=2 * pr * se / max(pr + se, 1e-12)))
    df = pd.DataFrame(rows)
    avg = df.select_dtypes("number").mean().to_dict()
    avg["cls"] = "AVERAGE"; avg["pos"] = int(df["pos"].sum())
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    print(f"\n{label} — slice level, threshold 0.5")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to a val_predictions.npz")
    ap.add_argument("--fold", type=int, required=True,
                    help="fold this file came from (needed to reconstruct z)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--tag", default=None, help="label for the printed tables")
    a = ap.parse_args()

    src = Path(a.npz)
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    tag = a.tag or src.parent.name

    d = np.load(src, allow_pickle=True)
    P, Y = d["probs"], d["targets"]
    study = d["study"].astype(str)
    print(f"loaded {len(P):,} slice predictions from {src.parent.name}/{src.name}")
    print(f"  arrays present: {sorted(d.keys())}")

    if "z" in d.files:
        z = d["z"].astype(np.float32)
        print(f"  using stored z")
    else:
        print(f"  no z array — reconstructing from memmap_index (fold {a.fold})")
        z = reconstruct_z(study, a.fold)

    before = slice_metrics(P, Y, f"{tag} — BEFORE smoothing")
    Ps = smooth(P, study, z, a.window, a.sigma)
    after = slice_metrics(Ps, Y, f"{tag} — AFTER smoothing")

    out = src.parent / (src.stem + "_smoothed.npz")
    np.savez_compressed(out, probs=Ps, targets=Y, study=study, z=z)
    print(f"\n[saved] {out}")

    e_b = float(before[before.cls == "epidural"]["F1"].iloc[0])
    e_a = float(after[after.cls == "epidural"]["F1"].iloc[0])
    a_b = float(before[before.cls == "AVERAGE"]["F1"].iloc[0])
    a_a = float(after[after.cls == "AVERAGE"]["F1"].iloc[0])
    print(f"\nDELTA  EDH F1 {e_b:.4f} -> {e_a:.4f}  ({e_a-e_b:+.4f})")
    print(f"DELTA  avg F1 {a_b:.4f} -> {a_a:.4f}  ({a_a-a_b:+.4f})")
    print(f"\n(WsGSA attribute +3.9% avg F1 to their JFM smoother.)")

    cmp = pd.concat([before.assign(state="before"), after.assign(state="after")])
    csv = src.parent / f"smoothing_comparison_{tag}.csv"
    cmp.to_csv(csv, index=False)
    print(f"[logged] {csv}")


if __name__ == "__main__":
    main()