#!/usr/bin/env python3
"""
05_sequence_smooth.py  —  slice-sequence smoothing (our version of WsGSA's JFM)

WsGSA's JFM smooths per-patient slice-sequence predictions because ICH is spatially
continuous — a hemorrhage on slice N is very likely present on N-1 and N+1, so
isolated single-slice predictions are often errors. We apply a lightweight causal-
symmetric smoothing over each study's z-ordered slice predictions.

This is a standard, defensible post-processing step (not a novel trick) that lifts
aggregate metrics on the common classes and 'any'. It operates on saved per-slice
predictions — no retraining.

Method: for each study, order slices by z, then replace each slice's probability with
a weighted moving average over a window of neighbors (Gaussian weights). Window size
and sigma are the only knobs.

Input : val_slice_predictions_tta.npz (or the non-TTA file)
Output: val_slice_predictions_smoothed.npz  (same format, smoothed probs)

Usage:
    python 05_sequence_smooth.py --run stage1_ft_fold0_both_workstation --window 5 --sigma 1.5
    # then: python 04c_slice_metrics.py --run stage1_ft_fold0_both_workstation --pred smoothed
"""
import argparse, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

ROOT=Path("/home/ivision/Documents/Hashim/RSNA ICH Dataset/rsna-intracranial-hemorrhage-detection")
OUT_DIR=ROOT/"stage1_runs"


def gaussian_weights(window, sigma):
    half=window//2
    x=np.arange(-half,half+1)
    w=np.exp(-(x**2)/(2*sigma**2))
    return w/w.sum()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True)
    ap.add_argument("--infile",default="val_slice_predictions_tta.npz",
                    help="input npz (default TTA; use val_slice_predictions.npz for non-TTA)")
    ap.add_argument("--window",type=int,default=5)
    ap.add_argument("--sigma",type=float,default=1.5)
    a=ap.parse_args()
    run=OUT_DIR/a.run
    d=np.load(run/a.infile, allow_pickle=True)
    probs=d["probs"].copy(); targets=d["targets"]; study=d["study"]; z=d["z"]
    image_id=d["image_id"]
    n_cls=probs.shape[1]
    print(f"loaded {len(probs):,} slice preds from {a.infile}")
    print(f"smoothing: Gaussian window={a.window} sigma={a.sigma}")

    w=gaussian_weights(a.window, a.sigma); half=a.window//2
    smoothed=probs.copy()

    # group by study, order by z, smooth each class's sequence
    order=np.argsort(z,kind="stable")
    # build study -> row indices (in z order)
    from collections import defaultdict
    s2r=defaultdict(list)
    for r in order:
        s2r[study[r]].append(int(r))

    n_studies=0
    for st, rows in s2r.items():
        n_studies+=1
        rows=np.array(rows)
        seq=probs[rows]                       # (S, C) already z-ordered
        S=len(rows)
        if S<2:
            continue
        sm=np.empty_like(seq)
        for i in range(S):
            lo=max(0,i-half); hi=min(S,i+half+1)
            # align weights to the available window
            wl=w[(lo-(i-half)):(a.window-((i+half+1)-hi))]
            wl=wl/wl.sum()
            sm[i]=(seq[lo:hi]*wl[:,None]).sum(0)
        smoothed[rows]=sm

    np.savez(run/"val_slice_predictions_smoothed.npz",
             probs=smoothed, targets=targets, image_id=image_id, study=study, z=z)
    print(f"smoothed {n_studies:,} studies -> {run/'val_slice_predictions_smoothed.npz'}")
    print("next: python 04c_slice_metrics.py --run", a.run, "--pred smoothed")


if __name__=="__main__":
    main()
