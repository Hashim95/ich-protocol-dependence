#!/usr/bin/env python3
"""
diagnose_minority_fp.py  —  validate (or kill) the hard-negative hypothesis

Before building any new method, answer with DATA:
  Q1. What FPR on negatives corresponds to our EDH_AP? (confirm the lever is FP suppression)
  Q2. Are EDH false-positives concentrated on confusable classes (SDH/SAH)?
      If yes -> a hard-negative-aware / cross-class method has real signal to exploit.
      If no  -> abandon that idea; EDH FPs are diffuse and we need a different angle.

Reads the val_predictions.npz written by 03b for a given fold, plus the study manifest
for the true multi-label targets.

Usage:
    python diagnose_minority_fp.py --run stage2_fold0_fixed_la_laptop
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

from ich_config import S2_DIR
SUBTYPES=["epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]
ALL=SUBTYPES+["any"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); a=ap.parse_args()
    d=np.load(S2_DIR / a.run / "val_predictions.npz", allow_pickle=True)
    probs, targets, study = d["probs"], d["targets"], d["study"]
    E=0  # epidural index
    p_edh=probs[:,E]; y_edh=targets[:,E].astype(int)
    N=len(y_edh); pos=int(y_edh.sum()); neg=N-pos
    print(f"val: {N} studies | EDH pos={pos} ({pos/N:.2%}) neg={neg}")
    print()

    # Q1: at the threshold giving ~80% recall, how many false positives, and precision?
    order=np.argsort(-p_edh)
    thr_grid=np.quantile(p_edh, np.linspace(0.5,0.999,50))
    print("threshold sweep (recall -> precision, #FP):")
    for r_target in [0.5,0.7,0.8,0.9]:
        # find lowest threshold achieving >= r_target recall
        best=None
        for t in sorted(set(p_edh),reverse=False):
            pred=(p_edh>=t).astype(int)
            tp=int(((pred==1)&(y_edh==1)).sum()); fp=int(((pred==1)&(y_edh==0)).sum())
            rec=tp/max(pos,1); prec=tp/max(tp+fp,1)
            if rec>=r_target: best=(t,rec,prec,fp)
        if best:
            t,rec,prec,fp=best
            print(f"  recall={rec:.2f}  precision={prec:.3f}  #FP={fp}  (thr={t:.3f})")
    print()

    # Q2: among EDH false positives at 80% recall, what other classes are present?
    # pick threshold for ~80% recall
    t80=None
    for t in sorted(set(p_edh)):
        pred=(p_edh>=t).astype(int); tp=int(((pred==1)&(y_edh==1)).sum())
        if tp/max(pos,1)>=0.8: t80=t
    pred=(p_edh>=t80).astype(int)
    fp_mask=(pred==1)&(y_edh==0)
    print(f"among {int(fp_mask.sum())} EDH FALSE POSITIVES (at ~80% recall):")
    base_rates={}
    for j,name in enumerate(ALL):
        if name in ("epidural","any"): continue
        base = targets[:,j].mean()
        infp = targets[fp_mask,j].mean() if fp_mask.sum()>0 else 0
        ratio = infp/max(base,1e-6)
        base_rates[name]=(infp,base,ratio)
        flag=" <-- ENRICHED" if ratio>1.5 else ""
        print(f"  {name:18s} present in {infp:.1%} of FPs vs {base:.1%} base rate  (x{ratio:.2f}){flag}")
    print()
    enriched=[k for k,(i,b,r) in base_rates.items() if r>1.5]
    if enriched:
        print(f"CONFIRMED: EDH false positives are enriched for {enriched}.")
        print("-> A hard-negative-aware / cross-class FP-suppression method has real signal.")
    else:
        print("NOT confirmed: EDH FPs are not concentrated on specific classes.")
        print("-> Different angle needed; do not build the cross-class suppressor.")

if __name__=="__main__":
    main()
