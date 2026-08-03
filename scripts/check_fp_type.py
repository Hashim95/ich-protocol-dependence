#!/usr/bin/env python3
"""
check_fp_type.py — decide the method design from data.

Of the EDH false positives, how many are on OTHER-hemorrhage studies
(subtype confusion) vs PURE-NEGATIVE studies (true false alarm)?
That ratio picks the novel method form.

Usage:
    python check_fp_type.py --run stage2_fold0_fixed_la_laptop
"""
import argparse, numpy as np
ROOT="/home/ivision/Documents/Hashim/RSNA ICH Dataset/rsna-intracranial-hemorrhage-detection"

ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); a=ap.parse_args()
d=np.load(f"{ROOT}/stage2_runs/{a.run}/val_predictions.npz", allow_pickle=True)
probs,targets=d["probs"],d["targets"]
E,ANY=0,5
p_edh=probs[:,E]; y_edh=targets[:,E].astype(int); y_any=targets[:,ANY].astype(int)
pos=int(y_edh.sum())

t80=None
for t in sorted(set(p_edh)):
    if (((p_edh>=t)&(y_edh==1)).sum())/max(pos,1)>=0.8: t80=t
pred=(p_edh>=t80).astype(int)
fp=(pred==1)&(y_edh==0); nfp=int(fp.sum())
on_h=int((fp&(y_any==1)).sum()); on_n=int((fp&(y_any==0)).sum())
print(f"EDH false positives at ~80% recall: {nfp}")
print(f"  on OTHER-hemorrhage studies (subtype confusion): {on_h} ({on_h/nfp:.1%})")
print(f"  on PURE-NEGATIVE studies (true false alarm):      {on_n} ({on_n/nfp:.1%})")
print()
if on_h/nfp>0.6:
    print("=> Subtype confusion dominates. Design: CONDITIONAL detect-then-subtype.")
elif on_n/nfp>0.6:
    print("=> Pure false alarms dominate. Design: margin-vs-negatives suppressor.")
else:
    print("=> MIXED. Design: hybrid conditional subtype head + margin term.")
