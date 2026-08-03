#!/usr/bin/env python3
"""watch_history.py — live view of a Stage-1/Stage-2 run's history.csv.
Shows EDH AUC/AP + macro each epoch, flags the best, and whether EDH is still
climbing or overfitting away. Read-only; safe to run while training.

    python watch_history.py --run stage1_fold1_logit_adjusted_workstation
    python watch_history.py --run stage1_fold1_logit_adjusted_workstation --watch
"""
import argparse, time, os
from pathlib import Path
import pandas as pd
from ich_config import S1_DIR, S2_DIR

def find_run(name):
    for base in (S1_DIR, S2_DIR):
        p = base / name / "history.csv"
        if p.exists(): return p
    return None

def show(csv):
    df = pd.read_csv(csv)
    cols = df.columns
    edh_auc = "epidural_auc" if "epidural_auc" in cols else None
    edh_ap  = "epidural_ap"  if "epidural_ap"  in cols else None
    print(f"\n{'ep':>3} {'macroAUC':>9} {'EDH_AUC':>8} {'EDH_AP':>7}   note")
    best_ap, best_ep = -1, -1
    for _, r in df.iterrows():
        ap = r.get(edh_ap, float('nan')) if edh_ap else float('nan')
        if ap == ap and ap > best_ap: best_ap, best_ep = ap, int(r["epoch"])
    for _, r in df.iterrows():
        ep = int(r["epoch"])
        ma = r.get("macro_auc", float('nan'))
        ea = r.get(edh_auc, float('nan')) if edh_auc else float('nan')
        ap = r.get(edh_ap,  float('nan')) if edh_ap  else float('nan')
        note = "<= best EDH_AP" if ep == best_ep else ""
        print(f"{ep:>3} {ma:>9.4f} {ea:>8.4f} {ap:>7.4f}   {note}")
    if best_ep > 0 and best_ep < len(df):
        print(f"\n  note: EDH_AP peaked at epoch {best_ep} ({best_ap:.4f}); "
              f"later epochs lower => watch for EDH overfitting.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--watch", action="store_true", help="refresh every 60s")
    a = ap.parse_args()
    while True:
        csv = find_run(a.run)
        if csv is None:
            print(f"no history.csv yet for {a.run} (run may not have finished epoch 1)")
        else:
            os.system("clear"); print(f"RUN: {a.run}")
            show(csv)
        if not a.watch: break
        time.sleep(60)

if __name__ == "__main__":
    main()