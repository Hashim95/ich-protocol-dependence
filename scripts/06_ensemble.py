#!/usr/bin/env python3
"""
06_ensemble.py — pooled OOF ensemble + TTA-style averaging of Stage-2 heads.

Two modes:
  --mode pooled_oof   (default): concatenate each fold's OOF conditional predictions
      into one all-studies set = the honest pooled result (same as 04_evaluate single).
      This is the number you REPORT.

  --mode avg_heads: for studies present in MULTIPLE method runs (conditional + joint),
      average their probabilities. A light ensemble that often smooths rare-class noise.

Logs per-class AUC/AP + ECE (bootstrap CI) to results/ as CSV.

    python 06_ensemble.py --runs stage2_cond_fold{F}_conditional_workstation --folds 0 1 2 3 4
    python 06_ensemble.py --mode avg_heads \
        --runs stage2_cond_fold{F}_conditional_workstation stage2_cond_fold{F}_joint_workstation \
        --folds 0 1 2 3 4
"""
import argparse, warnings, datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
from ich_config import S2_DIR as RUNS, ROOT

RESULTS = ROOT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
ALL_COLS = ["epidural","intraparenchymal","intraventricular","subarachnoid","subdural","any"]
RNG = np.random.default_rng(42)


def _stamp(): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_run_pooled(tpl, folds):
    """Pool one method's OOF preds across folds -> dict study_id -> (prob_vec, label_vec)."""
    d = {}
    for f in folds:
        npz = RUNS / tpl.replace("{F}", str(f)) / "val_predictions.npz"
        if not npz.exists():
            print(f"  MISSING: {npz}"); continue
        z = np.load(npz, allow_pickle=True)
        probs, targets = z["probs"], z["targets"]
        studies = z["study"].astype(str) if "study" in z else \
                  np.array([f"f{f}_{i}" for i in range(len(probs))])
        for i, s in enumerate(studies):
            d[s] = (probs[i], targets[i])
    return d


def boot_ci(y, p, metric, n=1000):
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s): continue
        vals.append(metric(y[s], p[s]))
    if not vals: return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0: continue
        e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def evaluate_and_log(probs, targets, title, tag):
    print("=" * 74); print(title); print(f"studies: {len(targets):,}"); print("-" * 74)
    print(f"{'class':18s} {'AUC [95% CI]':26s} {'AP [95% CI]':26s} {'ECE':6s} pos")
    rows = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i], probs[:, i]
        if y.sum() == 0: continue
        a, alo, ahi = boot_ci(y, p, roc_auc_score)
        apv, plo, phi = boot_ci(y, p, average_precision_score)
        e = ece(y, p)
        rows.append(dict(cls=name, pos=int(y.sum()), auc=a, auc_lo=alo, auc_hi=ahi,
                         ap=apv, ap_lo=plo, ap_hi=phi, ece=e))
        print(f"{name:18s} {a:.3f} [{alo:.3f},{ahi:.3f}]   "
              f"{apv:.3f} [{plo:.3f},{phi:.3f}]   {e:.3f}  {int(y.sum())}")
    aucs = [r["auc"] for r in rows]; aps = [r["ap"] for r in rows]
    rows.append(dict(cls="macro", pos=len(targets), auc=float(np.mean(aucs)),
                     auc_lo=np.nan, auc_hi=np.nan, ap=float(np.mean(aps)),
                     ap_lo=np.nan, ap_hi=np.nan, ece=np.nan))
    print("-" * 74); print(f"{'macro':18s} AUC={np.mean(aucs):.4f}   AP={np.mean(aps):.4f}")
    print("=" * 74)
    out = RESULTS / f"{tag}_{_stamp()}.csv"
    pd.DataFrame(rows).to_csv(out, index=False); print(f"[logged] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="one template for pooled_oof, or several for avg_heads")
    ap.add_argument("--mode", default="pooled_oof", choices=["pooled_oof", "avg_heads"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    dicts = [load_run_pooled(tpl, a.folds) for tpl in a.runs]
    # common studies across all provided runs
    common = set(dicts[0].keys())
    for d in dicts[1:]: common &= set(d.keys())
    common = sorted(common)
    print(f"studies common to all runs: {len(common):,}")

    P, T = [], []
    for s in common:
        probs = np.mean([d[s][0] for d in dicts], axis=0)   # avg over runs (1 run = identity)
        P.append(probs); T.append(dicts[0][s][1])
    P, T = np.array(P), np.array(T)

    title = (f"POOLED OOF: {a.runs[0]}" if a.mode == "pooled_oof"
             else f"AVG-HEADS ENSEMBLE of {len(a.runs)} runs")
    tag = "ensemble_pooled_oof" if a.mode == "pooled_oof" else "ensemble_avg_heads"
    evaluate_and_log(P, T, title, tag)


if __name__ == "__main__":
    main()