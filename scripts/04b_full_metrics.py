#!/usr/bin/env python3
"""
04b_full_metrics.py  —  ICH pipeline, step 4b:  publication-comparable metrics

Your comparator papers report different metrics: WsGSA uses F1/recall/accuracy;
the RSNA challenge uses weighted logloss; the meta-analysis uses sensitivity/
specificity/AUC. To line your results up against ALL of them, this computes, per
subtype: AUC, AP, sensitivity(recall), specificity, precision, F1, F2, accuracy,
plus the official RSNA weighted logloss — with bootstrap 95% CIs.

CORRECTNESS (this is what makes it survive review):
  * F1/F2/sensitivity/specificity depend on a THRESHOLD. Choosing the threshold on
    the test set is leakage. We split the pooled predictions in half: select each
    class's threshold to maximize F1 on the SELECTION half, then REPORT all
    threshold metrics on the held-out EVALUATION half. Threshold-free metrics
    (AUC, AP, WLL) use the full pooled set.
  * This mirrors how careful papers report operating-point metrics and prevents
    the optimistic bias that inflated your earlier F2 numbers.

Usage:
  python 04b_full_metrics.py --runs stage2_fold{F}_conditional_laptop --folds 0
  python 04b_full_metrics.py --runs stage2_fold{F}_conditional_workstation --folds 0 1 2 3 4
"""
import argparse, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, fbeta_score)
warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("ICH_ROOT", "/path/to/rsna-intracranial-hemorrhage-detection"))
RUNS = ROOT / "stage2_runs"
SUBTYPES = ["epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]
ALL_COLS = SUBTYPES + ["any"]
RSNA_W = [1, 1, 1, 1, 1, 2]      # official: 'any' weighted 2x
RNG = np.random.default_rng(42)


def load_pooled(tpl, folds):
    P, T = [], []
    for f in folds:
        npz = RUNS / tpl.replace("{F}", str(f)) / "val_predictions.npz"
        if not npz.exists():
            print(f"  MISSING: {npz}"); continue
        d = np.load(npz, allow_pickle=True)
        P.append(d["probs"]); T.append(d["targets"])
    if not P: raise SystemExit("no predictions found")
    return np.concatenate(P), np.concatenate(T)


def select_threshold(y, p, criterion="f1", min_precision=0.10):
    """Threshold selected on the selection split, by the chosen criterion.
       f1  : maximize F1 (balanced)
       f2  : maximize F2 (recall-weighted — clinically favored for can't-miss bleeds)
       rec_at_prec : maximize recall subject to precision >= min_precision
                     (report high sensitivity at an acceptable precision floor)"""
    best_t, best = 0.5, -1
    for t in np.linspace(0.01, 0.99, 99):
        pred = (p >= t).astype(int)
        if criterion == "f1":
            s = f1_score(y, pred, zero_division=0)
        elif criterion == "f2":
            s = fbeta_score(y, pred, beta=2, zero_division=0)
        elif criterion == "rec_at_prec":
            tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum()); fn = int(((1 - pred) & y).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            s = rec if prec >= min_precision else -1
        else:
            raise ValueError(criterion)
        if s > best:
            best, best_t = s, t
    return best_t


def threshold_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum())
    fn = int(((1 - pred) & y).sum()); tn = int(((1 - pred) & (1 - y)).sum())
    return dict(
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
        precision=tp / max(tp + fp, 1),
        f1=f1_score(y, pred, zero_division=0),
        f2=fbeta_score(y, pred, beta=2, zero_division=0),
        accuracy=(tp + tn) / len(pred),
        threshold=t,
    )


def weighted_logloss(probs, targets, weights):
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    ll = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    w = np.array(weights)
    return float((ll * w[None, :]).sum() / (w.sum() * len(targets)))


def boot(y, p, fn, n=1000):
    idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s): continue
        try: v.append(fn(y[s], p[s]))
        except Exception: pass
    if not v: return float("nan"), float("nan"), float("nan")
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--criterion", default="f1", choices=["f1", "f2", "rec_at_prec"],
                    help="threshold-selection objective (f2 = recall-weighted, clinical)")
    ap.add_argument("--min_precision", type=float, default=0.10,
                    help="precision floor for --criterion rec_at_prec")
    a = ap.parse_args()

    probs, targets = load_pooled(a.runs, a.folds)
    N = len(targets)
    # split for non-leaking threshold selection
    perm = RNG.permutation(N); half = N // 2
    sel, ev = perm[:half], perm[half:]
    print("=" * 78)
    print(f"FULL METRICS (pooled OOF, {N:,} studies)  —  {a.runs}")
    print(f"threshold criterion: {a.criterion}"
          + (f" (min_precision={a.min_precision})" if a.criterion == "rec_at_prec" else ""))
    print(f"threshold selected on {len(sel):,} studies, reported on held-out {len(ev):,}")
    print("=" * 78)
    hdr = f"{'class':16s} {'AUC':13s} {'AP':13s} {'Sens':6s} {'Spec':6s} {'Prec':6s} {'F1':6s} {'F2':6s}"
    print(hdr); print("-" * 78)

    rows = {}
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue
        # threshold-free on full pooled
        auc, alo, ahi = boot(y, p, roc_auc_score)
        apv, plo, phi = boot(y, p, average_precision_score)
        # threshold selected on sel, metrics on ev
        t = select_threshold(y[sel], p[sel], criterion=a.criterion, min_precision=a.min_precision)
        m = threshold_metrics(y[ev], p[ev], t)
        rows[name] = dict(auc=auc, ap=apv, **m)
        print(f"{name:16s} {auc:.3f}[{alo:.2f},{ahi:.2f}] {apv:.3f}[{plo:.2f},{phi:.2f}] "
              f"{m['sensitivity']:.3f}  {m['specificity']:.3f}  {m['precision']:.3f}  "
              f"{m['f1']:.3f}  {m['f2']:.3f}")

    print("-" * 78)
    macro_auc = np.mean([rows[c]["auc"] for c in rows])
    macro_ap = np.mean([rows[c]["ap"] for c in rows])
    macro_f1 = np.mean([rows[c]["f1"] for c in rows])
    macro_sens = np.mean([rows[c]["sensitivity"] for c in rows])
    wll = weighted_logloss(probs, targets, RSNA_W)
    print(f"{'MACRO':16s} AUC={macro_auc:.4f}  AP={macro_ap:.4f}  "
          f"F1={macro_f1:.4f}  Sens={macro_sens:.4f}")
    print(f"{'RSNA weighted logloss':30s} = {wll:.4f}  (lower is better)")
    print("=" * 78)
    print("\nMinority-class focus (EDH):")
    e = rows.get("epidural")
    if e:
        print(f"  AUC={e['auc']:.3f}  AP={e['ap']:.3f}  Sensitivity={e['sensitivity']:.3f}  "
              f"Precision={e['precision']:.3f}  F1={e['f1']:.3f}  F2={e['f2']:.3f}")
        print(f"  (threshold={e['threshold']:.3f} selected on held-out split)")


if __name__ == "__main__":
    main()