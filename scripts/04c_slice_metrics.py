#!/usr/bin/env python3
"""
04c_slice_metrics.py  —  SLICE-LEVEL evaluation for fair comparison vs WsGSA

The base paper (Zhang et al., WsGSA, IEEE TCDS 2023) reports PER-IMAGE (slice-level)
metrics: threshold 0.5 (their eq. 9), per-class recall/precision/F1 (eqs 13-15),
averaged over classes. Their headline: avg F1 74.6%, EDH F1 46.7%, avg ACC 98.1%.

Your Stage-2 metrics are STUDY-level and NOT comparable to theirs. This script
evaluates Stage-1 PER-SLICE predictions two ways:

  [A] THEIR PROTOCOL  : threshold 0.5, macro-averaged F1/recall/precision/ACC/TPR/TNR
                        -> directly comparable to their Table VII / IV.
  [B] RIGOROUS        : threshold selected on a held-out split (no leakage),
                        + bootstrap 95% CIs, + per-class breakdown.
                        -> the modern-standard version reviewers expect.

Reporting BOTH is the point: [A] gives the fair head-to-head; [B] shows you meet
current standards (patient-disjoint upstream, CIs, honest thresholds) that WsGSA
does not. Their split is slice-level (leakage risk); yours is patient-grouped.

Input: val_slice_predictions.npz written by 02c (Stage-1).

Usage:
    python 04c_slice_metrics.py --run stage1_ft_fold0_both_laptop
    python 04c_slice_metrics.py --run stage1_supcon_fold0_workstation   # after Monday
"""
import argparse, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, fbeta_score)
warnings.filterwarnings("ignore")

ROOT = Path("/home/ivision/Documents/Hashim/RSNA ICH Dataset/rsna-intracranial-hemorrhage-detection")
S1 = ROOT / "stage1_runs"
# WsGSA class order for side-by-side: they list SAH,IPH,SDH,EDH,IVH + any.
# We keep our order and label clearly.
SUBTYPES = ["epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]
ALL_COLS = SUBTYPES + ["any"]
# WsGSA reported EDH F1 = 0.467, avg F1 = 0.746, avg ACC = 0.981 (for reference)
WSGSA = {"epidural":0.467, "intraparenchymal":None, "intraventricular":None,
         "subarachnoid":None, "subdural":0.755, "any":0.895, "avg_f1":0.746, "avg_acc":0.981}
RNG = np.random.default_rng(42)


def per_class_at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum())
    fn = int(((1 - pred) & y).sum()); tn = int(((1 - pred) & (1 - y)).sum())
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 0.0 if (prec + recall) == 0 else 2 * prec * recall / (prec + recall)
    f2 = fbeta_score(y, pred, beta=2, zero_division=0)
    acc = (tp + tn) / len(pred)
    tnr = tn / max(tn + fp, 1)
    return dict(recall=recall, precision=prec, f1=f1, f2=f2, acc=acc, tnr=tnr,
                tp=tp, fp=fp, fn=fn, tn=tn)


def select_f1_threshold(y, p):
    best_t, best = 0.5, -1
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best: best, best_t = f, t
    return best_t


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
    ap.add_argument("--run", required=True)
    ap.add_argument("--pred", default="base", choices=["base", "tta", "smoothed"],
                    help="which predictions: base / tta / smoothed")
    a = ap.parse_args()
    fname = {"base": "val_slice_predictions.npz",
             "tta": "val_slice_predictions_tta.npz",
             "smoothed": "val_slice_predictions_smoothed.npz"}[a.pred]
    npz = S1 / a.run / fname
    if not npz.exists():
        raise SystemExit(f"not found: {npz}\n(generate it first)")
    d = np.load(npz, allow_pickle=True)
    probs, targets = d["probs"], d["targets"]
    N = len(targets)

    # =============== [A] WsGSA PROTOCOL: threshold 0.5, macro-average ===============
    print("=" * 82)
    print(f"[A] WsGSA PROTOCOL (slice-level, threshold=0.5)  —  {a.run} [{a.pred}]")
    print(f"    directly comparable to Zhang et al. Table VII (EDH F1=0.467, avg F1=0.746)")
    print(f"    {N:,} slices")
    print("-" * 82)
    print(f"{'class':18s} {'recall':7s} {'prec':7s} {'F1':7s} {'ACC':7s}  {'WsGSA F1':9s} {'pos':>6s}")
    f1s, accs, recs, tnrs = [], [], [], []
    for i, name in enumerate(ALL_COLS):
        y = targets[:, i].astype(int); p = probs[:, i]
        if y.sum() == 0: continue
        m = per_class_at_threshold(y, p, 0.5)
        f1s.append(m["f1"]); accs.append(m["acc"]); recs.append(m["recall"]); tnrs.append(m["tnr"])
        ref = WSGSA.get(name)
        refs = f"{ref:.3f}" if ref else "  -  "
        print(f"{name:18s} {m['recall']:.3f}   {m['precision']:.3f}   {m['f1']:.3f}   "
              f"{m['acc']:.3f}    {refs:9s} {int(y.sum()):>6}")
    print("-" * 82)
    print(f"{'AVERAGE':18s} recall={np.mean(recs):.3f}  F1={np.mean(f1s):.4f}  "
          f"ACC={np.mean(accs):.4f}  TPR={np.mean(recs):.3f}  TNR={np.mean(tnrs):.3f}")
    print(f"{'WsGSA reference':18s} {'':13s} F1=0.7460   ACC=0.9810")
    print("=" * 82)

    # =============== [B] RIGOROUS: val-selected thresholds + bootstrap CI ===========
    perm = RNG.permutation(N); half = N // 2; sel, ev = perm[:half], perm[half:]
    print(f"\n[B] RIGOROUS (slice-level, val-selected thresholds, bootstrap 95% CI)")
    print(f"    threshold chosen on {len(sel):,} slices, reported on held-out {len(ev):,}")
    print("-" * 82)
    print(f"{'class':18s} {'AUC [95% CI]':22s} {'AP':7s} {'F1':7s} {'recall':7s} {'prec':7s}")
    for i, name in enumerate(ALL_COLS):
        y = targets[:, i].astype(int); p = probs[:, i]
        if y.sum() == 0: continue
        auc, lo, hi = boot(y, p, roc_auc_score)
        apv = average_precision_score(y, p)
        t = select_f1_threshold(y[sel], p[sel])
        m = per_class_at_threshold(y[ev], p[ev], t)
        print(f"{name:18s} {auc:.3f} [{lo:.3f},{hi:.3f}]   {apv:.3f}   "
              f"{m['f1']:.3f}   {m['recall']:.3f}   {m['precision']:.3f}")
    print("=" * 82)
    # EDH bootstrap CI on F1 at threshold 0.5 — the headline comparison number
    y = targets[:, 0].astype(int); p = probs[:, 0]
    def f1_at_half(yy, pp): return f1_score(yy, (pp >= 0.5).astype(int), zero_division=0)
    m, lo, hi = boot(y, p, f1_at_half)
    print(f"\nEDH F1 @0.5 (slice-level): {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"WsGSA reported EDH F1: 0.467 (point estimate, no CI, slice-level, "
          f"non-patient-grouped split)")
    if not np.isnan(hi) and hi >= 0.467:
        print("  -> your CI includes/exceeds their point estimate: COMPETITIVE.")
    else:
        print("  -> below their point estimate; report honestly + note their split/CI caveats.")


if __name__ == "__main__":
    main()
