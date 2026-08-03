#!/usr/bin/env python3
"""
04_evaluate.py — pooled OOF metrics + paired significance tests (AUC and AP).
Now LOGS every result to results/ as CSV + JSON with a timestamp, so numbers are
never lost. Prints to screen AND writes to disk.

    python 04_evaluate.py --runs stage2_cond_fold{F}_conditional_workstation --folds 0 1 2 3 4
    python 04_evaluate.py --compare \
        --run_a stage2_cond_fold{F}_conditional_workstation \
        --run_b stage2_cond_fold{F}_joint_workstation --folds 0 1 2 3 4
"""
import argparse, warnings, json, datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
from ich_config import S2_DIR as RUNS, ROOT

RESULTS = ROOT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
SUBTYPES = ["epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]
ALL_COLS = SUBTYPES + ["any"]
EDH_I = 0
RNG = np.random.default_rng(42)


def _stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_pooled(run_template, folds):
    P, T, S = [], [], []
    for f in folds:
        run = run_template.replace("{F}", str(f))
        npz = RUNS / run / "val_predictions.npz"
        if not npz.exists():
            print(f"  MISSING: {npz}"); continue
        d = np.load(npz, allow_pickle=True)
        P.append(d["probs"]); T.append(d["targets"])
        S.append(d["study"].astype(str) if "study" in d
                 else np.array([f"f{f}_{i}" for i in range(len(d['probs']))]))
    if not P:
        raise SystemExit("no predictions found — check run names / folds")
    return np.concatenate(P), np.concatenate(T), np.concatenate(S)


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


def compute_table(probs, targets):
    """Return a list of per-class dicts + macro row."""
    rows = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i], probs[:, i]
        if y.sum() == 0: continue
        a, alo, ahi = boot_ci(y, p, roc_auc_score)
        ap, plo, phi = boot_ci(y, p, average_precision_score)
        rows.append(dict(cls=name, pos=int(y.sum()),
                         auc=a, auc_lo=alo, auc_hi=ahi,
                         ap=ap, ap_lo=plo, ap_hi=phi, ece=ece(y, p)))
    aucs = [r["auc"] for r in rows]; aps = [r["ap"] for r in rows]
    rows.append(dict(cls="macro", pos=len(targets),
                     auc=float(np.mean(aucs)), auc_lo=np.nan, auc_hi=np.nan,
                     ap=float(np.mean(aps)), ap_lo=np.nan, ap_hi=np.nan, ece=np.nan))
    return rows


def report(rows, title, save_tag):
    print("=" * 74); print(title)
    print(f"pooled OOF: {rows[0]['pos'] if rows else 0} (per-class pos shown at right)")
    print("-" * 74)
    print(f"{'class':18s} {'AUC [95% CI]':26s} {'AP [95% CI]':26s} {'ECE':6s} pos")
    for r in rows:
        if r["cls"] == "macro":
            print("-" * 74)
            print(f"{'macro':18s} AUC={r['auc']:.4f}   AP={r['ap']:.4f}")
        else:
            print(f"{r['cls']:18s} {r['auc']:.3f} [{r['auc_lo']:.3f},{r['auc_hi']:.3f}]   "
                  f"{r['ap']:.3f} [{r['ap_lo']:.3f},{r['ap_hi']:.3f}]   {r['ece']:.3f}  {r['pos']}")
    print("=" * 74)
    # ---- LOG to CSV ----
    out = RESULTS / f"{save_tag}_{_stamp()}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[logged] {out}")


def align(pa, ta, sa, pb, tb, sb):
    ib = {s: i for i, s in enumerate(sb)}
    ra, rb = [], []
    for i, s in enumerate(sa):
        j = ib.get(s)
        if j is not None: ra.append(i); rb.append(j)
    ra, rb = np.array(ra), np.array(rb)
    assert np.array_equal(ta[ra], tb[rb]), "labels differ for the same study id"
    return pa[ra], pb[rb], ta[ra]


def paired_test(pa, pb, targets, metric, cls_idx=EDH_I, n=2000):
    y = targets[:, cls_idx]; a = pa[:, cls_idx]; b = pb[:, cls_idx]
    idx = np.arange(len(y)); diffs = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s): continue
        try: diffs.append(metric(y[s], a[s]) - metric(y[s], b[s]))
        except Exception: pass
    diffs = np.array(diffs)
    return dict(mean=float(diffs.mean()), lo=float(np.percentile(diffs, 2.5)),
                hi=float(np.percentile(diffs, 97.5)), p_a_gt_b=float((diffs > 0).mean()))


def _verdict(name, d):
    tag = ("A BETTER (CI excludes 0)" if d["lo"] > 0 else
           "A WORSE (CI excludes 0)" if d["hi"] < 0 else "no sig. difference")
    print(f"  {name}: mean diff = {d['mean']:+.4f}  95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
          f"P(A>B)={d['p_a_gt_b']:.1%}  -> {tag}")
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--run_a"); ap.add_argument("--run_b")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    a = ap.parse_args()

    if a.compare:
        pa, ta, sa = load_pooled(a.run_a, a.folds)
        pb, tb, sb = load_pooled(a.run_b, a.folds)
        pa, pb, tt = align(pa, ta, sa, pb, tb, sb)
        rows_a = compute_table(pa, tt); report(rows_a, f"METHOD A: {a.run_a}", "compare_A")
        rows_b = compute_table(pb, tt); report(rows_b, f"METHOD B: {a.run_b}", "compare_B")
        print(f"\nPAIRED BOOTSTRAP (A - B), aligned by study, {len(tt)} studies:\n EDH:")
        d_auc = paired_test(pa, pb, tt, roc_auc_score, EDH_I); t_auc = _verdict("AUC", d_auc)
        d_ap  = paired_test(pa, pb, tt, average_precision_score, EDH_I); t_ap = _verdict("AP ", d_ap)
        # ---- LOG the significance test to JSON ----
        log = dict(run_a=a.run_a, run_b=a.run_b, folds=a.folds, n_studies=len(tt),
                   edh_auc_test=dict(**d_auc, verdict=t_auc),
                   edh_ap_test=dict(**d_ap, verdict=t_ap))
        out = RESULTS / f"compare_significance_{_stamp()}.json"
        json.dump(log, open(out, "w"), indent=2)
        print(f"[logged] {out}")
    else:
        probs, targets, _ = load_pooled(a.runs, a.folds)
        rows = compute_table(probs, targets)
        report(rows, f"POOLED OOF: {a.runs}", "pooled")


if __name__ == "__main__":
    main()