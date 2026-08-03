#!/usr/bin/env python3
"""
09_clinical_utility.py — clinical-utility analysis from logged predictions.

WHY THIS EXISTS
---------------
AUC and F1 answer "is the model good?" Clinicians and clinical reviewers ask a
different question: "at an operating point I would actually deploy, what do I
gain and what does it cost me?" Most ICH classification papers never answer it.
Adding this is cheap (no retraining, pure post-processing of logged
predictions) and it is one of the gaps a high-impact venue will flag.

Four analyses, all with bootstrap confidence intervals:

  1. SENSITIVITY AT FIXED SPECIFICITY (90/95/98%)
     The conventional way to compare detectors at a controlled false-alarm rate.
     Reads directly off the ROC curve, no threshold tuning.

  2. SPECIFICITY AT FIXED SENSITIVITY (90/95/99%)
     The clinically dominant view for a subtype where a miss can be fatal.
     "If we insist on catching 95% of EDH, what fraction of normal studies do
     we still clear automatically?" For a rare, high-consequence class this is
     the more honest framing, and it exposes the precision ceiling plainly.

  3. DECISION CURVE ANALYSIS (Vickers & Elkin, 2006)
     Net benefit across threshold probabilities, against treat-all and
     treat-none references. The standard clinical-ML answer to "is this model
     useful, not merely accurate?" A model whose net-benefit curve sits below
     treat-all across the plausible threshold range provides no clinical value
     regardless of its AUC.

  4. TRIAGE WORKLOAD
     Alerts per 100 studies, PPV, and number-needed-to-review at each operating
     point. This is what determines whether a radiology department would accept
     the system, and it makes the rare-class precision ceiling concrete: at
     ~1.6% prevalence, high sensitivity implies reviewing many negatives per
     true positive.

USAGE
-----
    python 09_clinical_utility.py --runs stage2_cond_fold{F}_conditional_workstation \
        --folds 0 1 2 3 4

    # compare two methods on the decision curve
    python 09_clinical_utility.py --runs ...conditional... --folds 0 1 2 3 4 \
        --compare stage2_cond_fold{F}_joint_workstation

OUTPUTS (results/)
    clinical_sens_at_spec_{tag}.csv
    clinical_spec_at_sens_{tag}.csv
    clinical_decision_curve_{tag}.csv
    clinical_workload_{tag}.csv
    figures/decision_curve_{tag}.png/.pdf
    figures/sens_at_spec_{tag}.png/.pdf
"""
import argparse, datetime, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

from ich_config import S2_DIR, ROOT, ALL_COLS

RESULTS = ROOT / "results"; FIGS = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)

NICE = {"epidural": "EDH", "intraparenchymal": "IPH", "intraventricular": "IVH",
        "subarachnoid": "SAH", "subdural": "SDH", "any": "Any"}

SPEC_POINTS = [0.90, 0.95, 0.98]      # ask supervisors; all three reported
SENS_POINTS = [0.90, 0.95, 0.99]
N_BOOT = 1000
SEED = 20260727

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def _stamp(): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig); print(f"[figure] {FIGS/name}.png/.pdf")


def load_pooled(tpl, folds):
    P, T, S = [], [], []
    for f in folds:
        npz = S2_DIR / tpl.replace("{F}", str(f)) / "val_predictions.npz"
        if not npz.exists():
            print(f"  MISSING: {npz}"); continue
        d = np.load(npz, allow_pickle=True)
        P.append(d["probs"]); T.append(d["targets"])
        S.append(d["study"].astype(str) if "study" in d
                 else np.array([f"f{f}_{i}" for i in range(len(d['probs']))]))
    if not P: raise SystemExit("no predictions found")
    return np.concatenate(P), np.concatenate(T), np.concatenate(S)


# ============================================================ CORE OPERATIONS
def sens_at_spec(y, p, target_spec):
    """Sensitivity at the highest threshold achieving >= target specificity.
    Read off the ROC curve, so no threshold search and no selection bias."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = (1 - fpr) >= target_spec
    if not ok.any():
        return np.nan, np.nan
    i = np.argmax(tpr * ok)          # best sensitivity among valid points
    return float(tpr[i]), float(thr[i])


def spec_at_sens(y, p, target_sens):
    """Specificity at the threshold achieving >= target sensitivity."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = tpr >= target_sens
    if not ok.any():
        return np.nan, np.nan
    spec = 1 - fpr
    i = np.argmax(spec * ok)
    return float(spec[i]), float(thr[i])


def boot_stat(y, p, fn, target, n=N_BOOT, seed=SEED):
    """Percentile CI for an operating-point statistic."""
    rng = np.random.default_rng(seed)
    N = len(y); idx = np.arange(N); v = []
    for _ in range(n):
        s = rng.choice(idx, N, replace=True)
        if y[s].sum() < 2:
            continue
        val, _ = fn(y[s], p[s], target)
        if np.isfinite(val):
            v.append(val)
    if not v:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def workload(y, p, thr):
    """What a department actually experiences at this operating point."""
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn_ = int(((pred == 0) & (y == 1)).sum())
    alerts = tp + fp
    ppv = tp / max(alerts, 1)
    return dict(threshold=thr, alerts=alerts,
                alerts_per_100=100.0 * alerts / len(y),
                PPV=ppv,
                number_needed_to_review=(1.0 / ppv) if ppv > 0 else np.nan,
                TP=tp, FP=fp, FN=fn_, missed=fn_)


def net_benefit(y, p, pt):
    """Vickers & Elkin net benefit at threshold probability pt.

        NB = TP/N - (FP/N) * (pt / (1 - pt))

    The odds factor encodes the clinician's implied trade: at pt, one missed
    case is considered as costly as (1-pt)/pt unnecessary reviews.
    """
    pred = (p >= pt).astype(int)
    N = len(y)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    return tp / N - (fp / N) * (pt / max(1 - pt, 1e-9))


def net_benefit_all(y, pt):
    """Reference strategy: flag every study."""
    prev = y.mean()
    return prev - (1 - prev) * (pt / max(1 - pt, 1e-9))


# ============================================================ ANALYSES
def table_sens_at_spec(probs, targets, tag):
    rows = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue
        for sp in SPEC_POINTS:
            s, t = sens_at_spec(y, p, sp)
            lo, hi = boot_stat(y, p, sens_at_spec, sp)
            w = workload(y, p, t) if np.isfinite(t) else {}
            rows.append(dict(cls=NICE[name], pos=int(y.sum()),
                             target_specificity=sp, sensitivity=s,
                             sens_lo=lo, sens_hi=hi, **w))
    df = pd.DataFrame(rows)
    out = RESULTS / f"clinical_sens_at_spec_{tag}_{_stamp()}.csv"
    df.to_csv(out, index=False); print(f"[logged] {out.name}")
    print("\nSENSITIVITY AT FIXED SPECIFICITY")
    print(df[["cls", "pos", "target_specificity", "sensitivity", "sens_lo",
              "sens_hi", "PPV", "alerts_per_100", "missed"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return df


def table_spec_at_sens(probs, targets, tag):
    rows = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue
        for se in SENS_POINTS:
            s, t = spec_at_sens(y, p, se)
            lo, hi = boot_stat(y, p, spec_at_sens, se)
            w = workload(y, p, t) if np.isfinite(t) else {}
            rows.append(dict(cls=NICE[name], pos=int(y.sum()),
                             target_sensitivity=se, specificity=s,
                             spec_lo=lo, spec_hi=hi, **w))
    df = pd.DataFrame(rows)
    out = RESULTS / f"clinical_spec_at_sens_{tag}_{_stamp()}.csv"
    df.to_csv(out, index=False); print(f"[logged] {out.name}")
    print("\nSPECIFICITY AT FIXED SENSITIVITY  (the rare-class view)")
    print(df[["cls", "pos", "target_sensitivity", "specificity", "spec_lo",
              "spec_hi", "PPV", "alerts_per_100", "number_needed_to_review"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return df


def decision_curves(probs, targets, tag, probs_b=None, name_a="Model", name_b=None):
    """Net benefit vs threshold probability, per class."""
    grid = np.linspace(0.001, 0.50, 200)
    log = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.ravel()
    for i, name in enumerate(ALL_COLS):
        y = targets[:, i].astype(int); p = probs[:, i]
        if y.sum() == 0: continue
        nb_m = np.array([net_benefit(y, p, t) for t in grid])
        nb_a = np.array([net_benefit_all(y, t) for t in grid])
        ax = axes[i]
        ax.plot(grid, nb_m, lw=1.8, label=name_a)
        if probs_b is not None:
            nb_b = np.array([net_benefit(y, probs_b[:, i], t) for t in grid])
            ax.plot(grid, nb_b, lw=1.5, ls="--", label=name_b)
            for t, a, b in zip(grid, nb_m, nb_b):
                log.append(dict(cls=NICE[name], pt=t, nb_model=a, nb_compare=b))
        else:
            for t, a in zip(grid, nb_m):
                log.append(dict(cls=NICE[name], pt=t, nb_model=a))
        ax.plot(grid, nb_a, lw=1, color="gray", label="flag all")
        ax.axhline(0, color="k", lw=1, ls=":", label="flag none")
        ax.set_xlim(0, 0.5)
        lo = min(0, float(np.nanmin(nb_m)) * 0.5)
        ax.set_ylim(lo, max(float(np.nanmax(nb_m)) * 1.2, 0.01))
        ax.set_title(f"{NICE[name]} (prev {y.mean()*100:.1f}%)")
        ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit")
        ax.legend(fontsize=8)
    fig.suptitle("Decision curve analysis (pooled out-of-fold)")
    fig.tight_layout()
    _save(fig, f"decision_curve_{tag}")
    df = pd.DataFrame(log)
    out = RESULTS / f"clinical_decision_curve_{tag}_{_stamp()}.csv"
    df.to_csv(out, index=False); print(f"[logged] {out.name}")

    # where does the model beat both references?
    print("\nDECISION CURVE — useful threshold range (model above both references)")
    for name in ALL_COLS:
        sub = df[df.cls == NICE[name]]
        if sub.empty: continue
        i = ALL_COLS.index(name)
        y = targets[:, i].astype(int)
        ref = np.array([max(net_benefit_all(y, t), 0.0) for t in sub["pt"]])
        good = sub["pt"].values[sub["nb_model"].values > ref]
        if len(good):
            print(f"  {NICE[name]:>4}: pt in [{good.min():.3f}, {good.max():.3f}]")
        else:
            print(f"  {NICE[name]:>4}: never exceeds both references — "
                  f"no net clinical benefit at any threshold")
    return df


def fig_sens_at_spec(df, tag):
    fig, ax = plt.subplots(figsize=(9, 5))
    classes = [c for c in df["cls"].unique()]
    x = np.arange(len(classes)); w = 0.26
    for j, sp in enumerate(SPEC_POINTS):
        sub = df[df.target_specificity == sp].set_index("cls").reindex(classes)
        vals = sub["sensitivity"].values
        err = np.vstack([vals - sub["sens_lo"].values,
                         sub["sens_hi"].values - vals])
        ax.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=3,
               label=f"spec {int(sp*100)}%")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("sensitivity"); ax.set_ylim(0, 1)
    ax.set_title("Sensitivity at fixed specificity (95% CI)")
    ax.legend()
    _save(fig, f"sens_at_spec_{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--compare", default=None)
    ap.add_argument("--tag", default="conditional")
    a = ap.parse_args()

    probs, targets, studies = load_pooled(a.runs, a.folds)
    print(f"loaded pooled OOF: {len(targets):,} studies\n")

    pb = None
    if a.compare:
        p2, t2, s2 = load_pooled(a.compare, a.folds)
        ib = {s: i for i, s in enumerate(s2)}
        ra, rb = [], []
        for i, s in enumerate(studies):
            j = ib.get(s)
            if j is not None: ra.append(i); rb.append(j)
        ra, rb = np.array(ra), np.array(rb)
        probs, targets, pb = probs[ra], targets[ra], p2[rb]
        print(f"comparison aligned on {len(ra):,} studies\n")

    d1 = table_sens_at_spec(probs, targets, a.tag)
    d2 = table_spec_at_sens(probs, targets, a.tag)
    decision_curves(probs, targets, a.tag, pb, "Conditional",
                    "Joint" if pb is not None else None)
    fig_sens_at_spec(d1, a.tag)

    # workload summary at the recall-first operating point
    wl = d2[d2.target_sensitivity == 0.95][
        ["cls", "pos", "specificity", "PPV", "alerts_per_100",
         "number_needed_to_review"]]
    out = RESULTS / f"clinical_workload_{a.tag}_{_stamp()}.csv"
    wl.to_csv(out, index=False)
    print(f"\n[logged] {out.name}")
    print("\nTRIAGE WORKLOAD AT 95% SENSITIVITY")
    print(wl.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nDONE. tables -> {RESULTS}   figures -> {FIGS}")


if __name__ == "__main__":
    main()