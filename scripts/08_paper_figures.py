#!/usr/bin/env python3
"""
08_paper_figures.py — all Paper-1 figures + comprehensive metrics tables.

Generates the standard figure/metric set used across ICH and long-tailed medical
imaging papers, from logged pooled predictions. Every figure also writes its
underlying numbers to results/ as CSV. Robust: each block is independent; one
failure won't stop the rest.

MAIN-TEXT figures: ROC, PR, calibration, method-comparison, t-SNE.
SUPPLEMENTARY: score distributions, training curves, prevalence, FP-confusion.

---------------------------------------------------------------------------
REVISION v2 — statistical corrections. Changes from v1:

 1. MACRO-AUC CONFIDENCE INTERVAL. v1 built the macro row by averaging the
    per-class CI bounds. That is not a confidence interval for the macro
    statistic: it ignores between-class correlation and is systematically too
    narrow. v2 bootstraps macro-AUC directly (resample studies once per draw,
    compute all six class AUCs on that same resample, then average).

 2. THRESHOLD GRID. v1 searched [0.001] + linspace(0.01, 0.99, 99). EDH
    selected exactly 0.001 — the grid FLOOR — meaning the optimum lay outside
    the searched range and the reported operating point was a grid artefact.
    v2 uses a log-spaced grid from 1e-5 and warns when any class still lands
    on the floor.

 3. OPERATING-POINT VARIANCE. v1 used ONE random split-half, so EDH
    sensitivity rested on ~177 positives from a single partition (SE ~0.037)
    and was reported as a bare point estimate. v2 repeats the split-half
    n_repeats times and reports mean with a [2.5, 97.5] percentile interval.

 4. RNG CONTAMINATION. v1 shared one global RNG between boot_ci and
    best_threshold, so fig_fp_confusion — called after thousands of bootstrap
    draws — computed its threshold from a different permutation than the
    metrics table did. The two disagreed silently. v2 gives threshold
    selection, macro bootstrap, and t-SNE sampling their own seeded RNGs, so
    every number is reproducible and mutually consistent.

 5. POINT ESTIMATES. v1 reported the bootstrap MEAN of AUC/AP as the point
    estimate. v2 reports the plug-in value on the observed data (convention),
    with the bootstrap used only for the interval.

 6. NEW: fixed-threshold-0.5 table. The v1 docstring promised metrics "at
    Youden-optimal threshold AND at 0.5" but only ever computed the former.
    v2 writes a separate metrics_at_thr050_*.csv — this is the protocol-matched
    comparison against papers that report at a fixed 0.5 threshold.
---------------------------------------------------------------------------

Usage:
  # single method (conditional headline):
  python 08_paper_figures.py --runs stage2_cond_fold{F}_conditional_workstation --folds 0 1 2 3 4

  # with method comparison (conditional vs joint):
  python 08_paper_figures.py \
      --runs stage2_cond_fold{F}_conditional_workstation --folds 0 1 2 3 4 \
      --compare stage2_cond_fold{F}_joint_workstation

  # add t-SNE of Stage-1 features:
  python 08_paper_figures.py --runs stage2_cond_fold{F}_conditional_workstation \
      --folds 0 1 2 3 4 --features stage1_fold0_logit_adjusted_workstation

  # faster (fewer threshold repeats):
  python 08_paper_figures.py --runs ... --folds 0 1 2 3 4 --repeats 25
"""
import argparse, warnings, datetime, glob
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, brier_score_loss)

from ich_config import S2_DIR, FEAT_DIR, S1_DIR, ROOT, ALL_COLS

RESULTS = ROOT / "results"; FIGS = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)

# --- Separate, independently seeded generators -----------------------------
# One shared generator caused v1's threshold to depend on how many bootstrap
# draws had already run. Each consumer now owns its stream.
RNG_BOOT = np.random.default_rng(42)      # bootstrap CIs
SEED_THR = 12345                          # threshold split-half selection
SEED_MACRO = 777                          # macro-AUC bootstrap
SEED_TSNE = 2024                          # t-SNE subsampling

NICE = {"epidural": "EDH", "intraparenchymal": "IPH", "intraventricular": "IVH",
        "subarachnoid": "SAH", "subdural": "SDH", "any": "Any"}

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


# ============================================================ STATISTICS
def boot_ci(y, p, metric, n=1000):
    """Bootstrap percentile interval. Point estimate is computed separately as
    the plug-in value; this returns (mean, lo, hi) and callers use lo/hi."""
    idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = RNG_BOOT.choice(idx, len(idx), replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s): continue
        try: v.append(metric(y[s], p[s]))
        except Exception: pass
    if not v: return np.nan, np.nan, np.nan
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# Dense at the low end. Rare classes optimise near zero under Youden's J, and a
# grid that stops at 0.001 pins them to the floor rather than finding an optimum.
THR_GRID = np.unique(np.concatenate([
    np.logspace(-5, -1, 80),        # 1e-5 .. 0.1
    np.linspace(0.10, 0.99, 90),    # 0.10 .. 0.99
]))


def _op_metrics(ys, ps, t):
    """All threshold-dependent metrics at threshold t. Computed from the
    confusion counts directly so no sklearn zero-division surprises."""
    pred = (ps >= t).astype(int)
    tp = int(((pred == 1) & (ys == 1)).sum()); fp = int(((pred == 1) & (ys == 0)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum()); tn = int(((pred == 0) & (ys == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    f2 = 5 * prec * sens / max(4 * prec + sens, 1e-12)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / den) if den > 0 else np.nan
    return dict(sensitivity=sens, specificity=spec, precision=prec, NPV=npv,
                F1=f1, F2=f2, accuracy=(tp + tn) / max(len(ys), 1),
                balanced_acc=(sens + spec) / 2, MCC=mcc,
                youden_J=sens + spec - 1,
                TP=tp, FP=fp, FN=fn, TN=tn)


def threshold_and_op_metrics(y, p, criterion="youden", n_repeats=50, seed=SEED_THR):
    """Repeated non-leaky split-half: choose the threshold on one half, evaluate
    on the other, repeat. Returns (median threshold, mean metrics, lo, hi).

    Repeating matters: with 354 EDH positives a single split leaves ~177 for
    evaluation, so one-shot sensitivity carries SE ~0.037. The percentile
    interval over repeats makes that uncertainty visible instead of hiding it
    behind a point estimate.
    """
    rng = np.random.default_rng(seed)          # dedicated stream, not global
    n = len(y); half = n // 2
    ts, acc = [], []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        sel, ev = perm[:half], perm[half:]
        ys, ps = y[sel], p[sel]
        if ys.sum() == 0 or y[ev].sum() == 0:
            continue
        best_t, best_s = 0.5, -np.inf
        for t in THR_GRID:
            m = _op_metrics(ys, ps, t)
            s = (m["youden_J"] if criterion == "youden"
                 else m["F1"] if criterion == "f1"
                 else m["F2"] if criterion == "f2"
                 else m["youden_J"])
            if s > best_s:
                best_s, best_t = s, t
        ts.append(best_t)
        acc.append(_op_metrics(y[ev], p[ev], best_t))
    if not acc:
        return np.nan, {}, {}, {}
    keys = list(acc[0].keys())
    mean = {k: float(np.mean([a[k] for a in acc])) for k in keys}
    lo   = {k: float(np.percentile([a[k] for a in acc], 2.5)) for k in keys}
    hi   = {k: float(np.percentile([a[k] for a in acc], 97.5)) for k in keys}
    return float(np.median(ts)), mean, lo, hi


def macro_auc_ci(probs, targets, n=1000, seed=SEED_MACRO):
    """Correct macro-AUC bootstrap.

    v1 averaged the per-class CI bounds, which is not a CI for the macro
    statistic. Here each draw resamples STUDIES once, computes every class AUC
    on that same resample, then averages — preserving the correlation between
    classes that share studies.
    """
    rng = np.random.default_rng(seed)
    N = probs.shape[0]; idx = np.arange(N); vals = []
    for _ in range(n):
        s = rng.choice(idx, N, replace=True)
        per = []
        for i in range(len(ALL_COLS)):
            ys = targets[s, i].astype(int)
            if ys.sum() == 0 or ys.sum() == len(ys):
                continue
            try: per.append(roc_auc_score(ys, probs[s, i]))
            except Exception: pass
        if per:
            vals.append(float(np.mean(per)))
    if not vals:
        return np.nan, np.nan, np.nan
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


# ============================================================ METRICS TABLES
def comprehensive_metrics(probs, targets, tag, n_repeats=50):
    """Per-class metrics at the Youden-optimal threshold (selected non-leaky,
    repeated). AUC/AP are plug-in point estimates with bootstrap CIs.
    Threshold-dependent metrics carry [2.5, 97.5] intervals over repeats."""
    rows = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue

        auc_pt = float(roc_auc_score(y, p))              # plug-in, not boot mean
        ap_pt  = float(average_precision_score(y, p))
        _, alo, ahi = boot_ci(y, p, roc_auc_score)
        _, plo, phi = boot_ci(y, p, average_precision_score)
        brier = float(brier_score_loss(y, p))

        t, m, lo, hi = threshold_and_op_metrics(y, p, "youden", n_repeats)

        row = dict(cls=NICE[name], pos=int(y.sum()),
                   AUC=auc_pt, AUC_lo=alo, AUC_hi=ahi,
                   AP=ap_pt, AP_lo=plo, AP_hi=phi,
                   threshold=t,
                   thr_at_grid_floor=bool(np.isfinite(t) and t <= THR_GRID[0] * 1.001),
                   brier=brier, n_repeats=n_repeats)
        for k in m:
            row[k] = m[k]; row[f"{k}_lo"] = lo[k]; row[f"{k}_hi"] = hi[k]

        # --- F2-optimal operating point (recall-weighted; the clinically
        # motivated choice for a subtype where a miss can be fatal) ---
        t2, m2, lo2, hi2 = threshold_and_op_metrics(y, p, "f2", n_repeats)
        row["threshold_f2"] = t2
        for k in m2:
            row[f"{k}_f2"] = m2[k]
            row[f"{k}_f2_lo"] = lo2[k]; row[f"{k}_f2_hi"] = hi2[k]

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- MACRO row: unweighted mean of per-class point estimates.
    # AUC interval comes from the proper macro bootstrap, NOT from averaging
    # per-class bounds. Threshold is left blank (averaging thresholds across
    # classes with different prevalences is meaningless).
    mac_auc, mac_lo, mac_hi = macro_auc_ci(probs, targets)
    macro = {c: "" for c in df.columns}
    macro["cls"] = "MACRO"; macro["pos"] = int(df["pos"].sum())
    macro["AUC"] = float(df["AUC"].mean())
    macro["AUC_lo"] = mac_lo; macro["AUC_hi"] = mac_hi
    macro["AP"] = float(df["AP"].mean())
    macro["brier"] = float(df["brier"].mean())
    macro["threshold"] = ""
    macro["thr_at_grid_floor"] = ""
    macro["n_repeats"] = n_repeats
    for k in ("sensitivity", "specificity", "precision", "NPV", "F1", "F2",
              "accuracy", "balanced_acc", "MCC", "youden_J"):
        if k in df.columns:
            macro[k] = float(df[k].mean())
        if f"{k}_f2" in df.columns:                 
            macro[f"{k}_f2"] = float(df[f"{k}_f2"].mean())
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)

    out = RESULTS / f"metrics_comprehensive_{tag}_{_stamp()}.csv"
    df.to_csv(out, index=False); print(f"[logged] {out}")
    print(f"  macro-AUC (bootstrapped correctly): "
          f"{mac_auc:.4f} [{mac_lo:.4f}, {mac_hi:.4f}]")

    floor = [r["cls"] for r in rows if r.get("thr_at_grid_floor")]
    if floor:
        print(f"  [WARNING] threshold still at grid floor ({THR_GRID[0]:.1e}) "
              f"for: {floor} — optimum may lie below the searched range")
    return df


def metrics_at_fixed_threshold(probs, targets, tag, thr=0.5):
    """Metrics at a FIXED threshold, evaluated on all data.

    No split-half is needed because nothing is selected from the data. This is
    the protocol-matched table for comparison against papers that report at a
    fixed 0.5 threshold (e.g. WsGSA), and it is the number to cite when making
    a like-for-like claim.
    """
    rows = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue
        m = _op_metrics(y, p, thr)
        rows.append(dict(cls=NICE[name], pos=int(y.sum()), threshold=thr, **m))
    df = pd.DataFrame(rows)
    macro = {c: "" for c in df.columns}
    macro["cls"] = "MACRO"; macro["pos"] = int(df["pos"].sum()); macro["threshold"] = thr
    for k in ("sensitivity", "specificity", "precision", "NPV", "F1", "F2",
              "accuracy", "balanced_acc", "MCC", "youden_J"):
        macro[k] = float(df[k].mean())
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)
    out = RESULTS / f"metrics_at_thr{str(thr).replace('.','')}_{tag}_{_stamp()}.csv"
    df.to_csv(out, index=False); print(f"[logged] {out}")
    print(f"  avg F1 @ {thr}: {macro['F1']:.4f}   "
          f"EDH F1 @ {thr}: {df[df['cls']=='EDH']['F1'].iloc[0]:.4f}"
          if (df["cls"] == "EDH").any() else "")
    return df


# ============================================================ FIGURES
def fig_roc(probs, targets, tag):
    fig, ax = plt.subplots(figsize=(6, 6))
    log = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue
        fpr, tpr, _ = roc_curve(y, p); auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f"{NICE[name]} (AUC={auc:.3f})", lw=1.8)
        for a, b in zip(fpr, tpr): log.append(dict(cls=NICE[name], fpr=a, tpr=b))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (pooled OOF)"); ax.legend(fontsize=9, loc="lower right")
    _save(fig, f"roc_{tag}")
    pd.DataFrame(log).to_csv(RESULTS / f"roc_points_{tag}.csv", index=False)


def fig_pr(probs, targets, tag):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue
        pr, rc, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
        ax.plot(rc, pr, label=f"{NICE[name]} (AP={ap:.3f})", lw=1.8)
        ax.axhline(y.mean(), ls=":", lw=0.6, alpha=0.3)   # prevalence baseline
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves (pooled OOF)")
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, f"pr_{tag}")


def fig_calibration(probs, targets, tag, bins=15):
    fig, ax = plt.subplots(figsize=(6, 6)); log = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue
        edges = np.linspace(0, 1, bins + 1); xs, ys = [], []
        for b in range(bins):
            m = (p >= edges[b]) & (p < edges[b + 1])
            if m.sum() == 0: continue
            xs.append(p[m].mean()); ys.append(y[m].mean())
            log.append(dict(cls=NICE[name], pred=p[m].mean(),
                            obs=y[m].mean(), n=int(m.sum())))
        ax.plot(xs, ys, "o-", ms=3, lw=1.2, label=NICE[name])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (reliability) curves"); ax.legend(fontsize=9)
    _save(fig, f"calibration_{tag}")
    pd.DataFrame(log).to_csv(RESULTS / f"calibration_points_{tag}.csv", index=False)


def fig_score_dist(probs, targets, tag):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axes = axes.ravel()
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]; ax = axes[i]
        ax.hist(p[y == 0], bins=40, alpha=0.5, label="neg", density=True)
        ax.hist(p[y == 1], bins=40, alpha=0.5, label="pos", density=True)
        ax.set_title(NICE[name]); ax.legend(fontsize=8); ax.set_xlabel("score")
    fig.suptitle("Score distributions (pos vs neg)"); fig.tight_layout()
    _save(fig, f"score_dist_{tag}")


def fig_method_compare(pa, pb, targets, name_a, name_b, tag):
    """Grouped bar: per-class AUC with bootstrap CI error bars, A vs B."""
    labels, aa, ea, bb, eb, log = [], [], [], [], [], []
    for i, name in enumerate(ALL_COLS):
        y = targets[:, i].astype(int)
        if y.sum() == 0: continue
        a_pt = float(roc_auc_score(y, pa[:, i]))
        b_pt = float(roc_auc_score(y, pb[:, i]))
        _, alo, ahi = boot_ci(y, pa[:, i], roc_auc_score)
        _, blo, bhi = boot_ci(y, pb[:, i], roc_auc_score)
        labels.append(NICE[name])
        aa.append(a_pt); ea.append([a_pt - alo, ahi - a_pt])
        bb.append(b_pt); eb.append([b_pt - blo, bhi - b_pt])
        log.append(dict(cls=NICE[name], A_auc=a_pt, A_lo=alo, A_hi=ahi,
                        B_auc=b_pt, B_lo=blo, B_hi=bhi, diff=a_pt - b_pt))
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, aa, w, yerr=np.array(ea).T, capsize=3, label=name_a)
    ax.bar(x + w/2, bb, w, yerr=np.array(eb).T, capsize=3, label=name_b)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("AUC")
    ax.set_ylim(0.75, 1.0); ax.set_title("Per-class AUC: method comparison (95% CI)")
    ax.legend()
    _save(fig, f"method_compare_{tag}")
    pd.DataFrame(log).to_csv(RESULTS / f"method_compare_{tag}.csv", index=False)


def fig_training_curves():
    """EDH AUC / EDH AP / macro AUC across epochs, all folds (from history.csv)."""
    hists = sorted(glob.glob(str(S1_DIR / "stage1_fold*_logit_adjusted_workstation" / "history.csv")))
    if not hists:
        print("  no history.csv found; skipping training curves"); return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for h in hists:
        df = pd.read_csv(h); fold = Path(h).parent.name.split("_")[1]
        axes[0].plot(df["epoch"], df["macro_auc"], marker="o", ms=3, label=fold)
        axes[1].plot(df["epoch"], df["epidural_auc"], marker="o", ms=3, label=fold)
        axes[2].plot(df["epoch"], df["epidural_ap"], marker="o", ms=3, label=fold)
    for ax, t in zip(axes, ["macro AUC", "EDH AUC", "EDH AP"]):
        ax.set_xlabel("epoch"); ax.set_title(t); ax.legend(fontsize=8, title="fold")
    fig.suptitle("Stage-1 training curves across folds"); fig.tight_layout()
    _save(fig, "training_curves")


def fig_fp_confusion(probs, targets, tag, n_repeats=25, criterion="f2"):
    """EDH false-positive breakdown by co-occurring subtype, at the fixed 0.5
    operating point reported throughout the paper. Criterion-based thresholds
    (Youden, F2) collapse toward zero at this prevalence and yield a diagnostic
    dominated by low-confidence false positives."""
    E = 0; p = probs[:, E]; y = targets[:, E].astype(int)
    t = 0.5
    fp = (p >= t) & (y == 0)
    if fp.sum() == 0:
        print("  no EDH FPs at threshold; skipping"); return
    labels, vals, base = [], [], []
    for j, name in enumerate(ALL_COLS):
        if name in ("epidural", "any"): continue
        labels.append(NICE[name])
        vals.append(targets[fp, j].mean()); base.append(targets[:, j].mean())
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, vals, w, label="among EDH false positives")
    ax.bar(x + w/2, base, w, label="base rate")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("fraction of studies")
    ax.set_title(f"EDH false-positive subtype enrichment (threshold {t:.4g})")
    ax.legend()
    _save(fig, f"fp_confusion_{tag}")
    pd.DataFrame({"subtype": labels, "in_fp": vals, "base": base,
                  "enrichment": np.array(vals) / np.array(base),
                  "threshold": t, "n_fp": int(fp.sum())}
                 ).to_csv(RESULTS / f"fp_confusion_{tag}.csv", index=False)
    print(f"  fp_confusion computed at threshold {t:.4g} on {int(fp.sum())} EDH FPs")


def fig_prevalence(targets):
    counts = targets.sum(0).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([NICE[c] for c in ALL_COLS], counts)
    ax.set_ylabel("positive studies"); ax.set_title("Class prevalence (study level)")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c}\n{c/len(targets)*100:.1f}%",
                ha="center", va="bottom", fontsize=8)
    _save(fig, "class_prevalence")
    pd.DataFrame({"cls": [NICE[c] for c in ALL_COLS], "positives": counts,
                  "prevalence": counts / len(targets)}
                 ).to_csv(RESULTS / "class_prevalence.csv", index=False)


def fig_tsne(feat_run, max_pts=6000):
    """t-SNE of Stage-1 features: EDH-vs-SDH separation (the confusion story)."""
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("  sklearn TSNE unavailable; skipping"); return
    shards = sorted(glob.glob(str(FEAT_DIR / feat_run / "shard_*.npz")))
    if not shards:
        print(f"  no features in {FEAT_DIR/feat_run}; skipping t-SNE"); return
    rng = np.random.default_rng(SEED_TSNE)     # dedicated stream -> reproducible
    F, L = [], []
    for s in shards:
        d = np.load(s, allow_pickle=True); F.append(d["feat"]); L.append(d["labels"])
    F = np.concatenate(F).astype(np.float32); L = np.concatenate(L)
    edh = L[:, 0] > 0; sdh = (L[:, 4] > 0) & (~edh); neg = L.sum(1) == 0
    idx = np.concatenate([
        np.where(edh)[0],
        rng.choice(np.where(sdh)[0], min(2000, int(sdh.sum())), replace=False),
        rng.choice(np.where(neg)[0], min(2000, int(neg.sum())), replace=False)])
    rng.shuffle(idx); idx = idx[:max_pts]
    print(f"  running t-SNE on {len(idx)} points (may take a few min)...")
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=42).fit_transform(F[idx])
    lab = np.where(L[idx, 0] > 0, "EDH", np.where(L[idx, 4] > 0, "SDH", "other"))
    fig, ax = plt.subplots(figsize=(7, 7))
    for cls, col in [("other", "#cccccc"), ("SDH", "#1f77b4"), ("EDH", "#d62728")]:
        m = lab == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=8, c=col, label=cls, alpha=0.6)
    ax.set_title("Stage-1 feature space (t-SNE): EDH vs SDH"); ax.legend()
    _save(fig, f"tsne_{feat_run}")


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--compare", default=None, help="second run template for method comparison")
    ap.add_argument("--features", default=None, help="features run for t-SNE")
    ap.add_argument("--tag", default="conditional", help="label used in output filenames")
    ap.add_argument("--repeats", type=int, default=50,
                    help="split-half repeats for threshold selection (25 is faster)")
    a = ap.parse_args()

    tag = a.tag
    probs, targets, studies = load_pooled(a.runs, a.folds)
    print(f"loaded pooled OOF: {len(targets):,} studies")
    print(f"threshold grid: {len(THR_GRID)} points, {THR_GRID[0]:.1e} .. {THR_GRID[-1]:.2f}")
    print(f"split-half repeats: {a.repeats}\n")

    def safe(fn, *args, name="", **kw):
        try: fn(*args, **kw)
        except Exception as e: print(f"  [skip {name}] {type(e).__name__}: {e}")

    # metrics tables
    safe(comprehensive_metrics, probs, targets, tag, a.repeats, name="metrics")
    safe(metrics_at_fixed_threshold, probs, targets, tag, 0.5, name="metrics@0.5")

    # main-text figures
    safe(fig_roc, probs, targets, tag, name="roc")
    safe(fig_pr, probs, targets, tag, name="pr")
    safe(fig_calibration, probs, targets, tag, name="calibration")

    # supplementary
    safe(fig_score_dist, probs, targets, tag, name="score_dist")
    safe(fig_fp_confusion, probs, targets, tag, name="fp_confusion")
    safe(fig_prevalence, targets, name="prevalence")
    safe(fig_training_curves, name="training_curves")

    if a.compare:
        pb, tb, sb = load_pooled(a.compare, a.folds)
        ib = {s: i for i, s in enumerate(sb)}
        ra, rb = [], []
        for i, s in enumerate(studies):
            j = ib.get(s)
            if j is not None: ra.append(i); rb.append(j)
        ra, rb = np.array(ra), np.array(rb)
        print(f"\nmethod comparison aligned on {len(ra):,} studies")
        safe(fig_method_compare, probs[ra], pb[rb], targets[ra],
             "Conditional", "Joint", tag, name="method_compare")

    if a.features:
        safe(fig_tsne, a.features, name="tsne")

    print(f"\nDONE. figures -> {FIGS}   metrics/csv -> {RESULTS}")


if __name__ == "__main__":
    main()