#!/usr/bin/env python3
"""
08_paper_figures.py — all Paper-1 figures + a comprehensive metrics table.

Generates the standard figure/metric set used across ICH and long-tailed medical
imaging papers, from your logged pooled predictions. Every figure also writes its
underlying numbers to results/ as CSV. Robust: each block is independent; one
failure won't stop the rest.

MAIN-TEXT figures (the strong ones): ROC, PR, calibration, method-comparison, t-SNE.
SUPPLEMENTARY: score distributions, training curves, prevalence, FP-confusion.

Usage:
  # single method (your conditional headline):
  python 08_paper_figures.py --runs stage2_cond_fold{F}_conditional_workstation --folds 0 1 2 3 4

  # with method comparison (conditional vs joint):
  python 08_paper_figures.py \
      --runs stage2_cond_fold{F}_conditional_workstation --folds 0 1 2 3 4 \
      --compare stage2_cond_fold{F}_joint_workstation

  # add t-SNE of Stage-1 features (needs a features/ dir):
  python 08_paper_figures.py --runs stage2_cond_fold{F}_conditional_workstation \
      --folds 0 1 2 3 4 --features stage1_fold0_logit_adjusted_workstation
"""
import argparse, warnings, datetime, glob
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, f1_score, fbeta_score,
                             matthews_corrcoef, brier_score_loss, confusion_matrix)

from ich_config import S2_DIR, FEAT_DIR, S1_DIR, ROOT, ALL_COLS

RESULTS = ROOT / "results"; FIGS = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
NICE = {"epidural":"EDH","intraparenchymal":"IPH","intraventricular":"IVH",
        "subarachnoid":"SAH","subdural":"SDH","any":"Any"}

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


def boot_ci(y, p, metric, n=1000):
    idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s): continue
        try: v.append(metric(y[s], p[s]))
        except Exception: pass
    if not v: return np.nan, np.nan, np.nan
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def best_threshold(y, p, criterion="youden"):
    """Non-leaky threshold on a held-out half; return threshold from the OTHER half."""
    n = len(y); perm = RNG.permutation(n); half = n // 2
    ys, ps = y[perm[:half]], p[perm[:half]]
    best_t, best = 0.5, -1
    for t in np.unique(np.concatenate([[0.001], np.linspace(0.01, 0.99, 99)])):
        pred = (ps >= t).astype(int)
        tp = ((pred == 1) & (ys == 1)).sum(); fp = ((pred == 1) & (ys == 0)).sum()
        fn = ((pred == 0) & (ys == 1)).sum(); tn = ((pred == 0) & (ys == 0)).sum()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        if criterion == "youden": s = sens + spec - 1
        elif criterion == "f1":   s = f1_score(ys, pred, zero_division=0)
        elif criterion == "f2":   s = fbeta_score(ys, pred, beta=2, zero_division=0)
        else: s = sens + spec - 1
        if s > best: best, best_t = s, t
    return best_t, perm[half:]   # eval indices = the other half


# ============================================================ COMPREHENSIVE METRICS
def comprehensive_metrics(probs, targets, tag):
    """Every metric the literature reports, per class, at Youden-optimal threshold
    (selected non-leaky) AND at 0.5. Bootstrap CIs on AUC/AP. Logged to CSV."""
    rows = []
    for i, name in enumerate(ALL_COLS):
        y, p = targets[:, i].astype(int), probs[:, i]
        if y.sum() == 0: continue
        auc, alo, ahi = boot_ci(y, p, roc_auc_score)
        ap, plo, phi = boot_ci(y, p, average_precision_score)
        brier = brier_score_loss(y, p)
        t, ev = best_threshold(y, p, "youden")
        ye, pe = y[ev], (p[ev] >= t).astype(int)
        tp = ((pe == 1) & (ye == 1)).sum(); fp = ((pe == 1) & (ye == 0)).sum()
        fn = ((pe == 0) & (ye == 1)).sum(); tn = ((pe == 0) & (ye == 0)).sum()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
        rows.append(dict(
            cls=NICE[name], pos=int(y.sum()),
            AUC=auc, AUC_lo=alo, AUC_hi=ahi, AP=ap, AP_lo=plo, AP_hi=phi,
            threshold=t, sensitivity=sens, specificity=spec, precision=prec, NPV=npv,
            F1=f1_score(ye, pe, zero_division=0),
            F2=fbeta_score(ye, pe, beta=2, zero_division=0),
            accuracy=(tp + tn) / max(len(ye), 1),
            balanced_acc=(sens + spec) / 2,
            MCC=matthews_corrcoef(ye, pe) if len(np.unique(ye)) > 1 else np.nan,
            youden_J=sens + spec - 1, brier=brier))
    df = pd.DataFrame(rows)
    # macro row
    num = df.select_dtypes(include=[float]).mean(numeric_only=True)
    macro = {c: (num[c] if c in num else "") for c in df.columns}
    macro["cls"] = "MACRO"; macro["pos"] = int(df["pos"].sum())
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)
    out = RESULTS / f"metrics_comprehensive_{tag}_{_stamp()}.csv"
    df.to_csv(out, index=False); print(f"[logged] {out}")
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
    ax.set_title("Precision–Recall curves (pooled OOF)"); ax.legend(fontsize=9, loc="upper right")
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
            log.append(dict(cls=NICE[name], pred=p[m].mean(), obs=y[m].mean(), n=int(m.sum())))
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
    labels, aa, ea, bb, eb = [], [], [], [], []
    log = []
    for i, name in enumerate(ALL_COLS):
        y = targets[:, i].astype(int)
        if y.sum() == 0: continue
        am, alo, ahi = boot_ci(y, pa[:, i], roc_auc_score)
        bm, blo, bhi = boot_ci(y, pb[:, i], roc_auc_score)
        labels.append(NICE[name]); aa.append(am); ea.append([am-alo, ahi-am])
        bb.append(bm); eb.append([bm-blo, bhi-bm])
        log.append(dict(cls=NICE[name], A_auc=am, A_lo=alo, A_hi=ahi,
                        B_auc=bm, B_lo=blo, B_hi=bhi, diff=am-bm))
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
    if not hists: print("  no history.csv found; skipping training curves"); return
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


def fig_fp_confusion(probs, targets, tag):
    """EDH false-positive breakdown by co-occurring subtype (your diagnostic, as a figure)."""
    E = 0; p = probs[:, E]; y = targets[:, E].astype(int)
    t, _ = best_threshold(y, p, "youden")
    fp = (p >= t) & (y == 0)
    if fp.sum() == 0: print("  no EDH FPs at threshold; skipping"); return
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
    ax.set_title("EDH false-positive subtype enrichment"); ax.legend()
    _save(fig, f"fp_confusion_{tag}")
    pd.DataFrame({"subtype": labels, "in_fp": vals, "base": base,
                  "enrichment": np.array(vals)/np.array(base)}
                 ).to_csv(RESULTS / f"fp_confusion_{tag}.csv", index=False)


def fig_prevalence(targets):
    counts = targets.sum(0).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([NICE[c] for c in ALL_COLS], counts)
    ax.set_ylabel("positive studies"); ax.set_title("Class prevalence (study level)")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c}\n{c/len(targets)*100:.1f}%", ha="center", va="bottom", fontsize=8)
    _save(fig, "class_prevalence")


def fig_tsne(feat_run, max_pts=6000):
    """t-SNE of Stage-1 features: shows EDH-vs-SDH separation (the confusion story)."""
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("  sklearn TSNE unavailable; skipping"); return
    shards = sorted(glob.glob(str(FEAT_DIR / feat_run / "shard_*.npz")))
    if not shards: print(f"  no features in {FEAT_DIR/feat_run}; skipping t-SNE"); return
    F, L = [], []
    for s in shards:
        d = np.load(s, allow_pickle=True); F.append(d["feat"]); L.append(d["labels"])
    F = np.concatenate(F).astype(np.float32); L = np.concatenate(L)
    edh = L[:, 0] > 0; sdh = (L[:, 4] > 0) & (~edh); neg = L.sum(1) == 0
    # take all EDH, sample SDH and negatives
    idx = np.concatenate([np.where(edh)[0],
                          RNG.choice(np.where(sdh)[0], min(2000, sdh.sum()), replace=False),
                          RNG.choice(np.where(neg)[0], min(2000, neg.sum()), replace=False)])
    RNG.shuffle(idx); idx = idx[:max_pts]
    print(f"  running t-SNE on {len(idx)} points (may take a few min)...")
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(F[idx])
    lab = np.where(L[idx, 0] > 0, "EDH", np.where(L[idx, 4] > 0, "SDH", "other"))
    fig, ax = plt.subplots(figsize=(7, 7))
    for cls, col in [("other", "#cccccc"), ("SDH", "#1f77b4"), ("EDH", "#d62728")]:
        m = lab == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=8, c=col, label=cls, alpha=0.6)
    ax.set_title("Stage-1 feature space (t-SNE): EDH vs SDH"); ax.legend()
    _save(fig, f"tsne_{feat_run}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--compare", default=None, help="second run template for method comparison")
    ap.add_argument("--features", default=None, help="features run for t-SNE")
    a = ap.parse_args()

    tag = "conditional"
    probs, targets, studies = load_pooled(a.runs, a.folds)
    print(f"loaded pooled OOF: {len(targets):,} studies\n")

    def safe(fn, *args, name=""):
        try: fn(*args)
        except Exception as e: print(f"  [skip {name}] {type(e).__name__}: {e}")

    # metrics table (the big CSV)
    safe(comprehensive_metrics, probs, targets, tag, name="metrics")
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
        # align by study
        ib = {s: i for i, s in enumerate(sb)}
        ra, rb = [], []
        for i, s in enumerate(studies):
            j = ib.get(s)
            if j is not None: ra.append(i); rb.append(j)
        ra, rb = np.array(ra), np.array(rb)
        safe(fig_method_compare, probs[ra], pb[rb], targets[ra],
             "Conditional", "Joint", tag, name="method_compare")

    if a.features:
        safe(fig_tsne, a.features, name="tsne")

    print(f"\nDONE. figures -> {FIGS}   metrics/csv -> {RESULTS}")


if __name__ == "__main__":
    main()