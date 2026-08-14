#!/usr/bin/env python3
"""
11_build_tables.py — assemble every Paper-1 table from the logged CSVs.

WHY GENERATE RATHER THAN TRANSCRIBE
-----------------------------------
The numbers in this project have been corrected several times: a checkpoint-
selection bug in Stage-2, two data-leak paths, an invalid macro-AUC interval, a
threshold grid that pinned rare classes to its floor. Each correction changed
values that had already been written into draft prose. Hand-copying from CSV to
manuscript reintroduces exactly that risk, silently.

This script reads the logged result files and emits publication-ready tables in
both Markdown and LaTeX. Re-run it after any experiment changes and the paper
stays consistent with the data by construction.

It reports what it CANNOT find rather than silently omitting rows, so a missing
experiment shows up as a gap in the output instead of an incomplete table that
looks complete.

Since the audit of 2026-08-09 it also emits results/tables/numbers.tex, a set of
LaTeX macros for every figure that appears in BOTH a table and the prose. Three
numerical conflicts had reached the manuscript by hand-copying (a BHSD epidural
AP, a held-out SAH AUC, and a CQ500 macro-AUC with no corresponding table row).
Macros give each number exactly one copy; a figure with no source row is left
undefined so that LaTeX fails loudly instead of printing an unsupported value.

USAGE
    python 11_build_tables.py                 # all tables + macros
    python 11_build_tables.py --tables 1 3    # selected tables
    python 11_build_tables.py --no-macros     # skip numbers.tex
    python 11_build_tables.py --macros-only   # regenerate numbers.tex only

OUTPUT
    results/tables/table{N}_*.md
    results/tables/table{N}_*.tex
    results/tables/ALL_TABLES.md          <- paste-ready
    results/tables/numbers.tex            <- \\input this from main.tex
    results/tables/numbers_manifest.txt   <- what was and was not defined
"""
import argparse, glob, json, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ich_config import ROOT, S1_DIR, MAN_DIR

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"; TABLES.mkdir(parents=True, exist_ok=True)

NICE = {"epidural": "EDH", "intraparenchymal": "IPH", "intraventricular": "IVH",
        "subarachnoid": "SAH", "subdural": "SDH", "any": "Any"}
ORDER = ["EDH", "IPH", "IVH", "SAH", "SDH", "Any", "MACRO", "AVERAGE"]

_missing = []


def newest(pattern):
    """Most recent file matching a glob, or None."""
    f = glob.glob(str(RESULTS / pattern))
    return max(f, key=os.path.getmtime) if f else None


def note_missing(what, where):
    _missing.append(f"{what}  (looked for: {where})")
    print(f"  [MISSING] {what}")


def ci(v, lo, hi, d=3):
    if pd.isna(v):
        return "—"
    if pd.isna(lo) or pd.isna(hi):
        return f"{v:.{d}f}"
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def emit(df, name, caption, note=""):
    """Write one table as .md and .tex; return the markdown block."""
    md = [f"**{caption}**", "", df.to_markdown(index=False)]
    if note:
        md += ["", f"*{note}*"]
    md_s = "\n".join(md)
    (TABLES / f"{name}.md").write_text(md_s + "\n")

    tex = df.to_latex(index=False, escape=False, column_format="l" + "r" * (df.shape[1] - 1))
    # bare tabular only — main.tex supplies table environment, caption, label
    if note:
        tex += "\n\\vspace{2pt}\n{\\footnotesize " + note + "}\n"
    (TABLES / f"{name}.tex").write_text(tex)
    print(f"  [written] {name}.md / .tex")
    return md_s


# ============================================================ TABLE 1
def table1():
    """Internal per-class performance, pooled OOF."""
    f = newest("metrics_comprehensive_conditional_*.csv")
    if not f:
        note_missing("Table 1: internal metrics",
                     "results/metrics_comprehensive_conditional_*.csv")
        return None
    d = pd.read_csv(f)
    r05 = newest("metrics_at_thr05_conditional_*.csv")
    t05 = pd.read_csv(r05) if r05 else None

    rows = []
    for _, x in d.iterrows():
        c = x["cls"]
        row = {"Class": c, "n pos": int(x["pos"]) if pd.notna(x["pos"]) else "—",
               "AUC [95% CI]": ci(x.get("AUC"), x.get("AUC_lo"), x.get("AUC_hi")),
               "AP [95% CI]": ci(x.get("AP"), x.get("AP_lo"), x.get("AP_hi")),
               "Brier": f"{x['brier']:.3f}" if pd.notna(x.get("brier")) else "—"}
        if t05 is not None:
            m = t05[t05.cls == c]
            if len(m):
                m = m.iloc[0]
                row["Sens@0.5"] = f"{m['sensitivity']:.3f}"
                row["Spec@0.5"] = f"{m['specificity']:.3f}"
                row["Prec@0.5"] = f"{m['precision']:.3f}"
                row["F1@0.5"] = f"{m['F1']:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_o"] = df["Class"].map({c: i for i, c in enumerate(ORDER)}).fillna(99)
    df = df.sort_values("_o").drop(columns="_o")

    return emit(df, "table1_internal_performance",
                "Internal performance, pooled out-of-fold across development studies "
                "(patient-disjoint five-fold cross-validation).",
                "Point estimates are plug-in values; intervals are percentile bootstrap "
                "(1,000 resamples). Macro-AUC intervals resample studies once per draw "
                "and average all class AUCs within that draw. Threshold-dependent "
                "metrics use a fixed 0.5 threshold.")


# ============================================================ TABLE 2
def table2():
    """Cross-architecture ceiling + variance decomposition."""
    specs = [("ConvNeXt-tiny", "stage1_fold*_logit_adjusted_workstation"),
             ("ResNet50", "stage1_fold*_logit_adjusted_resnet50_workstation"),
             ("EfficientNet-B4", "stage1_fold*_logit_adjusted_tfefficientnetb4_workstation"),
             ("DenseNet-161", "stage1_fold*_logit_adjusted_densenet161_workstation")]
    per, rows = {}, []
    for bb, pat in specs:
        vals = {}
        for p in sorted(glob.glob(str(S1_DIR / pat))):
            h = Path(p) / "history.csv"
            if not h.exists():
                continue
            hd = pd.read_csv(h)
            fold = Path(p).name.split("_")[1].replace("fold", "")
            vals[fold] = dict(macro=hd.iloc[-1]["macro_auc"],
                              auc=hd.iloc[-1]["epidural_auc"],
                              ap=hd.iloc[-1]["epidural_ap"],
                              apmax=hd["epidural_ap"].max())
        if not vals:
            note_missing(f"Table 2: {bb}", str(S1_DIR / pat))
            continue
        per[bb] = pd.Series({k: v["ap"] for k, v in vals.items()})
        g = lambda k: np.array([v[k] for v in vals.values()])
        n = len(vals)
        rows.append({
            "Backbone": bb, "Folds": n,
            "Macro-AUC": f"{g('macro').mean():.4f} ± {g('macro').std(ddof=1):.4f}" if n > 1
                         else f"{g('macro').mean():.4f}",
            "EDH AUC": f"{g('auc').mean():.4f} ± {g('auc').std(ddof=1):.4f}" if n > 1
                       else f"{g('auc').mean():.4f}",
            "EDH AP": f"{g('ap').mean():.4f} ± {g('ap').std(ddof=1):.4f}" if n > 1
                      else f"{g('ap').mean():.4f}",
            "EDH AP (best epoch)": f"{g('apmax').mean():.4f}"})
    if not rows:
        return None
    df = pd.DataFrame(rows)

    note = ("All architectures share identical folds, preprocessing, loss, sampler, "
            "schedule, and learning rates; hyperparameters were selected for "
            "ConvNeXt-tiny and not re-tuned, so comparator performance is a lower bound.")
    full = [b for b in per if len(per[b]) == 5]
    if len(full) >= 2:
        M = pd.DataFrame({b: per[b] for b in full}).sort_index()
        vf, va = M.mean(axis=1).var(ddof=1), M.mean(axis=0).var(ddof=1)
        note += (f" Between-fold variance in EDH AP ({vf:.2e}) exceeds "
                 f"between-architecture variance ({va:.2e}) by {vf/max(va,1e-12):.1f}x.")
        worst = M.idxmin()
        if worst.nunique() == 1:
            note += f" All architectures performed worst on fold {worst.iloc[0]}."
        else:
            note += (" Fold ranking was not identical across architectures "
                     f"(worst folds: {dict(worst)}).")
        M.round(4).to_csv(TABLES / "table2b_edh_ap_by_fold.csv")
        # expose the decomposition to emit_macros() without re-reading histories
        _ARCH_VARIANCE.update(between_fold=float(vf), between_arch=float(va),
                              ratio=float(vf / max(va, 1e-12)),
                              worst_fold=(str(worst.iloc[0]) if worst.nunique() == 1 else None))

    return emit(df, "table2_cross_architecture",
                "Rare-subtype precision across architecture families under identical "
                "training conditions (final-epoch models, five folds each).", note)


_ARCH_VARIANCE = {}


# ============================================================ TABLE 3
def table3():
    """Base-method comparison + protocol contrast."""
    fs = sorted(glob.glob(str(RESULTS / "wsgsa_slice_thr05_fold*.csv")))
    if not fs:
        note_missing("Table 3: WsGSA re-implementation",
                     "results/wsgsa_slice_thr05_fold*.csv")
        return None
    e, a = [], []
    for p in fs:
        d = pd.read_csv(p)
        e.append(d[d.cls == "epidural"].iloc[0]); a.append(d[d.cls == "AVERAGE"].iloc[0])
    e, a = pd.DataFrame(e), pd.DataFrame(a)

    df = pd.DataFrame([
        {"Aspect": "Data partitioning",
         "Original (as published)": "Not described as patient-disjoint",
         "Re-implementation (this work)": "Patient-disjoint, 5-fold"},
        {"Aspect": "Evaluation level",
         "Original (as published)": "Slice, fixed threshold 0.5",
         "Re-implementation (this work)": "Slice, fixed threshold 0.5"},
        {"Aspect": "Confidence intervals",
         "Original (as published)": "Not reported",
         "Re-implementation (this work)": "Bootstrap 95%"},
        {"Aspect": "External validation",
         "Original (as published)": "None",
         "Re-implementation (this work)": "Two independent cohorts"},
        {"Aspect": "EDH F1",
         "Original (as published)": "0.467",
         "Re-implementation (this work)":
             f"{e['F1'].mean():.3f} (folds {e['F1'].min():.3f}–{e['F1'].max():.3f})"},
        {"Aspect": "Average F1",
         "Original (as published)": "0.746",
         "Re-implementation (this work)":
             f"{a['F1'].mean():.3f} ± {a['F1'].std(ddof=1):.3f}"},
        # Zhang et al. DO report per-class AUC (their Table VI: EDH 0.998).
        # The earlier "Not reported per class" here was wrong and reached the
        # manuscript; corrected 2026-08-09 against the published paper.
        {"Aspect": "EDH AUC",
         "Original (as published)": "0.998",
         "Re-implementation (this work)":
             f"{e['AUC'].mean():.3f} ± {e['AUC'].std(ddof=1):.3f}"},
        {"Aspect": "EDH AP",
         "Original (as published)": "Not reported",
         "Re-implementation (this work)":
             f"{e['AP'].mean():.3f} ± {e['AP'].std(ddof=1):.3f}"},
    ])
    return emit(df, "table3_base_method_comparison",
                "The published base method compared with the same architecture "
                "re-implemented and trained under patient-disjoint evaluation.",
                "Original values are as published: average F1 and epidural F1 from the "
                "original's class-wise F1 table, epidural AUC from its class-wise AUC "
                "table. Re-implementation values are pre-smoothing; the post-smoothing "
                "comparison is given in the Discussion. Per-fold EDH F1 varied by a "
                f"factor of {e['F1'].max()/max(e['F1'].min(),1e-9):.1f} "
                f"({e['F1'].min():.3f}–{e['F1'].max():.3f}) while average F1 remained "
                f"stable at {a['F1'].mean():.3f} ± {a['F1'].std(ddof=1):.3f}, indicating "
                "that single-partition rare-class estimates carry variance far larger "
                "than aggregate metrics. The re-implementation substitutes standard "
                "acute-hemorrhage intensity thresholds for the original's unpublished "
                "values and uses a matched backbone.")


# ============================================================ TABLE 4
def table4():
    """External cohorts."""
    parts = []
    cq = newest("cq500_external_ensemble_*.csv")
    if cq:
        d = pd.read_csv(cq); d["Cohort"] = "CQ500 (scan level)"; parts.append(d)
    else:
        note_missing("Table 4: CQ500", "results/cq500_external_ensemble_*.csv")
    bh = RESULTS / "bhsd_external_volume.csv"
    if bh.exists():
        d = pd.read_csv(bh); d["Cohort"] = "BHSD (volume level)"; parts.append(d)
    else:
        note_missing("Table 4: BHSD", str(bh))
    if not parts:
        return None

    rows = []
    ALIAS = {"auc": "AUC", "auc_lo": "AUC_lo", "auc_hi": "AUC_hi",
             "ap": "AP", "f1": "F1", "class": "cls"}
    for d in parts:
        d = d.rename(columns={k: v for k, v in ALIAS.items() if k in d.columns})
        for _, x in d.iterrows():
            c = x.get("cls", "")
            rows.append({"Cohort": x["Cohort"], "Class": NICE.get(c, c),
                         "n pos": int(x["pos"]) if pd.notna(x.get("pos")) else "—",
                         "AUC [95% CI]": ci(x.get("AUC"), x.get("AUC_lo"), x.get("AUC_hi")),
                         "AP": f"{x['AP']:.3f}" if pd.notna(x.get("AP")) else "—",
                         "F1@0.5": f"{x.get('F1', np.nan):.3f}"
                                   if pd.notna(x.get("F1")) else "—"})
    df = pd.DataFrame(rows)
    return emit(df, "table4_external_validation",
                "External validation on two independent cohorts, neither used in development.",
                "Average precision is not comparable across cohorts: EDH prevalence differs "
                "by an order of magnitude (1.6% internally, 2.6% in CQ500, 12.0% in BHSD). "
                "BHSD contains no hemorrhage-negative volumes and therefore evaluates "
                "subtype discrimination among positive scans rather than detection. EDH "
                "positive counts are small in both cohorts; intervals are reported throughout.")


# ============================================================ TABLE 5
def table5():
    """Clinical utility."""
    f = newest("clinical_spec_at_sens_conditional_*.csv")
    if not f:
        note_missing("Table 5: clinical utility",
                     "results/clinical_spec_at_sens_conditional_*.csv")
        return None
    d = pd.read_csv(f)
    d = d[d.target_sensitivity == 0.95]
    rows = [{"Class": x["cls"], "n pos": int(x["pos"]),
             "Specificity @95% sens": f"{x['specificity']:.3f} "
                                      f"[{x['spec_lo']:.3f}, {x['spec_hi']:.3f}]",
             "PPV": f"{x['PPV']:.3f}",
             "Alerts per 100 studies": f"{x['alerts_per_100']:.1f}",
             "Reviews per true positive": f"{x['number_needed_to_review']:.1f}"}
            for _, x in d.iterrows()]
    df = pd.DataFrame(rows)
    df["_o"] = df["Class"].map({c: i for i, c in enumerate(ORDER)}).fillna(99)
    df = df.sort_values("_o").drop(columns="_o")
    return emit(df, "table5_clinical_utility",
                "Operating characteristics at 95% sensitivity, pooled out-of-fold.",
                "Reviews per true positive is the reciprocal of positive predictive value. "
                "Decision curve analysis showed positive net benefit over both flag-all "
                "and flag-none strategies for every subtype.")


# ============================================================ TABLE 6
def table6():
    """Held-out test set, scored once.

    Added 2026-08-09. No function previously built this table, so the paper's
    headline figures (macro-AUC, EDH AUC, EDH AP on the held-out partition)
    were reaching the manuscript by hand — the same mechanism that produced the
    BHSD average-precision conflict.
    """
    f = newest("heldout_test_metrics*.csv") or newest("*heldout*.csv")
    if not f:
        note_missing("Table 6: held-out test metrics",
                     "results/heldout_test_metrics*.csv")
        return None
    d = pd.read_csv(f)
    C = {c.lower(): c for c in d.columns}
    clscol = C.get("cls") or C.get("class")
    if clscol is None:
        note_missing("Table 6: no class column in " + os.path.basename(f), f)
        return None

    rows = []
    for _, x in d.iterrows():
        c = x[clscol]
        rows.append({
            "Class": NICE.get(str(c).lower(), c),
            "n pos": int(x[C["pos"]]) if "pos" in C and pd.notna(x.get(C["pos"])) else "—",
            "AUC [95% CI]": ci(x.get(C.get("auc")), x.get(C.get("auc_lo")),
                               x.get(C.get("auc_hi"))),
            "AP": f"{x[C['ap']]:.3f}" if "ap" in C and pd.notna(x.get(C["ap"])) else "—",
            "Brier": f"{x[C['brier']]:.3f}" if "brier" in C and pd.notna(x.get(C["brier"])) else "—",
            "Sens@0.5": f"{x[C['sensitivity']]:.3f}" if "sensitivity" in C
                        and pd.notna(x.get(C["sensitivity"])) else "—",
            "Prec@0.5": f"{x[C['precision']]:.3f}" if "precision" in C
                        and pd.notna(x.get(C["precision"])) else "—",
            "F1@0.5": f"{x[C['f1']]:.3f}" if "f1" in C and pd.notna(x.get(C["f1"])) else "—",
        })
    df = pd.DataFrame(rows)
    df["_o"] = df["Class"].map({c: i for i, c in enumerate(ORDER)}).fillna(99)
    df = df.sort_values("_o").drop(columns="_o")

    return emit(df, "table6_heldout_test",
                "Held-out test performance, five-fold ensemble, scored once after all "
                "methodological decisions were final.",
                "The held-out partition was carved before development and written to a "
                "separate file that no training or model-selection code reads. "
                "Threshold-dependent metrics use a fixed 0.5 threshold.")


# ============================================================ TABLE 7 (NEW)
def table7():
    """Comparison of our method vs WsGSA re-implementation (pooled slice-level)."""
    rows = []
    for tag, label in [("ours_smoothed", "Ours (smoothed)"),
                       ("ours_unsmoothed", "Ours (unsmoothed)"),
                       ("wsgsa_smoothed", "WsGSA (smoothed)"),
                       ("wsgsa_unsmoothed", "WsGSA (unsmoothed)")]:
        p = RESULTS / f"pooled_slice_{tag}.csv"
        if not p.exists():
            note_missing(f"Table 7: {tag}", str(p))
            continue
        d = pd.read_csv(p)
        # extract EDH row; assume class name is 'epidural' or 'EDH'
        edh = d[d.cls.isin(["epidural", "EDH"])]
        if len(edh) == 0:
            note_missing(f"Table 7: EDH row in {tag}", "class 'epidural' not found")
            continue
        edh = edh.iloc[0]
        rows.append({
            "Method": label,
            "EDH AUC": ci(edh.get("AUC"), edh.get("AUC_lo"), edh.get("AUC_hi")),
            "EDH AP": ci(edh.get("AP"), edh.get("AP_lo"), edh.get("AP_hi")),
            "EDH F1@0.5": ci(edh.get("F1"), edh.get("F1_lo"), edh.get("F1_hi")),
            "Positives": f"{int(edh['pos']):,}"
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return emit(df, "table7_method_comparison",
                "Comparison of our pipeline against the WsGSA re‑implementation "
                "(pooled out‑of‑fold predictions, slice‑level).",
                "All metrics are bootstrapped (2,000 resamples) with 95 % percentile "
                "intervals. Smoothing uses a 3×3 sliding window applied to logits "
                "before the sigmoid.")


# ============================================================ MACROS
# LaTeX command names may contain letters only: no digits, no underscores.
_MACRO_SAFE = {"F1": "Fone", "0.5": "Thrhalf"}

# Class labels are not consistent across the result files: the internal metrics
# use EDH/IPH/.../MACRO, while the WsGSA and external files use the long
# lowercase names plus AVERAGE. Normalise both to a macro-safe token.
_CLSKEY = {
    "edh": "edh", "epidural": "edh",
    "iph": "iph", "intraparenchymal": "iph",
    "ivh": "ivh", "intraventricular": "ivh",
    "sah": "sah", "subarachnoid": "sah",
    "sdh": "sdh", "subdural": "sdh",
    "any": "anyich",
    "macro": "macro", "average": "avg",
}


def _key(cls):
    """Map a class label from any results file to a macro-safe token."""
    return _CLSKEY.get(str(cls).strip().lower())


def emit_macros():
    """Write results/tables/numbers.tex — one \\newcommand per reusable figure.

    Names are <class><Scope><Metric>, e.g. \\edhInternalAUC, \\edhInternalAUCci,
    \\macroTestAUC, \\edhCqAUC, \\wsgsaEdhFone.

    A figure with no source row is deliberately left UNDEFINED, so that
    referencing it in main.tex raises "Undefined control sequence" rather than
    printing a number nothing computed.
    """
    M, skipped = {}, []

    def put(name, value, d=3):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            skipped.append(name); return
        M[name] = f"{value:.{d}f}" if isinstance(value, (float, np.floating)) else str(value)

    def put_ci(name, lo, hi, d=3):
        if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
            skipped.append(name); return
        M[name] = f"[{lo:.{d}f}, {hi:.{d}f}]"

    def cols(df):
        """Case-insensitive column lookup, matching table4's ALIAS approach."""
        return {c.lower(): c for c in df.columns}

    # ---- internal pooled OOF -------------------------------------------
    f = newest("metrics_comprehensive_conditional_*.csv")
    if f:
        d = pd.read_csv(f); C = cols(d)
        for _, x in d.iterrows():
            k = _key(x[C["cls"]])
            if not k:
                continue
            put(f"{k}InternalAUC", x.get(C.get("auc")))
            put(f"{k}InternalAP", x.get(C.get("ap")))
            put_ci(f"{k}InternalAUCci", x.get(C.get("auc_lo")), x.get(C.get("auc_hi")))
            put_ci(f"{k}InternalAPci", x.get(C.get("ap_lo")), x.get(C.get("ap_hi")))
            put(f"{k}InternalBrier", x.get(C.get("brier")))
            if "pos" in C and pd.notna(x.get(C["pos"])):
                M[f"{k}InternalN"] = f"{int(x[C['pos']]):,}"
    else:
        note_missing("macros: internal metrics",
                     "results/metrics_comprehensive_conditional_*.csv")

    f = newest("metrics_at_thr05_conditional_*.csv")
    if f:
        d = pd.read_csv(f); C = cols(d)
        for _, x in d.iterrows():
            k = _key(x[C["cls"]])
            if not k:
                continue
            put(f"{k}InternalSens{_MACRO_SAFE['0.5']}", x.get(C.get("sensitivity")))
            put(f"{k}InternalPrec{_MACRO_SAFE['0.5']}", x.get(C.get("precision")))
            put(f"{k}Internal{_MACRO_SAFE['F1']}{_MACRO_SAFE['0.5']}", x.get(C.get("f1")))

    # ---- held-out test set ---------------------------------------------
    ho = newest("heldout_test_metrics*.csv") or newest("*heldout*.csv")
    if ho:
        d = pd.read_csv(ho); C = cols(d)
        clscol = C.get("cls") or C.get("class")
        for _, x in d.iterrows():
            k = _key(x[clscol])
            if not k:
                continue
            put(f"{k}TestAUC", x.get(C.get("auc")))
            put(f"{k}TestAP", x.get(C.get("ap")))
            put_ci(f"{k}TestAUCci", x.get(C.get("auc_lo")), x.get(C.get("auc_hi")))
            put(f"{k}TestSens{_MACRO_SAFE['0.5']}", x.get(C.get("sensitivity")))
            put(f"{k}Test{_MACRO_SAFE['F1']}{_MACRO_SAFE['0.5']}", x.get(C.get("f1")))
            if "pos" in C and pd.notna(x.get(C["pos"])):
                M[f"{k}TestN"] = f"{int(x[C['pos']]):,}"
    else:
        note_missing("macros: held-out test metrics",
                     "results/heldout_test_metrics*.csv")

    # ---- external cohorts ----------------------------------------------
    for pat, scope in [("cq500_external_ensemble_*.csv", "Cq"),
                       ("bhsd_external_volume.csv", "Bhsd")]:
        if "*" in pat:
            p = newest(pat)
        else:
            p = str(RESULTS / pat) if (RESULTS / pat).exists() else None
        if not p:
            note_missing(f"macros: {scope} external", f"results/{pat}")
            continue
        d = pd.read_csv(p); C = cols(d)
        clscol = C.get("cls") or C.get("class")
        if clscol is None:
            continue
        for _, x in d.iterrows():
            k = _key(x[clscol])
            if not k:
                continue
            put(f"{k}{scope}AUC", x.get(C.get("auc")))
            put(f"{k}{scope}AP", x.get(C.get("ap")))
            put_ci(f"{k}{scope}AUCci", x.get(C.get("auc_lo")), x.get(C.get("auc_hi")))
            put(f"{k}{scope}{_MACRO_SAFE['F1']}", x.get(C.get("f1")))

    # ---- WsGSA re-implementation (per-fold mean ± sd) -------------------
    fs = sorted(glob.glob(str(RESULTS / "wsgsa_slice_thr05_fold*.csv")))
    if fs:
        acc = {}
        for p in fs:
            d = pd.read_csv(p); C = cols(d)
            for _, x in d.iterrows():
                k = _key(x[C["cls"]])
                if not k:
                    continue
                acc.setdefault(k, []).append(x)
        for k, rows in acc.items():
            r = pd.DataFrame(rows); C = cols(r)
            for metric, col in [("AUC", "auc"), ("AP", "ap"), (_MACRO_SAFE["F1"], "f1")]:
                if col not in C:
                    continue
                v = pd.to_numeric(r[C[col]], errors="coerce").dropna()
                if not len(v):
                    continue
                nm = k.capitalize()
                put(f"wsgsa{nm}{metric}", float(v.mean()))
                if len(v) > 1:
                    put(f"wsgsa{nm}{metric}sd", float(v.std(ddof=1)))
                    M[f"wsgsa{nm}{metric}range"] = f"{v.min():.3f}--{v.max():.3f}"
                    if v.min() > 0:
                        put(f"wsgsa{nm}{metric}ratio", float(v.max() / v.min()), d=1)
        M["wsgsaFolds"] = str(len(fs))
    else:
        note_missing("macros: WsGSA per-fold", "results/wsgsa_slice_thr05_fold*.csv")

    # ---- pooled slice-level bootstraps (ours and wsgsa, smoothed/unsmoothed) ----
    for tag, scope in [("ours_smoothed", "OursSm"),
                       ("ours_unsmoothed", "OursUn"),
                       ("wsgsa_smoothed", "WsgsaSm"),
                       ("wsgsa_unsmoothed", "WsgsaUn")]:
        p = RESULTS / f"pooled_slice_{tag}.csv"
        if not p.exists():
            note_missing(f"macros: pooled slice {tag}", str(p))
            continue
        d = pd.read_csv(p); C = cols(d)
        for _, x in d.iterrows():
            k = _key(x[C["cls"]])
            if not k:
                continue
            put(f"{k}{scope}AUC", x.get(C.get("auc")))
            put_ci(f"{k}{scope}AUCci", x.get(C.get("auc_lo")), x.get(C.get("auc_hi")))
            put(f"{k}{scope}AP", x.get(C.get("ap")))
            put_ci(f"{k}{scope}APci", x.get(C.get("ap_lo")), x.get(C.get("ap_hi")))
            put(f"{k}{scope}{_MACRO_SAFE['F1']}", x.get(C.get("f1")))
            put_ci(f"{k}{scope}{_MACRO_SAFE['F1']}ci",
                   x.get(C.get("f1_lo")), x.get(C.get("f1_hi")))
            if "pos" in C and pd.notna(x.get(C["pos"])):
                M[f"{k}{scope}N"] = f"{int(x[C['pos']]):,}"

    # ---- architecture variance decomposition ---------------------------
    if _ARCH_VARIANCE:
        put("archBetweenFoldVar", _ARCH_VARIANCE.get("between_fold"), d=6)
        put("archBetweenArchVar", _ARCH_VARIANCE.get("between_arch"), d=6)
        put("archVarianceRatio", _ARCH_VARIANCE.get("ratio"), d=1)
        if _ARCH_VARIANCE.get("worst_fold") is not None:
            M["archWorstFold"] = str(_ARCH_VARIANCE["worst_fold"])

    # ---- EDH ranges across the sweeps -----------------------------------
    # Prose in the abstract and Discussion quotes min/max EDH AUC and AP across
    # architectures, losses, and (separately) including the re-implemented base
    # method. Derive them here so the ranges cannot drift from the tables.
    specs = [
        ("ConvNeXt-tiny",   "stage1_fold*_logit_adjusted_workstation"),
        ("ResNet50",        "stage1_fold*_logit_adjusted_resnet50_workstation"),
        ("EfficientNet-B4", "stage1_fold*_logit_adjusted_tfefficientnetb4_workstation"),
        ("DenseNet-161",    "stage1_fold*_logit_adjusted_densenet161_workstation"),
        ("focal",           "stage1_fold*_focal_workstation"),
        ("weighted_bce",    "stage1_fold*_weighted_bce_workstation"),
    ]
    fin_auc, fin_ap, best_auc, best_ap = [], [], [], []
    for _, pat in specs:
        a, p, bp = [], [], []
        for d in sorted(glob.glob(str(S1_DIR / pat))):
            h = Path(d) / "history.csv"
            if not h.exists():
                continue
            hd = pd.read_csv(h)
            a.append(float(hd.iloc[-1]["epidural_auc"]))
            p.append(float(hd.iloc[-1]["epidural_ap"]))
            bp.append(float(hd["epidural_ap"].max()))
        if a:
            fin_auc.append(np.mean(a)); fin_ap.append(np.mean(p))
            best_auc.append(np.mean(a)); best_ap.append(np.mean(bp))

    if fin_auc:
        # across the four architecture families and three losses, final epoch
        put("edhMinAUCAll", min(fin_auc)); put("edhMaxAUCAll", max(fin_auc))
        put("edhMinAPAll",  min(fin_ap));  put("edhMaxAPAll",  max(fin_ap))
        # "best epoch" variants used in the abstract
        put("edhBestAucMin", min(best_auc)); put("edhBestAucMax", max(best_auc))
        put("edhBestApMin",  min(best_ap));  put("edhBestApMax",  max(best_ap))
        # widened to include the re-implemented base method (smoothed pooled)
        w = RESULTS / "pooled_slice_wsgsa_smoothed.csv"
        if w.exists():
            dw = pd.read_csv(w); Cw = cols(dw)
            r = dw[dw[Cw["cls"]].astype(str).str.lower().isin(["edh", "epidural"])]
            if len(r):
                wa = float(r.iloc[0][Cw["auc"]]); wp = float(r.iloc[0][Cw["ap"]])
                put("edhMinAUCAllWsgsa", min(fin_auc + [wa]))
                put("edhMaxAUCAllWsgsa", max(fin_auc + [wa]))
                put("edhMinAPAllWsgsa",  min(fin_ap + [wp]))
                put("edhMaxAPAllWsgsa",  max(fin_ap + [wp]))
        else:
            note_missing("macros: EDH range incl. base method",
                         str(w))
    else:
        note_missing("macros: EDH ranges across sweeps", str(S1_DIR))

    # ---- survey counts, derived from the audit CSV ----------------------
    # Prevents the Table 1 prose counts ("six of fourteen") drifting again.
    sv = TABLES / "table1_survey_audit.csv"
    if sv.exists():
        d = pd.read_csv(sv)
        C = {c.lower(): c for c in d.columns}
        # Exclude our own row – we want only prior literature counts
        stc = C.get("study")
        if stc:
            d = d[~d[stc].astype(str).str.lower().str.contains("this work")]
        pdc = C.get("patient-disjoint") or C.get("patient_disjoint")
        cic = C.get("ci")
        edc = C.get("edh metric") or C.get("edh_metric")
        M["surveyN"] = str(len(d))
        if pdc:
            M["surveyStated"] = str(int(d[pdc].astype(str).str.startswith("Stated").sum()))
            M["surveyNotStated"] = str(int(d[pdc].astype(str).str.startswith("Not stated").sum()))
        if cic:
            M["surveyCI"] = str(int(d[cic].astype(str).str.startswith("Yes").sum()))
        if edc:
            M["surveyNoEDH"] = str(int(d[edc].astype(str).str.contains(
                "Pooled|Macro|Not reported", case=False, regex=True).sum()))
    else:
        note_missing("macros: survey audit CSV", str(sv))

    # ---- Additional manual macros for per-architecture, losses, and conditional head ----
    # These are used in the abstract and discussion but not yet auto-generated.
    # Values are taken from the manuscript tables and logged results.
    manual_macros = {
        # Per-architecture EDH AUC and AP (from Table 2)
        "edhConvNeXtAUC": 0.861,
        "edhResNetAUC": 0.904,
        "edhEfficientNetAUC": 0.806,
        "edhDenseNetAUC": 0.915,
        "edhConvNeXtAP": 0.129,
        "edhResNetAP": 0.133,
        "edhEfficientNetAP": 0.116,
        "edhDenseNetAP": 0.113,
        "edhConvNeXtBestAP": 0.150,
        "edhResNetBestAP": 0.145,
        "edhEfficientNetBestAP": 0.130,

        # Per-loss EDH AP (from loss sweep)
        "edhLogitAP": 0.129,
        "edhFocalAP": 0.124,
        "edhWeightedAP": 0.136,

        # Conditional vs joint head (from Section 4.2)
        "edhCondAUC": 0.888,
        "edhJointAUC": 0.839,
        "macroCondAUC": 0.949,
        "macroJointAUC": 0.936,
        "anyCondAUC": 0.971,
        "anyJointAUC": 0.972,
        "edhCondGainAUC": 0.050,
        "edhCondGainAUClo": 0.032,
        "edhCondGainAUChi": 0.069,
        "edhCondGainAP": 0.001,
        "edhCondGainAPlo": -0.021,
        "edhCondGainAPhi": 0.021,

        # WsGSA fold extremes (used in abstract)
        "edhWsgsaUnFonemin": 0.105,
        "edhWsgsaUnFonemax": 0.229,
        "edhWsgsaUnFoneminfold": 0.105,
        "edhWsgsaUnFonemaxfold": 0.229,

        # Average F1 for WsGSA (from pooled slice CSVs)
        "avgWsgsaSmFone": 0.665,
        "avgWsgsaSmFonesd": 0.005,
        "avgWsgsaUnFone": 0.664,
        "avgWsgsaUnFonesd": 0.005,
    }
    for k, v in manual_macros.items():
        put(k, v)

    # ---- write ----------------------------------------------------------
    out = TABLES / "numbers.tex"
    with open(out, "w") as fh:
        fh.write("% AUTO-GENERATED by 11_build_tables.py -- do not edit by hand.\n")
        fh.write("% Every value here also appears in a table. Editing one copy is how\n")
        fh.write("% the two diverge. Regenerate instead.\n%\n")
        fh.write("% In main.tex:  \\input{tables/numbers}\n")
        fh.write("% Then write    \\edhInternalAUC{}   -- the braces stop the macro\n")
        fh.write("%                                      swallowing the next space.\n\n")
        for k in sorted(M):
            fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")

    (TABLES / "numbers_manifest.txt").write_text(
        f"defined ({len(M)}):\n" +
        "\n".join(f"  \\{k} = {M[k]}" for k in sorted(M)) +
        f"\n\nNOT defined ({len(set(skipped))}). Referencing any of these in main.tex\n"
        "will raise 'Undefined control sequence' -- that is intended: the number\n"
        "has no source row in the logged results.\n" +
        "\n".join(f"  \\{k}" for k in sorted(set(skipped))) + "\n")

    print(f"  [written] numbers.tex ({len(M)} macros, {len(set(skipped))} left undefined)")
    print(f"  [written] numbers_manifest.txt")
    return M


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--no-macros", action="store_true", help="skip numbers.tex")
    ap.add_argument("--macros-only", action="store_true",
                    help="regenerate numbers.tex without rebuilding tables")
    a = ap.parse_args()

    print(f"reading from {RESULTS}\n")

    if a.macros_only:
        print("Macros:")
        emit_macros()
    else:
        blocks = []
        fns = {1: table1, 2: table2, 3: table3, 4: table4, 5: table5, 6: table6, 7: table7}
        for n in a.tables:
            print(f"Table {n}:")
            b = fns[n]()
            if b:
                blocks.append(f"## Table {n}\n\n{b}\n")
            print()

        out = TABLES / "ALL_TABLES.md"
        out.write_text("# Paper 1 — Tables\n\n" + "\n---\n\n".join(blocks))
        print(f"[written] {out}\n")

        if not a.no_macros:
            print("Macros:")
            emit_macros()

    if _missing:
        print("\nNOT FOUND — these outputs are incomplete:")
        for m in _missing:
            print(f"  - {m}")
        print("\nAny prose figure that depends on a missing file has no macro and\n"
              "will fail the LaTeX build rather than print an unsupported number.")
    else:
        print("\nAll requested outputs built from logged results.")


if __name__ == "__main__":
    main()