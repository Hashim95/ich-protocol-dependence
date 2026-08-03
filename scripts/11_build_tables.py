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

USAGE
    python 11_build_tables.py                 # all tables
    python 11_build_tables.py --tables 1 3    # selected tables
    python 11_build_tables.py --latex         # LaTeX only

OUTPUT
    results/tables/table{N}_*.md
    results/tables/table{N}_*.tex
    results/tables/ALL_TABLES.md      <- paste-ready
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
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n"
           + tex + ("\\\\[2pt]\n\\footnotesize " + note + "\n" if note else "")
           + "\\end{table}\n")
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

    return emit(df, "table2_cross_architecture",
                "Rare-subtype precision across architecture families under identical "
                "training conditions (final-epoch models, five folds each).", note)


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
        {"Aspect": "EDH AUC",
         "Original (as published)": "Not reported per class",
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
                "Per-fold EDH F1 varied by a factor of "
                f"{e['F1'].max()/max(e['F1'].min(),1e-9):.1f} "
                f"({e['F1'].min():.3f}–{e['F1'].max():.3f}) while average F1 remained "
                f"stable at {a['F1'].mean():.3f} ± {a['F1'].std(ddof=1):.3f}, indicating "
                "that single-partition rare-class estimates carry variance far larger "
                "than aggregate metrics. Original values are as published; the "
                "re-implementation substitutes standard acute-hemorrhage intensity "
                "thresholds for the original's unpublished values and uses a matched "
                "backbone.")


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


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()

    print(f"reading from {RESULTS}\n")
    blocks, fns = [], {1: table1, 2: table2, 3: table3, 4: table4, 5: table5}
    for n in a.tables:
        print(f"Table {n}:")
        b = fns[n]()
        if b:
            blocks.append(f"## Table {n}\n\n{b}\n")
        print()

    out = TABLES / "ALL_TABLES.md"
    out.write_text("# Paper 1 — Tables\n\n" + "\n---\n\n".join(blocks))
    print(f"[written] {out}")

    if _missing:
        print("\nNOT FOUND — these tables are incomplete:")
        for m in _missing:
            print(f"  - {m}")
    else:
        print("\nAll requested tables built from logged results.")


if __name__ == "__main__":
    main()