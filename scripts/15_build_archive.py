#!/usr/bin/env python3
"""
15_build_archive.py — assemble the Zenodo deposition from ivision.

Builds two archives, because they have different sizes, audiences and licences:

  ich-protocol-dependence-code-v1.0.0.zip   ~5 MB
      Scripts, the partition assignment, generated tables, LICENSE,
      CITATION.cff. This is what a reader needs to verify the split and rerun
      the pipeline. Cite this DOI in the manuscript.

  ich-protocol-dependence-weights-v1.0.0.zip   ~1 GB
      The five Stage-1 checkpoints and the Stage-2 heads. Linked to the code
      record as "is supplemented by".

WHY TWO
-------
A reviewer checking your central claim needs the fold table, not the weights.
Bundling a gigabyte of checkpoints in front of that makes the thing they
actually want harder to reach. Splitting also means the code record downloads
in seconds.

WHAT IS DELIBERATELY EXCLUDED
-----------------------------
  * The 333 GB memory-map -- regenerable from the DICOMs in one pass.
  * Cached features -- regenerable from the checkpoints.
  * Any DICOM, NIfTI or raw pixel data. RSNA, CQ500 and BHSD are redistributed
    by their own custodians under their own terms; putting slices in a Zenodo
    record would breach those terms and is not necessary for reproduction.
  * Prediction .npz files by default: they are derived artefacts, and including
    them lets someone reproduce your tables without rerunning anything, which
    is arguably a feature. Pass --with-predictions if you want them.

USAGE
    python 15_build_archive.py                 # code archive only
    python 15_build_archive.py --weights       # both
    python 15_build_archive.py --dry-run       # list contents, write nothing
"""
import argparse, hashlib, json, os, shutil, sys, zipfile
from datetime import date
from pathlib import Path

SCRIPTS = Path("/home/ivision/Documents/Hashim/VS Project files")
ROOT = Path(os.environ.get(
    "ICH_ROOT",
    "/home/ivision/Documents/Hashim/RSNA ICH Dataset/rsna-intracranial-hemorrhage-detection"))
OUT = Path.home() / "zenodo_deposition"
VERSION = "1.0.0"

# Scripts that constitute the pipeline. Diagnostics and superseded backups are
# excluded: a release should contain what was run, not the archaeology.
SCRIPT_FILES = [
    "ich_config.py", "windowing.py",
    "00_preprocess_to_memmap.py", "01_prepare_manifests.py", "01b_prepare_split.py",
    "02_train_stage1.py", "02b_train_stage1_supcon.py",
    "03a_extract_features.py", "03b_train_sala_head.py", "03c_conditional_subtype.py",
    "04_evaluate.py", "04b_full_metrics.py", "04c_slice_metrics.py",
    "05_sequence_smooth.py", "05b_smooth_any.py",
    "06_cq500_prepare.py", "06b_bhsd_prepare.py", "06_ensemble.py",
    "07_cq500_evaluate.py", "08_paper_figures.py", "09_clinical_utility.py",
    "10_wsgsa_baseline.py", "11_build_tables.py", "12_heldout_test.py",
    "13_qualitative_figure.py", "14_collect_for_paper.py", "15_build_archive.py",
    "diagnose_minority_fp.py",
]

MANIFESTS = ["folds.parquet", "heldout_test.parquet",
             "split_meta.json", "split_summary.csv"]

LICENSE = """MIT License

Copyright (c) {year} Hashim Ayub and Hasan Mujtaba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""".format(year=date.today().year)

CITATION = """cff-version: 1.2.0
message: "If you use this software, please cite the accompanying paper."
title: "Rare-Subtype Performance in Intracranial Hemorrhage Classification Is Protocol-Dependent"
version: "{v}"
date-released: "{d}"
license: MIT
repository-code: "https://github.com/Hashim95/ich-protocol-dependence"
authors:
  - family-names: "Ayub"
    given-names: "Hashim"
    email: "i220201@nu.edu.pk"
    affiliation: "National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad"
  - family-names: "Mujtaba"
    given-names: "Hasan"
    email: "hasan.mujtaba@nu.edu.pk"
    affiliation: "National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad"
keywords:
  - intracranial hemorrhage
  - computed tomography
  - class imbalance
  - evaluation methodology
  - reproducibility
abstract: >-
  Code and partition assignments for a systematic re-evaluation of rare-subtype
  performance in intracranial hemorrhage classification. Includes the exact
  patient-disjoint fold assignment (hash f72016749123e8ce) on which all reported
  results were computed.
""".format(v=VERSION, d=date.today().isoformat())

README = """# Rare-Subtype Performance in ICH Classification Is Protocol-Dependent

Code and partition assignments for the paper of the same name.

## The partition

`manifests/folds.parquet` and `manifests/heldout_test.parquet` contain study
identifiers and fold numbers only: no patient data, no images. They allow exact
reproduction of the partition on which every result in the paper is computed.
`01b_prepare_split.py` regenerates them deterministically from the RSNA study
manifest; the assignment hash `f72016749123e8ce` was verified identical on two
machines running different PyTorch versions.

## Pipeline, in execution order

    00_preprocess_to_memmap.py   DICOM -> uint8 memmap, shared windowing
    01_prepare_manifests.py      slice and study manifests
    01b_prepare_split.py         held-out test + 5 patient-disjoint folds
    02_train_stage1.py           slice classifier (--backbone, --loss)
    03a_extract_features.py      frozen backbone -> cached embeddings
    03c_conditional_subtype.py   Stage-2 conditional and joint heads
    04_evaluate.py               pooled OOF metrics + paired bootstrap
    05b_smooth_any.py            sequence smoothing, any prediction file
    06b_bhsd_prepare.py          BHSD external cohort
    07_cq500_evaluate.py         CQ500 external cohort
    08_paper_figures.py          figures and metrics tables
    09_clinical_utility.py       sens@spec, decision curves, workload
    10_wsgsa_baseline.py         base-method re-implementation
    11_build_tables.py           assembles all paper tables
    12_heldout_test.py           final evaluation, run once
    13/14_*.py                   qualitative figure, collected values

## Protocol

Fixed 10-epoch schedule, final-epoch model reported, no metric-based checkpoint
selection at any stage. All threshold-dependent metrics at a fixed 0.5
threshold. Metrics pooled out-of-fold. The held-out test set is absent from
`folds.parquet`, so training code selecting `fold != k` excludes it
structurally rather than by convention.

## Data

RSNA 2019, CQ500 and BHSD are available from their respective distributors and
are not redistributed here.

## Environment

See `requirements_ivision.txt` and `requirements_celsius.txt`. All folds of a
given experiment were trained on a single machine.

## Model weights

Archived separately; see the record linked as "is supplemented by".
"""


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="store_true", help="also build the weights archive")
    ap.add_argument("--with-predictions", action="store_true",
                    help="include val_predictions.npz files in the code archive")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    stage = OUT / "code"
    if not a.dry_run:
        if OUT.exists():
            shutil.rmtree(OUT)
        (stage / "scripts").mkdir(parents=True)
        (stage / "manifests").mkdir()
        (stage / "results" / "tables").mkdir(parents=True)

    manifest, missing = [], []

    def add(src, dst):
        src = Path(src)
        if not src.exists():
            missing.append(str(src)); return
        manifest.append((str(dst), src.stat().st_size, sha256(src) if not a.dry_run else ""))
        if not a.dry_run:
            (stage / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, stage / dst)

    print("=" * 66)
    print("CODE ARCHIVE")
    print("=" * 66)
    for f in SCRIPT_FILES:
        add(SCRIPTS / f, Path("scripts") / f)
    for f in MANIFESTS:
        add(ROOT / "manifests" / f, Path("manifests") / f)
    for f in sorted((ROOT / "results" / "tables").glob("*")):
        add(f, Path("results/tables") / f.name)
    for f in ["HELDOUT_TEST_RECEIPT.json", "heldout_test_metrics.csv",
              "heldout_test_per_fold.csv", "paper_collected_values.json",
              "qualitative_figure_studies.json", "fp_confusion_conditional.csv"]:
        add(ROOT / "results" / f, Path("results") / f)
    for f in sorted(Path.home().glob("requirements_*.txt")):
        add(f, Path(f.name))

    if a.with_predictions:
        for d in sorted((ROOT / "stage2_runs").glob("*/val_predictions.npz")):
            add(d, Path("results/predictions") / f"{d.parent.name}.npz")

    if not a.dry_run:
        (stage / "LICENSE").write_text(LICENSE)
        (stage / "CITATION.cff").write_text(CITATION)
        (stage / "README.md").write_text(README)
        (stage / "MANIFEST.sha256").write_text(
            "\n".join(f"{h}  {p}" for p, _, h in manifest if h) + "\n")

    tot = sum(sz for _, sz, _ in manifest)
    print(f"  {len(manifest)} files, {tot/1e6:.1f} MB")
    if missing:
        print(f"\n  NOT FOUND ({len(missing)}) -- check before publishing:")
        for m in missing:
            print("   ", m)

    if not a.dry_run:
        z = OUT / f"ich-protocol-dependence-code-v{VERSION}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(stage))
        print(f"\n  [written] {z}  ({z.stat().st_size/1e6:.1f} MB)")
        shutil.rmtree(stage)

    if a.weights:
        print("\n" + "=" * 66)
        print("WEIGHTS ARCHIVE")
        print("=" * 66)
        w = OUT / f"ich-protocol-dependence-weights-v{VERSION}.zip"
        n, tot = 0, 0
        if not a.dry_run:
            with zipfile.ZipFile(w, "w", zipfile.ZIP_DEFLATED) as zf:
                for pat, sub in [("stage1_runs/*/best.pt", "stage1"),
                                 ("stage2_runs/*/best.pt", "stage2")]:
                    for p in sorted(ROOT.glob(pat)):
                        zf.write(p, f"{sub}/{p.parent.name}.pt")
                        n += 1; tot += p.stat().st_size
            print(f"  {n} checkpoints, {tot/1e9:.2f} GB raw")
            print(f"  [written] {w}  ({w.stat().st_size/1e9:.2f} GB)")
        else:
            for pat in ["stage1_runs/*/best.pt", "stage2_runs/*/best.pt"]:
                for p in sorted(ROOT.glob(pat)):
                    n += 1; tot += p.stat().st_size
            print(f"  would include {n} checkpoints, {tot/1e9:.2f} GB")

    print("\n" + "=" * 66)
    print("NEXT STEPS ON ZENODO")
    print("=" * 66)
    print("""  1. zenodo.org -> New Upload -> Reserve DOI  (do this FIRST; the DOI
     goes into the submitted manuscript)
  2. Upload the code zip. Access -> Restricted.
  3. Second record for the weights zip, also Restricted. Under
     Related identifiers add the code DOI as "is supplemented by".
  4. Paste both DOIs into main.tex, Data and Code Availability.
  5. On acceptance: edit both records -> Access -> Open. DOIs do not change.""")


if __name__ == "__main__":
    main()