# Rare-Subtype Performance in Intracranial Hemorrhage Classification Is Protocol-Dependent

Code and data-partition assignments for the paper of the same name.

## What is here

- `manifests/folds.parquet` — the 17,396 development studies with fold assignment
- `manifests/heldout_test.parquet` — the 4,348 held-out test studies
- `manifests/split_meta.json` — seed, partition sizes, assignment hash `f72016749123e8ce`
- `scripts/` — full pipeline, numbered in execution order
- `results/tables/` — all paper tables, generated from logged results

The manifests contain study identifiers and fold numbers only: no patient data
and no images. They allow exact reproduction of the partition on which every
result in the paper is computed. `01b_prepare_split.py` regenerates them
deterministically from the RSNA study manifest; the hash was verified identical
on two machines running different PyTorch versions.

## Pipeline
00_preprocess_to_memmap.py   DICOM -> uint8 memmap, shared windowing
01_prepare_manifests.py      slice/study manifests
01b_prepare_split.py         held-out test + 5 patient-disjoint folds
02_train_stage1.py           slice classifier (--backbone, --loss)
03a_extract_features.py      frozen backbone -> cached embeddings
03c_conditional_subtype.py   Stage-2 conditional / joint heads
04_evaluate.py               pooled OOF metrics + paired bootstrap
05b_smooth_any.py            sequence smoothing for any prediction file
06b_bhsd_prepare.py          BHSD external cohort
07_cq500_evaluate.py         CQ500 external cohort
08_paper_figures.py          figures + metrics tables
09_clinical_utility.py       sens@spec, decision curves, workload
10_wsgsa_baseline.py         base-method re-implementation
11_build_tables.py           assembles all paper tables


## Protocol

Fixed 10-epoch schedule, final-epoch model reported, no metric-based checkpoint
selection. All threshold-dependent metrics at a fixed 0.5 threshold. Metrics
pooled out-of-fold. The held-out test set is absent from `folds.parquet`, so
training code selecting `fold != k` excludes it structurally.

## Environments

PyTorch 2.6.0+cu124 (primary) and 2.13.0+cu130 (base-method re-implementation);
timm 1.0.28, NumPy 2.4.4, pandas 3.0.3, scikit-learn 1.9.0, nibabel 5.4.2,
OpenCV 5.0.0. See `requirements_*.txt`.

## Data

RSNA 2019, CQ500, and BHSD are available from their respective distributors and
are not redistributed here.

## Citation

[to be added on acceptance]
