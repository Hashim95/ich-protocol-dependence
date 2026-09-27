# Rare-Subtype Performance in Intracranial Hemorrhage Classification Is Protocol-Dependent: Evidence for a Data-Level Precision Ceiling

Code and data-partition assignments for the paper of the same name
(Ayub & Mujtaba). This deposit lets a reader reproduce every reported result on
the exact patient-disjoint partition the paper uses.

## What is here

- `manifests/folds.parquet` — the 17,396 development studies with fold assignment
- `manifests/heldout_test.parquet` — the 4,348 held-out test studies
- `manifests/split_meta.json` — seed, partition sizes, assignment hash `f72016749123e8ce`
- `manifests/split_summary.csv` — per-fold counts
- `scripts/` — the full pipeline, numbered in execution order
- `results/tables/` — every paper table, generated from logged results
- `requirements.txt` — pinned dependency versions
- `LICENSE` — MIT (code only; the imaging cohorts are not redistributed)

The manifests contain study identifiers and fold numbers only: no patient data
and no images. They allow exact reproduction of the partition on which every
result in the paper is computed. `01b_prepare_split.py` regenerates them
deterministically from the RSNA study manifest; the assignment hash was verified
identical on two machines running different PyTorch versions.

## Pipeline

Shared modules (imported, not run directly):

    ich_config.py                paths, constants, ALL_COLS, EDH index
    windowing.py                 HU windowing shared across all cohorts

Preprocessing and partition:

    00_preprocess_to_memmap.py   DICOM -> uint8 memmap, shared windowing
    01_prepare_manifests.py      slice- and study-level manifests
    01b_prepare_split.py         held-out test + 5 patient-disjoint folds

Stage 1 (slice representation):

    02_train_stage1.py           six-way slice classifier (--backbone, --loss)
    02b_train_stage1_supcon.py   supervised-contrastive variant
    03a_extract_features.py      frozen backbone -> cached study embeddings

Stage 2 (study-level heads):

    03b_train_sala_head.py       attention-pooling head
    03c_conditional_subtype.py   conditional / joint detect-then-subtype heads

Evaluation:

    04_evaluate.py               pooled out-of-fold metrics + paired bootstrap
    04b_full_metrics.py          complete per-class metric tables
    04c_slice_metrics.py         slice-level metrics
    05_sequence_smooth.py        Gaussian sequence smoothing over geometric z
    05b_smooth_any.py            smoothing for an arbitrary prediction file
    06_ensemble.py               five-fold ensemble for the held-out test

External cohorts:

    06_cq500_prepare.py          CQ500 ingest
    06b_bhsd_prepare.py          BHSD ingest
    07_cq500_evaluate.py         CQ500 external evaluation

Base method and held-out test:

    10_wsgsa_baseline.py         re-implementation of Zhang et al. (2023)
    12_heldout_test.py           score the sealed test set once

Figures, tables, and analyses:

    08_paper_figures.py          ROC, PR, calibration, method comparison, t-SNE
    09_clinical_utility.py       sens@spec, decision curves, triage workload
    11_build_tables.py           assemble all paper tables; emit numbers.tex
    13_qualitative_figure.py     representative-case analysis
    diagnose_minority_fp.py      EDH false-positive subtype enrichment (3.9x)
    dump_stage2_attention.py     Stage-2 attention profile for Figure 3(a)
    make_survey_audit.py         Table 1 survey audit CSV, derived from the paper
    generate_slice_preds.py      per-slice predictions from a trained checkpoint

Collection:

    14_collect_for_paper.py      gather figures/tables for the manuscript
    15_build_archive.py          assemble this deposit

Note. The architecture figures in the paper (Figures 1 and 3) draw the head
sections schematically. The three source cohorts (RSNA, CQ500, BHSD) are
distributed under non-commercial licences incompatible with the article's open
licence, so no cohort image is reproduced and no image-rasterising script is
included.

## Protocol

Fixed 10-epoch schedule, final-epoch model reported, no metric-based checkpoint
selection. All threshold-dependent metrics at a fixed 0.5 threshold. Metrics
pooled out-of-fold. The held-out test set is absent from `folds.parquet`, so
training code selecting `fold != k` excludes it structurally.

## Requirements

See `requirements.txt`. Key versions:

    python 3.11
    torch 2.6.0+cu124        # primary pipeline
    torch 2.13.0+cu130       # base-method re-implementation (10_wsgsa_baseline.py)
    timm 1.0.28, numpy 2.4.4, pandas 3.0.3, scikit-learn 1.9.0,
    nibabel 5.4.2, opencv 5.0.0

The two PyTorch versions are the only environment difference between the primary
pipeline and the base-method re-implementation; all other package versions are
identical. Hardware used: NVIDIA RTX 4070 Ti (12 GB) and RTX A4000 (16 GB).

## Setup

Set environment variables pointing to your local dataset locations before
running any script:

```bash
export ICH_ROOT=/path/to/rsna-intracranial-hemorrhage-detection
export ICH_CQ=/path/to/CQ500
export ICH_BHSD=/path/to/BHSD/archive
export ICH_STORE_RES=384
export ICH_TRAIN_RES=384
```

## Usage

```bash
python 01b_prepare_split.py            # verify the printed hash is f72016749123e8ce
python 00_preprocess_to_memmap.py      # one-off, ~333 GB
python 02_train_stage1.py --fold 0 --backbone convnext_tiny --loss logit_adjusted
python 03a_extract_features.py --run stage1_fold0_logit_adjusted
python 03c_conditional_subtype.py --features stage1_fold0_logit_adjusted --fold 0 --method conditional
python 08_paper_figures.py \
    --runs 'stage2_cond_fold{F}_conditional' --folds 0 1 2 3 4 \
    --compare 'stage2_cond_fold{F}_joint'
```

The `{F}` placeholder is substituted per fold and must be present, or the same
fold is pooled repeatedly.

## Data

RSNA 2019, CQ500, and BHSD are obtained from their original distributors under
their own terms and are not redistributed here.

- RSNA: <https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection>
  (dataset description <https://doi.org/10.1148/ryai.2020190211>)
- CQ500: <http://headctstudy.qure.ai/dataset>
- BHSD: <https://github.com/White65534/BHSD>

## Citation

```
Ayub H, Mujtaba H. Rare-Subtype Performance in Intracranial Hemorrhage
Classification Is Protocol-Dependent: Evidence for a Data-Level Precision
Ceiling.

Code and partition assignments: https://doi.org/10.5281/zenodo.21891827
Trained model weights:          https://doi.org/10.5281/zenodo.21892318
```

## License

Code is released under the MIT License (see `LICENSE`). The imaging cohorts are
not redistributed and remain under their distributors' terms.
