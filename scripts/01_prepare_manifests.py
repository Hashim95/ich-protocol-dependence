#!/usr/bin/env python3
"""
01_prepare_manifests.py  —  ICH pipeline, step 1 of N

Builds everything the training scripts need, ONCE, and caches to disk:
  1. slice_manifest.parquet   : one row per CT slice -> dicom path + 6 labels
  2. study_manifest.parquet   : one row per study    -> study-level labels (max over slices)
  3. folds.parquet            : patient-grouped 5-fold assignment (for pooled OOF eval)

Design decisions baked in (discussed in planning):
  * pivot_table(aggfunc='max') to survive duplicate (image_id, subtype) rows in the CSV.
  * one-time parallel DICOM header scan for StudyInstanceUID + PatientID (cached).
  * StratifiedGroupKFold on a multilabel bit-key, grouped by PatientID -> no patient leakage.
  * epidural is ~0.42% -> per-fold EDH counts are tiny, so downstream eval is POOLED OOF,
    not per-fold averaged. This script just records the fold id; the eval script pools.

Run once on the laptop (CPU only). Safe to re-run: it skips work already cached.
"""

import os, sys, json, hashlib, argparse, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------- #
# CONFIG  — edit ROOT if your data lives elsewhere; everything else is derived.
# ----------------------------------------------------------------------------- #
ROOT = Path(os.environ.get("ICH_ROOT", "/path/to/rsna-intracranial-hemorrhage-detection"))
CSV       = ROOT / "stage_2_train.csv"
DCM_DIR   = ROOT / "stage_2_train"
OUT_DIR   = ROOT / "manifests"          # all cached outputs go here
N_FOLDS   = 5
SEED      = 42
SUBTYPES  = ["epidural", "intraparenchymal", "intraventricular",
             "subarachnoid", "subdural"]          # 5 subtypes (order matters, kept everywhere)
ALL_COLS  = SUBTYPES + ["any"]                     # 6 labels total; 'any' handled separately
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- #
# STEP 1  —  slice manifest from the CSV (pivot long -> wide, dedupe-safe)
# ----------------------------------------------------------------------------- #
def build_slice_labels() -> pd.DataFrame:
    cache = OUT_DIR / "slice_labels.parquet"
    if cache.exists():
        print(f"[1/4] slice labels: cached -> {cache.name}")
        return pd.read_parquet(cache)

    print(f"[1/4] slice labels: parsing {CSV.name} ...")
    df = pd.read_csv(CSV)
    # ID looks like  ID_<hash>_<subtype>
    parts = df["ID"].str.split("_", expand=True)
    df["image_id"] = parts[0] + "_" + parts[1]
    df["subtype"]  = parts[2]
    df["Label"]    = df["Label"].astype("int8")

    # aggfunc='max' collapses the handful of duplicate rows without error
    wide = (df.pivot_table(index="image_id", columns="subtype",
                           values="Label", aggfunc="max")
              .reindex(columns=ALL_COLS)      # enforce column order
              .fillna(0).astype("int8")
              .reset_index())

    # sanity: 'any' should equal OR of the 5 subtypes for the vast majority of rows.
    or5 = (wide[SUBTYPES].sum(axis=1) > 0).astype("int8")
    mism = int((or5 != wide["any"]).sum())
    print(f"      unique slices: {len(wide):,} | 'any' vs OR(subtypes) mismatches: {mism:,} "
          f"({100*mism/len(wide):.2f}%  — small mismatch is normal in RSNA)")

    wide.to_parquet(cache, index=False)
    print(f"      wrote {cache.name}")
    return wide


# ----------------------------------------------------------------------------- #
# STEP 2  —  one-time DICOM header scan for study/patient grouping
# ----------------------------------------------------------------------------- #
def _read_one_header(args):
    """Worker: read only the tags we need, no pixels."""
    image_id, path = args
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True,
                             specific_tags=["StudyInstanceUID", "PatientID",
                                            "SOPInstanceUID", "ImagePositionPatient",
                                            "InstanceNumber"])
        study = str(getattr(ds, "StudyInstanceUID", "")) or "UNK_STUDY"
        patient = str(getattr(ds, "PatientID", "")) or study   # fall back to study
        # z position for later slice ordering (Stage 2 needs sequence order)
        z = 0.0
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            z = float(ipp[2])
        elif getattr(ds, "InstanceNumber", None) is not None:
            z = float(ds.InstanceNumber)
        return image_id, study, patient, z, True
    except Exception:
        return image_id, "UNK_STUDY", "UNK_PATIENT", 0.0, False


def build_header_index(slice_df: pd.DataFrame) -> pd.DataFrame:
    cache = OUT_DIR / "header_index.parquet"
    if cache.exists():
        print(f"[2/4] header scan: cached -> {cache.name}")
        return pd.read_parquet(cache)

    print(f"[2/4] header scan: reading {len(slice_df):,} DICOM headers "
          f"with {N_WORKERS} workers (one-time, ~10-15 min)...")

    tasks = [(iid, str(DCM_DIR / f"{iid}.dcm")) for iid in slice_df["image_id"]]
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_read_one_header, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="      headers"):
            results.append(fut.result())

    hdr = pd.DataFrame(results, columns=["image_id", "study", "patient", "z", "ok"])
    n_bad = int((~hdr["ok"]).sum())
    if n_bad:
        print(f"      WARNING: {n_bad:,} headers failed to read (kept, grouped as UNK).")
    print(f"      studies: {hdr['study'].nunique():,} | patients: {hdr['patient'].nunique():,}")
    hdr.to_parquet(cache, index=False)
    print(f"      wrote {cache.name}")
    return hdr


# ----------------------------------------------------------------------------- #
# STEP 3  —  merge, build slice + study manifests
# ----------------------------------------------------------------------------- #
def build_manifests(slice_df, hdr):
    slice_cache = OUT_DIR / "slice_manifest.parquet"
    study_cache = OUT_DIR / "study_manifest.parquet"
    if slice_cache.exists() and study_cache.exists():
        print(f"[3/4] manifests: cached")
        return pd.read_parquet(slice_cache), pd.read_parquet(study_cache)

    print(f"[3/4] manifests: merging labels + headers ...")
    m = slice_df.merge(hdr[["image_id", "study", "patient", "z"]], on="image_id", how="left")
    m["path"] = m["image_id"].apply(lambda i: str(DCM_DIR / f"{i}.dcm"))

    # slice manifest
    slice_cols = ["image_id", "path", "study", "patient", "z"] + ALL_COLS
    slice_manifest = m[slice_cols].copy()
    slice_manifest.to_parquet(slice_cache, index=False)

    # study manifest: label = max over slices (study is positive if ANY slice is)
    agg = {c: "max" for c in ALL_COLS}
    study_manifest = (m.groupby(["study", "patient"], as_index=False)
                        .agg({**agg, "image_id": "count"})
                        .rename(columns={"image_id": "n_slices"}))
    study_manifest.to_parquet(study_cache, index=False)

    print(f"      slices: {len(slice_manifest):,} | studies: {len(study_manifest):,}")
    print(f"      study-level prevalence:")
    for c in ALL_COLS:
        p = study_manifest[c].mean()
        print(f"        {c:18s} {p:.4f}  ({int(study_manifest[c].sum()):>6,} studies)")
    return slice_manifest, study_manifest


# ----------------------------------------------------------------------------- #
# STEP 4  —  patient-grouped stratified 5-fold (recorded at study level)
# ----------------------------------------------------------------------------- #
def build_folds(study_manifest):
    cache = OUT_DIR / "folds.parquet"
    if cache.exists():
        print(f"[4/4] folds: cached -> {cache.name}")
        return pd.read_parquet(cache)

    print(f"[4/4] folds: StratifiedGroupKFold(n_splits={N_FOLDS}) grouped by patient ...")
    sm = study_manifest.reset_index(drop=True).copy()
    groups = sm["patient"].values

    def _try_split(strat_key):
        """Return fold assignment or None if a stratum is too small for N_FOLDS."""
        counts = pd.Series(strat_key).value_counts()
        if counts.min() < N_FOLDS:
            return None
        folds = np.full(len(sm), -1, dtype=int)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for f, (_, val_idx) in enumerate(sgkf.split(sm, strat_key, groups=groups)):
            folds[val_idx] = f
        return folds

    # Try progressively coarser stratification keys until one is feasible.
    # 1) full 5-bit multilabel key  2) (any, epidural) key  3) 'any' only
    full_key = (sm[SUBTYPES].values * np.array([16, 8, 4, 2, 1])).sum(axis=1).astype(int)
    edh_any_key = sm["epidural"].values * 2 + sm["any"].values
    fold_assign = None
    for name, key in [("full 5-bit multilabel", full_key),
                      ("(epidural, any)", edh_any_key),
                      ("'any' only", sm["any"].values.astype(int))]:
        fold_assign = _try_split(key)
        if fold_assign is not None:
            print(f"      stratified on: {name}")
            break
    if fold_assign is None:
        # last resort: plain grouped split, no stratification
        from sklearn.model_selection import GroupKFold
        print("      stratification infeasible; falling back to plain GroupKFold")
        fold_assign = np.full(len(sm), -1, dtype=int)
        for f, (_, val_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(sm, groups=groups)):
            fold_assign[val_idx] = f

    sm["fold"] = fold_assign

    # report EDH per fold (this is the number that motivates pooled-OOF eval)
    print(f"      epidural studies per validation fold:")
    for f in range(N_FOLDS):
        v = sm[sm.fold == f]
        print(f"        fold {f}: val_studies={len(v):>5,}  EDH={int(v['epidural'].sum()):>3}  "
              f"any={int(v['any'].sum()):>5}")
    # patient-disjointness assertion
    for f in range(N_FOLDS):
        tr_pat = set(sm[sm.fold != f]["patient"])
        va_pat = set(sm[sm.fold == f]["patient"])
        assert len(tr_pat & va_pat) == 0, f"patient leak in fold {f}!"
    print("      patient-disjoint across all folds: OK")

    sm[["study", "patient", "fold"]].to_parquet(cache, index=False)
    print(f"      wrote {cache.name}")
    return sm


# ----------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("ICH pipeline — step 1: manifest preparation")
    print(f"ROOT: {ROOT}")
    print(f"OUT : {OUT_DIR}")
    print("=" * 70)

    assert CSV.exists(),     f"CSV not found: {CSV}"
    assert DCM_DIR.exists(), f"DICOM dir not found: {DCM_DIR}"

    slice_df = build_slice_labels()
    hdr      = build_header_index(slice_df)
    slice_manifest, study_manifest = build_manifests(slice_df, hdr)
    folds    = build_folds(study_manifest)

    print("=" * 70)
    print("DONE. Cached files in", OUT_DIR)
    for f in ["slice_manifest.parquet", "study_manifest.parquet", "folds.parquet"]:
        p = OUT_DIR / f
        print(f"  {f:28s} {p.stat().st_size/1e6:7.1f} MB")
    print("Next: 02_train_stage1.py (slice-level ConvNeXt).")
    print("=" * 70)


if __name__ == "__main__":
    main()