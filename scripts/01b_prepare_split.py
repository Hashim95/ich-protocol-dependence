#!/usr/bin/env python3
"""
01b_prepare_split.py — carve a held-out test set, then rebuild the CV folds.

WHY THIS EXISTS
---------------
The original 01_prepare_manifests.py produced a 5-fold patient-disjoint CV over
all 21,744 studies with no held-out test set. That is fine for a pilot but has
two consequences that block Paper 1 and Paper 2:

  1. Every threshold, every model-selection choice, and every reported metric
     came from the same pooled out-of-fold predictions. Even with non-leaky
     split-half threshold selection, the *pipeline* was tuned on all the data.
     A top-tier reviewer will ask for a set that was touched exactly once.

  2. Five fold-models cannot be ensembled. Each has seen 4/5 of the data the
     others were validated on, so their disagreement is not predictive
     uncertainty. Paper 2's deep ensemble is impossible without a common
     unseen set.

This script fixes both, using the CACHED study_manifest.parquet. It does NOT
re-scan DICOM headers (~752k files) — steps 1-3 of the original script stay as
they are.

WHAT IT WRITES
--------------
  folds.parquet         DEVELOPMENT STUDIES ONLY, columns [study, patient, fold]
                        fold in 0..4. Drop-in replacement: downstream code that
                        does `sm[sm.fold != f]` for training now automatically
                        excludes the test set, because test studies are simply
                        absent from this file.

  heldout_test.parquet  Test studies, columns [study, patient]. NOTHING in the
                        training path should ever read this file. It is consumed
                        once, at the end of the project, by the final evaluation.

  split_summary.csv     Per-partition counts for all six labels. Goes straight
                        into the paper as the data table.

  split_meta.json       Seed, sizes, and a hash of the assignment, so the split
                        is reproducible for the Zenodo release.

DESIGN DECISIONS
----------------
  * Test fraction is 1/5 (20%), obtained as fold 0 of a StratifiedGroupKFold.
    Reusing the same machinery as the CV keeps stratification behaviour
    identical between the two levels.

  * Stratification falls back down a ladder: full multilabel bit-key ->
    (EDH, any) -> EDH only. With 354 EDH studies split six ways, over-fine
    strata produce partitions too small for the rarest class. EDH is what the
    entire paper rests on, so it is what the split protects.

  * Grouping is by PatientID at BOTH levels, so no patient appears in more than
    one partition anywhere.

  * Assertions abort rather than warn. Silent leakage invalidates a whole
    paper; a crash costs two minutes.

USAGE
-----
    python 01b_prepare_split.py                 # writes the new split
    python 01b_prepare_split.py --dry-run       # report only, write nothing
    python 01b_prepare_split.py --test-frac 0.2 --seed 42
"""
import argparse, json, hashlib, shutil, sys, datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------------- #
ROOT     = Path(os.environ["ICH_ROOT"])
OUT_DIR  = ROOT / "manifests"
SUBTYPES = ["epidural", "intraparenchymal", "intraventricular",
            "subarachnoid", "subdural"]
ALL_COLS = SUBTYPES + ["any"]
N_FOLDS  = 5
SEED     = 42
# ----------------------------------------------------------------------------- #


def _stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def strat_key(sm: pd.DataFrame, level: str) -> np.ndarray:
    """Stratification key. `level` picks how fine-grained it is."""
    if level == "bitkey":
        k = np.zeros(len(sm), dtype=np.int64)
        for b, c in enumerate(ALL_COLS):
            k += (sm[c].values.astype(np.int64) > 0) << b
        return k
    if level == "edh_any":
        return (sm["epidural"].values > 0).astype(int) * 2 + \
               (sm["any"].values > 0).astype(int)
    return (sm["epidural"].values > 0).astype(int)


def _viable(key: np.ndarray, groups: np.ndarray, n_splits: int) -> bool:
    """A stratum must have at least n_splits members AND enough distinct
    patients, or StratifiedGroupKFold cannot place it in every fold."""
    for v in np.unique(key):
        m = key == v
        if m.sum() < n_splits:
            return False
        if len(np.unique(groups[m])) < n_splits:
            return False
    return True


def split_once(sm: pd.DataFrame, n_splits: int, seed: int, what: str):
    """StratifiedGroupKFold with a fallback ladder on the stratification key.
    Returns (fold_assignment array, level_used)."""
    groups = sm["patient"].values
    for level in ("bitkey", "edh_any", "edh"):
        key = strat_key(sm, level)
        if not _viable(key, groups, n_splits):
            print(f"      [{what}] stratification '{level}' too fine; falling back")
            continue
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        assign = np.full(len(sm), -1, dtype=int)
        for f, (_, val_idx) in enumerate(sgkf.split(sm, key, groups)):
            assign[val_idx] = f
        if (assign < 0).any():
            print(f"      [{what}] '{level}' left {int((assign<0).sum())} unassigned; falling back")
            continue
        print(f"      [{what}] stratified on '{level}'")
        return assign, level
    raise SystemExit(f"[{what}] could not stratify at any level — inspect the manifest")


def label_counts(sm: pd.DataFrame) -> dict:
    d = {"studies": len(sm), "patients": int(sm["patient"].nunique())}
    for c in ALL_COLS:
        n = int((sm[c] > 0).sum())
        d[c] = n
        d[f"{c}_pct"] = round(100.0 * n / max(len(sm), 1), 3)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.20,
                    help="held-out fraction; 0.20 uses fold 0 of a 5-way split")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sm_path = OUT_DIR / "study_manifest.parquet"
    if not sm_path.exists():
        raise SystemExit(f"missing {sm_path} — run 01_prepare_manifests.py first")

    sm = pd.read_parquet(sm_path).reset_index(drop=True)
    print(f"loaded study manifest: {len(sm):,} studies, "
          f"{sm['patient'].nunique():,} patients\n")

    # ---------------- level 1: held-out test -------------------------------- #
    n_outer = int(round(1.0 / a.test_frac))
    print(f"[1/3] carving held-out test set (1/{n_outer} of studies)")
    outer, lvl_outer = split_once(sm, n_outer, a.seed, "test-split")
    is_test = outer == 0
    test = sm[is_test].copy().reset_index(drop=True)
    dev  = sm[~is_test].copy().reset_index(drop=True)
    print(f"      test: {len(test):,} studies / {test['patient'].nunique():,} patients")
    print(f"      dev : {len(dev):,} studies / {dev['patient'].nunique():,} patients\n")

    # ---------------- level 2: CV folds within development ------------------ #
    print(f"[2/3] {a.folds}-fold CV within development")
    inner, lvl_inner = split_once(dev, a.folds, a.seed, "cv-split")
    dev["fold"] = inner

    # ---------------- assertions -------------------------------------------- #
    print("\n[3/3] verifying disjointness")
    pat_test, pat_dev = set(test["patient"]), set(dev["patient"])
    overlap = pat_test & pat_dev
    assert not overlap, f"PATIENT LEAK: {len(overlap)} patients in both test and dev"
    print(f"      test <-> dev patient-disjoint: OK")

    for f in range(a.folds):
        tr = set(dev[dev.fold != f]["patient"])
        va = set(dev[dev.fold == f]["patient"])
        assert not (tr & va), f"PATIENT LEAK inside fold {f}"
        assert not (va & pat_test), f"PATIENT LEAK: fold {f} val overlaps test"
    print(f"      all {a.folds} folds patient-disjoint, none touch test: OK")

    assert len(test) + len(dev) == len(sm), "studies lost during split"
    assert set(test["study"]).isdisjoint(set(dev["study"])), "study id in both partitions"
    print(f"      no studies lost, no study in two partitions: OK")

    edh_test = int((test["epidural"] > 0).sum())
    edh_dev  = int((dev["epidural"] > 0).sum())
    print(f"\n      EDH studies -> test {edh_test}, dev {edh_dev}, "
          f"total {edh_test + edh_dev}")
    if edh_test < 40:
        print(f"      [WARNING] only {edh_test} EDH in test — external CIs will be wide")

    # ---------------- summary table ----------------------------------------- #
    rows = [{"partition": "TEST", **label_counts(test)},
            {"partition": "DEV_ALL", **label_counts(dev)}]
    for f in range(a.folds):
        rows.append({"partition": f"dev_fold{f}", **label_counts(dev[dev.fold == f])})
    summary = pd.DataFrame(rows)

    print("\nsplit summary")
    print(summary[["partition", "studies", "patients", "epidural", "epidural_pct",
                   "any", "subdural"]].to_string(index=False))

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    # ---------------- write -------------------------------------------------- #
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = OUT_DIR / "folds.parquet"
    if old.exists():
        bak = OUT_DIR / f"folds_PILOT_backup_{_stamp()}.parquet"
        shutil.copy2(old, bak)
        print(f"\n[backup] previous folds -> {bak.name}")

    dev[["study", "patient", "fold"]].to_parquet(old, index=False)
    test[["study", "patient"]].to_parquet(OUT_DIR / "heldout_test.parquet", index=False)
    summary.to_csv(OUT_DIR / "split_summary.csv", index=False)

    h = hashlib.sha256(
        ("|".join(sorted(test["study"].astype(str))) + "#" +
         "|".join(f"{s}:{f}" for s, f in
                  sorted(zip(dev["study"].astype(str), dev["fold"])))
         ).encode()).hexdigest()[:16]

    meta = dict(created=_stamp(), seed=a.seed, n_folds=a.folds,
                test_frac=a.test_frac, outer_strat=lvl_outer, inner_strat=lvl_inner,
                n_studies_total=len(sm), n_test=len(test), n_dev=len(dev),
                edh_test=edh_test, edh_dev=edh_dev, assignment_sha256_16=h)
    (OUT_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[written] folds.parquet          ({len(dev):,} DEV studies, folds 0-{a.folds-1})")
    print(f"[written] heldout_test.parquet   ({len(test):,} TEST studies)")
    print(f"[written] split_summary.csv, split_meta.json")
    print(f"[hash] {h}   <- record this in the paper for reproducibility")
    print("\nNOTE: folds.parquet now contains DEVELOPMENT STUDIES ONLY.")
    print("      Training code that selects `fold != f` therefore excludes the")
    print("      test set automatically. Do not add test rows back into it.")


if __name__ == "__main__":
    main()