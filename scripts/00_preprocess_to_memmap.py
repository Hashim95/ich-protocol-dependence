#!/usr/bin/env python3
"""
00_preprocess_to_memmap.py — decode every RSNA slice ONCE into a uint8 memmap.

Why: pydicom decode from a SATA SSD is the training bottleneck. Decoding once to a
uint8 memmap on ext4 lets the OS cache it in your 128 GB RAM, so training reads are
RAM-speed and the A4000 stays pinned near 100%.

Also fixes the z-ordering bug: computes geometric z = IPP . slice_normal
(normal = rowcos x colcos from ImageOrientationPatient), falling back to
InstanceNumber only when geometry is missing. Stage-2 / smoothing / CQ500 all rely
on correct order.

Requires: 01_prepare_manifests.py already run (needs manifests/slice_manifest.parquet).

Output (in manifests/../memmap/):
    slices_{STORE_RES}.dat      raw uint8 memmap  [N, 3, H, W]
    memmap_meta.json            shape/dtype so trainers can reopen it
    memmap_index.parquet        row -> image_id, study, patient, z, ok, + 6 labels
"""
import os, json, argparse, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from ich_config import (ROOT, MAN_DIR, MEMMAP_DIR, ALL_COLS, STORE_RES, setup_hardware)
from windowing import window_to_uint8

warnings.filterwarnings("ignore")
setup_hardware()

N_WORKERS = max(1, (os.cpu_count() or 8) - 4)     # leave a few cores free


def geometric_z(ds):
    """z = IPP projected onto slice normal. Fallback: InstanceNumber, then 0."""
    try:
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if ipp is not None and iop is not None and len(ipp) == 3 and len(iop) == 6:
            r = np.array(iop[:3], np.float64)
            c = np.array(iop[3:], np.float64)
            n = np.cross(r, c)
            return float(np.dot(np.array(ipp, np.float64), n))
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
    except Exception:
        pass
    try:
        return float(getattr(ds, "InstanceNumber", 0) or 0)
    except Exception:
        return 0.0


def _process_chunk(args):
    """Worker: write its row-range into the memmap, return per-row (z, ok)."""
    rows, paths, dat_path, shape = args
    H = shape[2]
    mm = np.memmap(dat_path, dtype=np.uint8, mode="r+", shape=shape)
    out = []
    for row, path in zip(rows, paths):
        z, ok = 0.0, False
        try:
            ds = pydicom.dcmread(path)
            z = geometric_z(ds)
            img = window_to_uint8(path, H)        # (H,W,3) uint8 or None
            if img is not None:
                mm[row] = img.transpose(2, 0, 1)  # -> (3,H,W)
                ok = True
        except Exception:
            pass
        if not ok:
            mm[row] = 0
        out.append((int(row), z, ok))
    mm.flush()
    del mm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=2000, help="rows per worker task")
    a = ap.parse_args()

    sm = pd.read_parquet(MAN_DIR / "slice_manifest.parquet").reset_index(drop=True)
    N = len(sm)
    H = STORE_RES
    shape = (N, 3, H, H)
    dat_path = MEMMAP_DIR / f"slices_{H}.dat"

    print("=" * 70)
    print(f"Preprocess -> memmap  |  N={N:,} slices  res={H}  workers={N_WORKERS}")
    print(f"  file: {dat_path}  (~{N*3*H*H/1e9:.0f} GB)")
    print("=" * 70)

    # pre-create the memmap on disk (zeros)
    mm = np.memmap(dat_path, dtype=np.uint8, mode="w+", shape=shape)
    del mm   # workers reopen in r+ and write disjoint rows

    paths = sm["path"].tolist()
    tasks = []
    for i in range(0, N, a.chunk):
        rows = list(range(i, min(i + a.chunk, N)))
        tasks.append((rows, [paths[r] for r in rows], str(dat_path), shape))

    z = np.zeros(N, np.float64); ok = np.zeros(N, bool)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_process_chunk, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="decode"):
            for row, zz, okk in fut.result():
                z[row] = zz; ok[row] = okk

    n_bad = int((~ok).sum())
    print(f"  decoded ok: {int(ok.sum()):,} | failed(zeros): {n_bad:,}")

    idx = sm.copy()
    idx["row"] = np.arange(N)
    idx["z"] = z            # corrected geometric z (overrides 01's z)
    idx["ok"] = ok
    keep = ["row", "image_id", "study", "patient", "z", "ok"] + ALL_COLS
    idx[keep].to_parquet(MEMMAP_DIR / "memmap_index.parquet", index=False)

    json.dump({"n": N, "c": 3, "h": H, "w": H, "dtype": "uint8",
               "store_res": H, "dat": dat_path.name},
              open(MEMMAP_DIR / "memmap_meta.json", "w"), indent=2)

    print(f"  wrote memmap_index.parquet + memmap_meta.json")
    print("DONE. Next: 02_train_stage1.py"); print("=" * 70)


if __name__ == "__main__":
    main()