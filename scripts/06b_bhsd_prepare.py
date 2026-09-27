#!/usr/bin/env python3
"""
06b_bhsd_prepare.py — BHSD as a SECOND independent external cohort.

WHY
---
CQ500 gives only 12 EDH positives, so its per-class EDH interval is nearly
uninformative ([0.613, 0.978]). BHSD adds 23 EDH volumes / 181 EDH slices from a
different acquisition population, roughly tripling the external EDH evidence.
"Validated on two independent cohorts" is also materially stronger to a reviewer
than one, and neither cohort contributed to development.

WHAT THE DATA IS (verified, not assumed)
----------------------------------------
  192 volumes, 6,404 slices, 512x512xN, in-plane 0.488 mm, slice ~5.3 mm.
  Files are gzipped NIfTI with NO extension, so nibabel cannot infer the format
  from the filename; they are read through an explicit gzip -> BytesIO path.
  Intensities are genuine Hounsfield units (observed range -3024 to 3071), so
  the shared windowing transfers unchanged -- this is what makes the cohort a
  credible external test rather than a preprocessing comparison.

  Mask label mapping, VERIFIED by counting rather than assumed:
      1 = EDH  ( 23 volumes,  181 slices)
      2 = IPH  (127 volumes,  888 slices)
      3 = IVH  (104 volumes,  713 slices)
      4 = SAH  (109 volumes,  976 slices)
      5 = SDH  ( 70 volumes,  765 slices)
  These counts match the published BHSD description. Scoring EDH predictions
  against IPH labels would yield plausible-looking but meaningless numbers, so
  the mapping is asserted at run time.

ORIENTATION CAVEAT -- READ THIS
-------------------------------
NIfTI arrays are not guaranteed to share DICOM's row/column convention. A
volume loaded raw may be transposed or flipped relative to the axial slices the
model was trained on. That would silently depress external performance and look
like a generalisation failure. `--phase check` writes sample PNGs so the
orientation can be confirmed visually BEFORE any numbers are believed. Do not
skip it.

USAGE
-----
    python 06b_bhsd_prepare.py --phase check      # 6 sample PNGs, look at them
    python 06b_bhsd_prepare.py --phase prepare    # decode -> memmap + manifest
    python 06b_bhsd_prepare.py --phase evaluate   # 5-fold ensemble, slice + volume
"""
import argparse, gzip, io, json, warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import nibabel as nib
import cv2
import torch
import torch.nn.functional as F
import timm
from sklearn.metrics import roc_auc_score, average_precision_score

from ich_config import (ROOT, S1_DIR, ALL_COLS, SUBTYPES, WINDOWS,
                        STORE_RES, TRAIN_RES, DEVICE, AMP_DTYPE, setup_hardware)

setup_hardware()

BHSD = Path(__import__("os").environ.get(
    "ICH_BHSD", os.environ.get("ICH_BHSD", "")))
OUT = ROOT / "bhsd"; OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)

# Verified by counting, not assumed. Asserted below.
LABEL_MAP = {1: "epidural", 2: "intraparenchymal", 3: "intraventricular",
             4: "subarachnoid", 5: "subdural"}
EXPECTED_VOL = {1: 23, 2: 127, 3: 104, 4: 109, 5: 70}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_nii(path):
    """Read a gzipped NIfTI lacking the .nii.gz extension."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    fh = nib.FileHolder(fileobj=io.BytesIO(raw))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def window_hu(a_hu, out_size=STORE_RES):
    """HU slice -> 3-channel uint8, using the SAME windows as RSNA and CQ500.

    ORIENTATION: BHSD NIfTI volumes report axcodes ('L','A','S'), so array axis 1
    runs posterior->anterior and row 0 is POSTERIOR. RSNA DICOM slices are stored
    anterior-first. Verified consistent across the cohort via nib.aff2axcodes.
    Without this flip the model would see vertically mirrored anatomy; nothing in
    the training augmentation covers a vertical flip (hflip only), so the error
    would depress external performance and be indistinguishable from a genuine
    generalisation failure.
    """
    a_hu = np.flipud(a_hu)
    ch = []
    for wl, ww in WINDOWS:
        lo, hi = wl - ww / 2, wl + ww / 2
        ch.append(np.clip((a_hu - lo) / (hi - lo + 1e-6), 0, 1))
    im = np.stack(ch, -1)
    if im.shape[:2] != (out_size, out_size):
        im = cv2.resize(im, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return (im * 255.0 + 0.5).astype(np.uint8)


def pairs():
    imgs = sorted((BHSD / "images").glob("*"))
    gts = sorted((BHSD / "ground truths").glob("*"))
    assert len(imgs) == len(gts), f"{len(imgs)} images vs {len(gts)} masks"
    for i, g in zip(imgs, gts):
        assert i.name == g.name, f"name mismatch: {i.name} vs {g.name}"
    return list(zip(imgs, gts))


# ============================================================ PHASE: check
def phase_check():
    """Write sample PNGs so orientation can be confirmed by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr = pairs()
    picks, seen = [], set()
    for ip, gp in pr:                      # one volume per subtype where possible
        a = np.asanyarray(load_nii(gp).dataobj)
        for L in range(1, 6):
            if L not in seen and (a == L).any():
                seen.add(L); picks.append((ip, gp, L)); break
        if len(picks) >= 5:
            break
    picks.append((pr[0][0], pr[0][1], 0))

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, (ip, gp, L) in zip(axes.ravel(), picks):
        vol = np.asanyarray(load_nii(ip).dataobj).astype(np.float32)
        msk = np.asanyarray(load_nii(gp).dataobj)
        k = (int(np.argmax((msk == L).sum(axis=(0, 1)))) if L
             else vol.shape[2] // 2)
        rgb = window_hu(vol[:, :, k], 384)
        ax.imshow(rgb[..., 0], cmap="gray")
        if L:
            ax.contour((cv2.resize(np.flipud(msk[:, :, k] == L).astype(np.uint8),
                                   (384, 384), interpolation=cv2.INTER_NEAREST)),
                       levels=[0.5], colors="r", linewidths=1)
        ax.set_title(f"{LABEL_MAP.get(L,'no lesion')}  slice {k}", fontsize=10)
        ax.axis("off")
    fig.suptitle("BHSD orientation check — brain window, lesion outlined.\n"
                 "Confirm axial view, anterior at top, and that lesions sit "
                 "where the subtype implies.", fontsize=11)
    fig.tight_layout()
    p = RESULTS / "bhsd_orientation_check.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] {p}")
    print("\nLook at this before trusting any BHSD number:")
    print("  * axial slices, anterior (frontal lobes) at the TOP")
    print("  * skull bright, brain mid-grey, background black")
    print("  * EDH hugging the inner skull table; IVH inside the ventricles")
    print("If the images are rotated or flipped relative to the RSNA training")
    print("data, external performance will be depressed for reasons that have")
    print("nothing to do with generalisation.")


# ============================================================ PHASE: prepare
def phase_prepare():
    pr = pairs()
    print(f"{len(pr)} volume/mask pairs")

    rows, n = [], 0
    for ip, gp in pr:
        n += load_nii(ip).shape[2]
    print(f"{n:,} slices -> memmap ({n*3*STORE_RES*STORE_RES/1e9:.1f} GB)")

    dat = OUT / f"bhsd_slices_{STORE_RES}.dat"
    mm = np.memmap(dat, dtype=np.uint8, mode="w+",
                   shape=(n, 3, STORE_RES, STORE_RES))

    vol_count = {L: 0 for L in range(1, 6)}
    r = 0
    for vi, (ip, gp) in enumerate(pr):
        vol = np.asanyarray(load_nii(ip).dataobj).astype(np.float32)
        msk = np.asanyarray(load_nii(gp).dataobj)
        assert vol.shape == msk.shape, f"{ip.name}: shape mismatch"
        for L in range(1, 6):
            if (msk == L).any():
                vol_count[L] += 1
        for k in range(vol.shape[2]):
            mm[r] = window_hu(vol[:, :, k]).transpose(2, 0, 1)
            lab = {LABEL_MAP[L]: int((msk[:, :, k] == L).any()) for L in range(1, 6)}
            lab["any"] = int(any(lab[c] for c in SUBTYPES))
            rows.append(dict(row=r, volume=ip.name, slice_idx=k,
                             z=float(k), **lab))
            r += 1
        if (vi + 1) % 40 == 0:
            print(f"  {vi+1}/{len(pr)} volumes", flush=True)
    mm.flush(); del mm

    # Guard: if the label mapping were wrong, every downstream number would be
    # plausible and wrong. Fail here instead.
    for L, exp in EXPECTED_VOL.items():
        got = vol_count[L]
        assert got == exp, (f"label {L} ({LABEL_MAP[L]}) found in {got} volumes, "
                            f"expected {exp}. LABEL_MAP may be wrong — stop and "
                            f"re-verify before evaluating.")
    print("  label mapping verified against expected volume counts: OK")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "bhsd_index.parquet", index=False)
    json.dump({"n": n, "c": 3, "h": STORE_RES, "w": STORE_RES,
               "dtype": "uint8", "dat": dat.name},
              open(OUT / "bhsd_meta.json", "w"), indent=2)

    print(f"\nslice-level prevalence")
    for c in ALL_COLS:
        print(f"  {c:<18} {int(df[c].sum()):>5} / {len(df):,} "
              f"({100*df[c].mean():.2f}%)")
    v = df.groupby("volume")[ALL_COLS].max()
    print(f"\nvolume-level prevalence")
    for c in ALL_COLS:
        print(f"  {c:<18} {int(v[c].sum()):>5} / {len(v):,} "
              f"({100*v[c].mean():.2f}%)")
    print(f"\n[written] {OUT/'bhsd_index.parquet'}, {dat.name}")


# ============================================================ PHASE: evaluate
def load_fold_models():
    models = []
    for f in range(5):
        ck = S1_DIR / f"stage1_fold{f}_logit_adjusted_workstation" / "best.pt"
        if not ck.exists():
            print(f"  missing {ck}"); continue
        d = torch.load(ck, map_location="cpu", weights_only=False)
        m = timm.create_model(d["cfg"]["backbone"], pretrained=False, num_classes=6)
        sd = {k.replace("net.", "", 1): v for k, v in d["model"].items()
              if k.startswith("net.")}
        m.load_state_dict(sd)
        models.append(m.to(DEVICE).eval().to(memory_format=torch.channels_last))
    print(f"loaded {len(models)} fold models")
    return models


def phase_evaluate(batch=64, pct=95):
    meta = json.load(open(OUT / "bhsd_meta.json"))
    df = pd.read_parquet(OUT / "bhsd_index.parquet")
    mm = np.memmap(OUT / meta["dat"], dtype=np.uint8, mode="r",
                   shape=(meta["n"], meta["c"], meta["h"], meta["w"]))
    models = load_fold_models()
    if not models:
        raise SystemExit("no fold models found")

    P = np.zeros((len(df), 6), np.float32)
    with torch.no_grad():
        for s in range(0, len(df), batch):
            x = torch.from_numpy(np.asarray(mm[s:s + batch])).to(DEVICE).float().div_(255.)
            if x.shape[-1] != TRAIN_RES:
                x = F.interpolate(x, size=(TRAIN_RES, TRAIN_RES),
                                  mode="bilinear", align_corners=False)
            x = ((x - _MEAN.to(DEVICE)) / _STD.to(DEVICE)).contiguous(
                memory_format=torch.channels_last)
            acc = 0
            for m in models:
                with torch.autocast("cuda", dtype=AMP_DTYPE):
                    acc = acc + torch.sigmoid(m(x).float())
            P[s:s + batch] = (acc / len(models)).cpu().numpy()
            if (s // batch) % 20 == 0:
                print(f"  {s:,}/{len(df):,}", flush=True)

    Y = df[ALL_COLS].values.astype(int)
    slice_tab = _metrics(P, Y, "BHSD — SLICE level, threshold 0.5")

    # volume level: percentile aggregation, matching the CQ500 protocol
    g = df.groupby("volume")
    vols, Pv, Yv = [], [], []
    for v, idx in g.groups.items():
        i = np.asarray(idx)
        vols.append(v)
        Pv.append(np.percentile(P[i], pct, axis=0))
        Yv.append(Y[i].max(0))
    Pv, Yv = np.stack(Pv), np.stack(Yv)
    vol_tab = _metrics(Pv, Yv, f"BHSD — VOLUME level ({pct}th pct), threshold 0.5")

    np.savez_compressed(OUT / "bhsd_predictions.npz", probs=P, targets=Y,
                        study=df["volume"].values.astype(str),
                        z=df["z"].values.astype(np.float32))
    slice_tab.to_csv(RESULTS / "bhsd_external_slice.csv", index=False)
    vol_tab.to_csv(RESULTS / "bhsd_external_volume.csv", index=False)
    print(f"\n[logged] bhsd_external_slice.csv, bhsd_external_volume.csv")
    print("Second independent external cohort. EDH n is small (23 volumes / "
          "181 slices) — always report the CI alongside the point estimate.")


def _metrics(P, Y, title):
    rows = []
    for i, c in enumerate(ALL_COLS):
        y, p = Y[:, i], P[:, i]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        se = tp / max(tp + fn, 1); pr = tp / max(tp + fp, 1)
        rng = np.random.default_rng(7); bs = []
        for _ in range(1000):
            s = rng.choice(len(y), len(y), replace=True)
            if 0 < y[s].sum() < len(s):
                bs.append(roc_auc_score(y[s], p[s]))
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
        rows.append(dict(cls=c, pos=int(y.sum()), AUC=float(roc_auc_score(y, p)),
                         AUC_lo=float(lo), AUC_hi=float(hi),
                         AP=float(average_precision_score(y, p)),
                         sensitivity=se, precision=pr,
                         F1=2 * pr * se / max(pr + se, 1e-12)))
    df = pd.DataFrame(rows)
    avg = df.select_dtypes("number").mean().to_dict()
    avg["cls"] = "MACRO"; avg["pos"] = int(df["pos"].sum())
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["check", "prepare", "evaluate"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pct", type=int, default=95)
    a = ap.parse_args()
    {"check": phase_check,
     "prepare": phase_prepare,
     "evaluate": lambda: phase_evaluate(a.batch, a.pct)}[a.phase]()


if __name__ == "__main__":
    main()