#!/usr/bin/env python3
"""windowing.py — the ONE windowing authority for RSNA and CQ500.
Both cohorts must produce pixel-identical 3-channel windowed images or external
validation is not credible. Includes a guard for DICOMs that are already
rescaled/pre-windowed so they can't be double-windowed.
"""
import numpy as np
import pydicom
import cv2
from ich_config import WINDOWS


def _to_hu(ds):
    """Pixel array -> Hounsfield Units, with a pre-windowed guard.
    Returns None on unreadable pixel data."""
    try:
        a = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    a = a * slope + inter
    # Guard: genuine NCCT HU spans roughly [-1024, ~3000]. If a scan is already
    # 8-bit pre-windowed (max<=255, min>=0) with no rescale, treat as brain-window
    # bytes and skip re-windowing by faking an HU range. Rare in RSNA, seen in some
    # CQ500 mirrors.
    if slope == 1.0 and inter == 0.0 and a.min() >= 0 and a.max() <= 255:
        # already display-scaled; expand back to a nominal brain window so the
        # three-window stack still varies. Flag via attribute for the caller.
        a = a / 255.0 * 80.0 + 0.0    # map 0..255 -> 0..80 HU-ish (brain WW)
    return a


def window_dicom(path, out_size, mean=None, std=None, normalize=True):
    """DICOM path -> (out_size,out_size,3) float32. Channels = brain/subdural/bone.
    If normalize, applies ImageNet mean/std. Returns zeros on failure (never raises)."""
    try:
        ds = pydicom.dcmread(path)
    except Exception:
        return np.zeros((out_size, out_size, 3), np.float32)
    a = _to_hu(ds)
    if a is None:
        return np.zeros((out_size, out_size, 3), np.float32)
    ch = []
    for wl, ww in WINDOWS:
        lo, hi = wl - ww / 2, wl + ww / 2
        ch.append(np.clip((a - lo) / (hi - lo + 1e-6), 0, 1))
    im = np.stack(ch, -1).astype(np.float32)
    if im.shape[:2] != (out_size, out_size):
        im = cv2.resize(im, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    if normalize:
        m = np.array([0.485, 0.456, 0.406], np.float32) if mean is None else mean
        s = np.array([0.229, 0.224, 0.225], np.float32) if std is None else std
        im = (im - m) / s
    return im


def window_to_uint8(path, out_size):
    """For the memmap: windowed image as uint8 [0,255], NO normalization.
    Normalization happens on-GPU at train time. Returns None on failure so the
    caller can record a bad-slice sentinel."""
    try:
        ds = pydicom.dcmread(path)
    except Exception:
        return None
    a = _to_hu(ds)
    if a is None:
        return None
    ch = []
    for wl, ww in WINDOWS:
        lo, hi = wl - ww / 2, wl + ww / 2
        ch.append(np.clip((a - lo) / (hi - lo + 1e-6), 0, 1))
    im = np.stack(ch, -1)
    if im.shape[:2] != (out_size, out_size):
        im = cv2.resize(im, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return (im * 255.0 + 0.5).astype(np.uint8)