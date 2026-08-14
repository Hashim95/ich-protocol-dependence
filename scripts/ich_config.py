#!/usr/bin/env python3
"""ich_config.py — single source of truth for paths, hardware, and resolution.
Import this at the top of every script instead of hardcoding ROOT/PROFILE.

    from ich_config import (ROOT, CQ_ROOT, BHSD_ROOT, MAN_DIR, MEMMAP_DIR,
                            SUBTYPES, ALL_COLS, WINDOWS, TRAIN_RES, STORE_RES,
                            AMP_DTYPE, DEVICE, setup_hardware)
    setup_hardware()
"""
import os
from pathlib import Path
import torch

# ----------------------------------------------------------------- paths
# Override per-machine with: export ICH_ROOT=/path/to/rsna
ROOT      = Path(os.environ.get("ICH_ROOT",   "/path/to/rsna-intracranial-hemorrhage-detection"))
CQ_ROOT   = Path(os.environ.get("ICH_CQ",     "/path/to/CQ500"))
BHSD_ROOT = Path(os.environ.get("ICH_BHSD",   "/path/to/BHSD/archive"))

MAN_DIR    = ROOT / "manifests"
MEMMAP_DIR = ROOT / "memmap"          # 00_preprocess writes here
FEAT_DIR   = ROOT / "features"
S1_DIR     = ROOT / "stage1_runs"
S2_DIR     = ROOT / "stage2_runs"
for d in (MAN_DIR, MEMMAP_DIR, FEAT_DIR, S1_DIR, S2_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- labels (order fixed everywhere)
SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
ALL_COLS = SUBTYPES + ["any"]
EDH_I, SDH_I, ANY_I = 0, 4, 5
WINDOWS  = [(40, 80), (80, 200), (600, 2800)]   # brain / subdural / bone (WL, WW)

# ----------------------------------------------------------------- resolution
# Store at 384 (fits RAM cache), train at up to 512 (GPU upsamples). One knob.
STORE_RES = int(os.environ.get("ICH_STORE_RES", 384))   # on-disk memmap size
TRAIN_RES = int(os.environ.get("ICH_TRAIN_RES", 384))   # model input size
# OOM fallback ladder for TRAIN_RES: 512 -> 448 -> 384 (just change ICH_TRAIN_RES)

# ----------------------------------------------------------------- hardware
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16       # Ampere: bf16, NO GradScaler anywhere
SEED      = 42

def setup_hardware(threads: int = 8):
    """Ampere throughput + leave CPU for DataLoader workers. Call once per script."""
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ----------------------------------------------------------------- profile (A4000 auto)
def gpu_profile():
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    return "workstation" if vram > 12 else "laptop"

PROFILE = gpu_profile()