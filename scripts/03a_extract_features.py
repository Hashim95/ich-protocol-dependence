#!/usr/bin/env python3
"""
03a_extract_features.py — run a trained Stage-1 backbone over the memmap once,
cache per-slice features for the Stage-2 heads (03b/03c). Frozen backbone =
identical features across every head ablation.

Handles checkpoints saved by torch.compile (keys prefixed '_orig_mod.').

    python 03a_extract_features.py --stage1_run stage1_fold1_logit_adjusted_workstation
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import timm
from tqdm import tqdm

from ich_config import (MAN_DIR, MEMMAP_DIR, FEAT_DIR, S1_DIR, ALL_COLS,
                        TRAIN_RES, AMP_DTYPE, DEVICE, PROFILE, setup_hardware)
warnings.filterwarnings("ignore")
setup_hardware()

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BATCH = 256 if PROFILE == "workstation" else 64


class MemmapDS(Dataset):
    def __init__(self, df, meta):
        self.df = df.reset_index(drop=True); self.meta = meta
        self.shape = (meta["n"], meta["c"], meta["h"], meta["w"])
        self.dat = str(MEMMAP_DIR / meta["dat"]); self.mm = None
    def _ensure(self):
        if self.mm is None:
            self.mm = np.memmap(self.dat, dtype=np.uint8, mode="r", shape=self.shape)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        self._ensure()
        r = self.df.iloc[i]
        return torch.from_numpy(np.asarray(self.mm[int(r["row"])])), int(i)


def gpu_prep(x_uint8):
    x = x_uint8.to(DEVICE, non_blocking=True).float().div_(255.0)
    if x.shape[-1] != TRAIN_RES:
        x = F.interpolate(x, size=(TRAIN_RES, TRAIN_RES), mode="bilinear", align_corners=False)
    x = (x - _MEAN.to(DEVICE)) / _STD.to(DEVICE)
    return x.contiguous(memory_format=torch.channels_last)


def strip_and_match(sd):
    """Strip torch.compile '_orig_mod.' then find the backbone prefix (net./enc.)."""
    sd = { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
           for k, v in sd.items() }
    prefix = None
    if any(k.startswith("net.") for k in sd):   prefix = "net."
    elif any(k.startswith("enc.") for k in sd): prefix = "enc."
    assert prefix is not None, f"no net./enc. backbone prefix in keys: {list(sd)[:6]}"
    enc_sd = {}
    for k, v in sd.items():
        if not k.startswith(prefix): continue
        nk = k[len(prefix):]
        # drop the 6-way classifier head (feature extractor is num_classes=0)
        if ("head.fc" in nk or nk.startswith("fc.")) and v.ndim <= 2 and v.shape[0] == 6:
            continue
        enc_sd[nk] = v
    return enc_sd, prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_run", required=True)
    ap.add_argument("--shard_size", type=int, default=50000)
    args = ap.parse_args()

    ckpt_path = S1_DIR / args.stage1_run / "best.pt"
    assert ckpt_path.exists(), f"not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = ckpt["cfg"]["backbone"]
    print(f"loading {backbone} from {args.stage1_run}")

    enc = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
    enc_sd, prefix = strip_and_match(ckpt["model"])
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    print(f"  prefix='{prefix}' feat_dim={enc.num_features} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 5:
        print(f"  WARNING backbone may be mis-loaded; first missing: {missing[:5]}")
    enc = enc.to(DEVICE, memory_format=torch.channels_last).eval()
    for p in enc.parameters(): p.requires_grad = False

    meta = json.load(open(MEMMAP_DIR / "memmap_meta.json"))
    idx  = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    idx  = idx[idx["ok"]].reset_index(drop=True)
    print(f"  extracting {len(idx):,} slices")

    ld = DataLoader(MemmapDS(idx, meta), batch_size=BATCH, shuffle=False,
                    num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=6)

    outdir = FEAT_DIR / args.stage1_run; outdir.mkdir(parents=True, exist_ok=True)
    feats, order, shard = [], [], 0
    def flush(feats, order, shard):
        if not feats: return shard
        Fm = np.concatenate(feats).astype(np.float16)
        sub = idx.iloc[np.array(order)]
        np.savez(outdir / f"shard_{shard:03d}.npz",
                 feat=Fm, labels=sub[ALL_COLS].values.astype(np.int8),
                 study=sub["study"].values.astype(str),
                 z=sub["z"].values.astype(np.float32),
                 image_id=sub["image_id"].values.astype(str))
        return shard + 1

    with torch.no_grad():
        for xb, ii in tqdm(ld, desc="extract"):
            x = gpu_prep(xb)
            with autocast("cuda", dtype=AMP_DTYPE):
                f = enc(x)
            feats.append(f.float().cpu().numpy()); order.extend(ii.numpy().tolist())
            if sum(len(a) for a in feats) >= args.shard_size:
                shard = flush(feats, order, shard); feats, order = [], []
    shard = flush(feats, order, shard)
    (outdir / "meta.txt").write_text(
        f"backbone={backbone}\nfeat_dim={enc.num_features}\nprofile={PROFILE}\n")
    print(f"DONE: {shard} shards -> {outdir}")


if __name__ == "__main__":
    main()