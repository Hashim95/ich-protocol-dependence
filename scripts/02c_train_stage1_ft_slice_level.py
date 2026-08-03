#!/usr/bin/env python3
"""
02b_train_stage1_supcon.py — Stage-1 with supervised contrastive hard-negative mining.

Representation-level contribution: alongside logit-adjusted classification, a
supervised-contrastive loss pulls EDH embeddings together and pushes them from
SDH-not-EDH slices (the specific confusion), building EDH-vs-SDH separability the
classifier alone can't create.

Corrected for the A4000/4070 pipeline:
  * config-driven paths (ich_config) — no more two-ROOT bug
  * bf16 autocast, NO GradScaler (Ampere/Ada)
  * reads the uint8 memmap, normalizes on GPU (no pydicom in hot loop)
  * contrastive loss computed in fp32 (exp() stability) — unchanged logic
  * saves val_slice_predictions.npz for 04c

    python 02b_train_stage1_supcon.py --fold 0
"""
import time, argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast
import timm
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from ich_config import (MAN_DIR, MEMMAP_DIR, S1_DIR, ALL_COLS, SUBTYPES,
                        EDH_I, SDH_I, TRAIN_RES, AMP_DTYPE, DEVICE, PROFILE,
                        SEED, setup_hardware)
warnings.filterwarnings("ignore")
setup_hardware()
np.random.seed(SEED)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CFG = {
    "workstation": dict(backbone="convnext_tiny.in12k_ft_in1k", batch=96, workers=24,
                        epochs=5, proj_dim=128, w_contrast=0.5, tau=0.1, w_hard=3.0),
    "laptop":      dict(backbone="convnext_nano.in12k_ft_in1k", batch=32, workers=8,
                        epochs=3, proj_dim=128, w_contrast=0.5, tau=0.1, w_hard=3.0),
    "cpu":         dict(backbone="convnext_nano.in12k_ft_in1k", batch=4,  workers=2,
                        epochs=1, proj_dim=128, w_contrast=0.5, tau=0.1, w_hard=3.0),
}[PROFILE]


class MemmapDS(Dataset):
    def __init__(self, df, meta, train=True):
        self.df = df.reset_index(drop=True); self.meta = meta; self.train = train
        self.shape = (meta["n"], meta["c"], meta["h"], meta["w"])
        self.dat = str(MEMMAP_DIR / meta["dat"]); self.mm = None
    def _ensure(self):
        if self.mm is None:
            self.mm = np.memmap(self.dat, dtype=np.uint8, mode="r", shape=self.shape)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        self._ensure()
        r = self.df.iloc[i]
        x = np.asarray(self.mm[int(r["row"])])
        if self.train and np.random.rand() < 0.5:
            x = x[:, :, ::-1].copy()
        y = r[ALL_COLS].values.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def gpu_prep(x_uint8):
    x = x_uint8.to(DEVICE, non_blocking=True).float().div_(255.0)
    if x.shape[-1] != TRAIN_RES:
        x = F.interpolate(x, size=(TRAIN_RES, TRAIN_RES), mode="bilinear", align_corners=False)
    x = (x - _MEAN.to(DEVICE)) / _STD.to(DEVICE)
    return x.contiguous(memory_format=torch.channels_last)


class SupConNet(nn.Module):
    def __init__(self, backbone, priors, proj_dim=128):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        d = self.enc.num_features
        self.cls = nn.Linear(d, 6)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, proj_dim))
        try: self.enc.set_grad_checkpointing(True)
        except Exception: pass
        with torch.no_grad():
            pr = torch.tensor(priors).clamp(1e-6, 1 - 1e-6)
            self.cls.bias.copy_(torch.log(pr / (1 - pr)).float())
    def forward(self, x):
        f = self.enc(x)
        return self.cls(f), F.normalize(self.proj(f), dim=1)


class LogitAdjustedBCE(nn.Module):
    def __init__(self, priors, tau=1.0):
        super().__init__()
        pr = torch.tensor(priors).clamp(1e-6, 1 - 1e-6)
        self.register_buffer("lp", torch.log(pr / (1 - pr))); self.tau = tau
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits + self.tau * self.lp.unsqueeze(0), targets)


def supcon_hardneg(z, Y, tau=0.1, w_hard=3.0):
    """EDH-focused SupCon; up-weights SDH-not-EDH hard negatives. fp32."""
    B = z.size(0)
    sim = (z @ z.t()) / tau
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    edh = Y[:, EDH_I] > 0
    if edh.sum() < 1:
        return torch.tensor(0.0, device=z.device)
    pos = (Y[:, EDH_I].unsqueeze(0) * Y[:, EDH_I].unsqueeze(1)).bool() & ~eye
    sdh_ne = (Y[:, SDH_I] > 0) & (Y[:, EDH_I] == 0)
    hard = edh.unsqueeze(1) & sdh_ne.unsqueeze(0) & ~eye
    w = torch.ones_like(sim).masked_fill(hard, w_hard)
    exp = torch.exp(sim) * w
    terms = []
    for i in torch.where(edh)[0]:
        if pos[i].sum() == 0: continue
        denom = exp[i][~eye[i]].sum()
        num = torch.exp(sim[i][pos[i]]).sum()
        terms.append(-torch.log(num / denom + 1e-12))
    if not terms:
        return torch.tensor(0.0, device=z.device)
    return torch.stack(terms).mean()


def evaluate(probs, targets):
    out, aucs, aps = {}, [], []
    for i, name in enumerate(ALL_COLS):
        gt = targets[:, i]
        if gt.sum() == 0 or gt.sum() == len(gt): auc = ap = float("nan")
        else: auc = roc_auc_score(gt, probs[:, i]); ap = average_precision_score(gt, probs[:, i])
        out[name] = dict(auc=auc, ap=ap, pos=int(gt.sum()))
        if not np.isnan(auc): aucs.append(auc); aps.append(ap)
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_ap"]  = float(np.mean(aps))  if aps  else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    run = S1_DIR / f"stage1_supcon_fold{a.fold}_{PROFILE}"; run.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(MEMMAP_DIR / "memmap_meta.json"))
    print("=" * 70)
    print(f"Stage-1 SupCon | fold={a.fold} profile={PROFILE} batch={CFG['batch']} "
          f"w_contrast={CFG['w_contrast']} w_hard={CFG['w_hard']}")
    print("=" * 70)

    idx   = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    folds = pd.read_parquet(MAN_DIR / "folds.parquet")
    idx = idx[idx["ok"]].merge(folds[["study", "fold"]], on="study", how="left")
    tr = idx[idx.fold != a.fold].copy(); va = idx[idx.fold == a.fold].copy()
    print(f"train={len(tr):,} val={len(va):,} EDH_train={int(tr['epidural'].sum())}")
    priors = [max(tr[c].mean(), 1e-6) for c in ALL_COLS]

    # sampler: enough EDH anchors + SDH hard-negs per batch
    w = np.ones(len(tr))
    w[tr["epidural"].values > 0] = 8.0
    w[(tr["subdural"].values > 0) & (tr["epidural"].values == 0)] = 3.0
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.float32), len(tr), replacement=True)

    trl = DataLoader(MemmapDS(tr, meta, True), batch_size=CFG["batch"], sampler=sampler,
                     num_workers=CFG["workers"], pin_memory=True, persistent_workers=True,
                     prefetch_factor=6, drop_last=True)
    val = DataLoader(MemmapDS(va, meta, False), batch_size=CFG["batch"], shuffle=False,
                     num_workers=CFG["workers"], pin_memory=True, persistent_workers=True,
                     prefetch_factor=6)

    model = SupConNet(CFG["backbone"], priors, CFG["proj_dim"]).to(DEVICE, memory_format=torch.channels_last)
    try: model = torch.compile(model)
    except Exception as e: print("compile skipped:", e)
    cls_loss = LogitAdjustedBCE(priors).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    steps = max(1, len(trl) * CFG["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps, pct_start=0.1)

    best, hist = -1, []
    for ep in range(CFG["epochs"]):
        model.train(); rc = rl = 0.0; t0 = time.time()
        for xb, yb in tqdm(trl, desc=f"E{ep+1}/{CFG['epochs']}", dynamic_ncols=True, leave=False):
            x = gpu_prep(xb); y = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=AMP_DTYPE):
                logits, z = model(x)
                lc = cls_loss(logits, y)
            lcon = supcon_hardneg(z.float(), y, CFG["tau"], CFG["w_hard"])   # fp32
            loss = lc + CFG["w_contrast"] * lcon
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            rl += lc.item(); rc += float(lcon)

        model.eval(); P, T = [], []
        with torch.no_grad():
            for xb, yb in tqdm(val, desc=f"E{ep+1} val", dynamic_ncols=True, leave=False):
                x = gpu_prep(xb)
                with autocast("cuda", dtype=AMP_DTYPE):
                    logits, _ = model(x)
                P.append(torch.sigmoid(logits.float()).cpu().numpy()); T.append(yb.numpy())
        probs, targets = np.concatenate(P), np.concatenate(T); m = evaluate(probs, targets)
        print(f"E{ep+1}: cls={rl/len(trl):.4f} con={rc/len(trl):.4f} "
              f"macroAUC={m['macro_auc']:.4f} EDH_AUC={m['epidural']['auc']:.4f} "
              f"EDH_AP={m['epidural']['ap']:.4f} SDH_AP={m['subdural']['ap']:.4f} ({time.time()-t0:.0f}s)")
        hist.append(dict(epoch=ep+1, macro_auc=m["macro_auc"],
                         **{f"{c}_auc": m[c]["auc"] for c in ALL_COLS},
                         **{f"{c}_ap": m[c]["ap"] for c in ALL_COLS}))
        if m["macro_auc"] > best:
            best = m["macro_auc"]
            torch.save({"model": model.state_dict(), "priors": priors,
                        "cfg": {"backbone": CFG["backbone"], "img": TRAIN_RES},
                        "metrics": m}, run / "best.pt")
            np.savez(run / "val_slice_predictions.npz", probs=probs, targets=targets,
                     image_id=va["image_id"].values.astype(str),
                     study=va["study"].values.astype(str),
                     z=va["z"].values.astype(np.float32))
            print(f"   -> saved (macroAUC={best:.4f})")
    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    print(f"DONE -> {run}")
    print("Note: backbone saved under 'enc.' (03a strips '_orig_mod.' + 'enc.').")


if __name__ == "__main__":
    main()