#!/usr/bin/env python3
"""
02_train_stage1.py — Stage-1 per-slice classifier (Paper-1 spine).

Gradual unfreeze + discriminative LR + augmentation to stop the EDH overfit
collapse; per-epoch checkpointing with --resume for power cuts; selection on
EDH average precision (the metric the paper needs).

    python 02_train_stage1.py --fold 0 --loss logit_adjusted
    python 02_train_stage1.py --fold 0 --loss logit_adjusted --resume
"""
import os, time, argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast
import timm
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from ich_config import (MAN_DIR, MEMMAP_DIR, S1_DIR, ALL_COLS, SUBTYPES,
                        TRAIN_RES, AMP_DTYPE, DEVICE, PROFILE, SEED, setup_hardware)
warnings.filterwarnings("ignore")
setup_hardware()
np.random.seed(SEED)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CFG = {
    "workstation": dict(backbone="convnext_tiny.in12k_ft_in1k", batch=128, workers=8,
                        epochs=6, head_lr=2e-4, bb_lr=2e-5, wd=0.05, patience=3),
    "laptop":      dict(backbone="convnext_nano.in12k_ft_in1k", batch=32, workers=8,
                        epochs=14, head_lr=2e-4, bb_lr=2e-5, wd=0.05, patience=3),
    "cpu":         dict(backbone="convnext_nano.in12k_ft_in1k", batch=4, workers=2,
                        epochs=1, head_lr=2e-4, bb_lr=2e-5, wd=0.05, patience=3),
}[PROFILE]

UNFREEZE = {1: ["stages.3"], 2: ["stages.2"], 3: ["stages.1"], 4: ["stages.0", "stem"]}


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
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
            if np.random.rand() < 0.3:
                g = np.float32(np.random.uniform(0.9, 1.1))
                x = np.clip(x.astype(np.float32) * g, 0, 255).astype(np.uint8)
        y = r[ALL_COLS].values.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def gpu_prep(x_uint8, train=False):
    x = x_uint8.to(DEVICE, non_blocking=True).float().div_(255.0)
    if x.shape[-1] != TRAIN_RES:
        x = F.interpolate(x, size=(TRAIN_RES, TRAIN_RES), mode="bilinear", align_corners=False)
    if train:
        B = x.size(0)
        ang = (torch.rand(B, device=x.device) - 0.5) * (2 * 8 * np.pi / 180)
        sc  = 1.0 + (torch.rand(B, device=x.device) - 0.5) * 0.1
        tx  = (torch.rand(B, device=x.device) - 0.5) * 0.06
        ty  = (torch.rand(B, device=x.device) - 0.5) * 0.06
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    x = (x - _MEAN.to(DEVICE)) / _STD.to(DEVICE)
    return x.contiguous(memory_format=torch.channels_last)


class Net(nn.Module):
    def __init__(self, backbone, priors):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=True, num_classes=6)
        try: self.net.set_grad_checkpointing(True)
        except Exception: pass
        with torch.no_grad():
            heads = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
            if heads:
                pr = torch.tensor(priors).clamp(1e-6, 1 - 1e-6)
                heads[-1].bias.copy_(torch.log(pr / (1 - pr)).float())
    def forward(self, x): return self.net(x)


def head_params(model):
    lin = [m for m in model.net.modules() if isinstance(m, nn.Linear)]
    return set(id(p) for p in lin[-1].parameters()) if lin else set()

def set_frozen(model, frozen=True):
    hp = head_params(model)
    for p in model.net.parameters():
        p.requires_grad = (not frozen) or (id(p) in hp)

def unfreeze_keys(model, keys):
    for n, p in model.net.named_parameters():
        if any(k in n for k in keys): p.requires_grad = True

def make_opt(model, cfg):
    hp = head_params(model); head, body = [], []
    for p in model.net.parameters():
        if not p.requires_grad: continue
        (head if id(p) in hp else body).append(p)
    groups = [{"params": head, "lr": cfg["head_lr"], "weight_decay": cfg["wd"]}]
    if body: groups.append({"params": body, "lr": cfg["bb_lr"], "weight_decay": cfg["wd"]})
    return torch.optim.AdamW(groups)


class LogitAdjusted(nn.Module):
    def __init__(self, priors, tau=1.0):
        super().__init__()
        pr = torch.tensor(priors).clamp(1e-6, 1 - 1e-6)
        self.register_buffer("lp", torch.log(pr / (1 - pr))); self.tau = tau
    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits + self.tau * self.lp.unsqueeze(0), y)

class WeightedBCE(nn.Module):
    def __init__(self, priors):
        super().__init__()
        pw = [min((1 - p) / max(p, 1e-6), 50.0) for p in priors]
        self.register_buffer("pw", torch.tensor(pw, dtype=torch.float32))
    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw)

class Focal(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0): super().__init__(); self.a, self.g = alpha, gamma
    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits); pt = p * y + (1 - p) * (1 - y)
        at = self.a * y + (1 - self.a) * (1 - y)
        return (at * (1 - pt) ** self.g * bce).mean()

def build_loss(name, priors):
    return {"logit_adjusted": LogitAdjusted(priors),
            "weighted_bce": WeightedBCE(priors), "focal": Focal()}[name]


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


def score_of(m, sel):
    if sel == "edh_ap":  return m["epidural"]["ap"]
    if sel == "edh_auc": return m["epidural"]["auc"]
    return m["macro_auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--loss", default="logit_adjusted",
                    choices=["logit_adjusted", "weighted_bce", "focal"])
    ap.add_argument("--select", default="edh_ap", choices=["edh_ap", "edh_auc", "macro_auc"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run = S1_DIR / f"stage1_fold{args.fold}_{args.loss}_{PROFILE}"; run.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(MEMMAP_DIR / "memmap_meta.json"))
    print("=" * 70)
    print(f"Stage-1 | fold={args.fold} loss={args.loss} select={args.select} "
          f"res={TRAIN_RES} batch={CFG['batch']} epochs={CFG['epochs']}")
    print("=" * 70)

    idx   = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    folds = pd.read_parquet(MAN_DIR / "folds.parquet")
    idx = idx[idx["ok"]].merge(folds[["study", "fold"]], on="study", how="left")
    tr = idx[idx.fold != args.fold].copy(); va = idx[idx.fold == args.fold].copy()
    print(f"train={len(tr):,} val={len(va):,} EDH_train={int(tr['epidural'].sum())}")
    priors = [max(tr[c].mean(), 1e-6) for c in ALL_COLS]

    w = np.where(tr[SUBTYPES].sum(1).values > 0, 3.0, 1.0)
    w[tr["epidural"].values > 0] = 8.0
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.float32), len(tr), replacement=True)

    trl = DataLoader(MemmapDS(tr, meta, True), batch_size=CFG["batch"], sampler=sampler,
                     num_workers=CFG["workers"], pin_memory=True, prefetch_factor=2, drop_last=True)
    val = DataLoader(MemmapDS(va, meta, False), batch_size=CFG["batch"], shuffle=False,
                     num_workers=CFG["workers"], pin_memory=True, prefetch_factor=2)

    model = Net(CFG["backbone"], priors).to(DEVICE, memory_format=torch.channels_last)
    set_frozen(model, True)
    opt = make_opt(model, CFG)
    crit = build_loss(args.loss, priors).to(DEVICE)

    start_ep, best, hist, bad = 0, -1, [], 0
    last_path = run / "last.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        set_frozen(model, True)
        for e, keys in UNFREEZE.items():
            if e <= ck["epoch"]: unfreeze_keys(model, keys)
        model.load_state_dict(ck["model"]); model.to(DEVICE, memory_format=torch.channels_last)
        opt = make_opt(model, CFG)
        try: opt.load_state_dict(ck["opt"])
        except Exception: print("  (optimizer state skipped — param groups changed)")
        start_ep, hist, bad = ck["epoch"] + 1, ck["hist"], ck.get("bad", 0)
        # CRITICAL: if the selection metric changed, the old 'best' is on a
        # different scale (e.g. AUC 0.89 vs AP 0.13) and would block all saves.
        if ck.get("select") == args.select:
            best = ck["best"]
        else:
            best = -1
            print(f"  selection metric changed ({ck.get('select')} -> {args.select}); best reset")
        print(f"RESUMED at epoch {start_ep+1} | best={best:.4f}")

    for ep in range(start_ep, CFG["epochs"]):
        if ep in UNFREEZE:
            unfreeze_keys(model, UNFREEZE[ep]); opt = make_opt(model, CFG)
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [E{ep+1}] unfroze {UNFREEZE[ep]} -> {n/1e6:.1f}M trainable")

        model.train(); rl, t0 = 0.0, time.time()
        for xb, yb in tqdm(trl, desc=f"E{ep+1}/{CFG['epochs']}", mininterval=5.0, ncols=80, leave=False):
            x = gpu_prep(xb, train=True); y = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=AMP_DTYPE):
                loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); rl += loss.item()

        model.eval(); P, T = [], []
        with torch.no_grad():
            for xb, yb in tqdm(val, desc=f"E{ep+1} val", mininterval=5.0, ncols=80, leave=False):
                x = gpu_prep(xb, train=False)
                with autocast("cuda", dtype=AMP_DTYPE):
                    logits = model(x)
                P.append(torch.sigmoid(logits.float()).cpu().numpy()); T.append(yb.numpy())
        probs, targets = np.concatenate(P), np.concatenate(T); m = evaluate(probs, targets)
        print(f"E{ep+1}: loss={rl/len(trl):.4f} macroAUC={m['macro_auc']:.4f} "
              f"EDH_AUC={m['epidural']['auc']:.4f} EDH_AP={m['epidural']['ap']:.4f} ({time.time()-t0:.0f}s)")
        hist.append(dict(epoch=ep+1, train_loss=rl/len(trl), macro_auc=m["macro_auc"],
                         **{f"{c}_auc": m[c]["auc"] for c in ALL_COLS},
                         **{f"{c}_ap": m[c]["ap"] for c in ALL_COLS}))
        pd.DataFrame(hist).to_csv(run / "history.csv", index=False)

        sc = score_of(m, args.select)
        if (not np.isnan(sc)) and sc > best:
            best, bad = sc, 0
            torch.save({"model": model.state_dict(), "priors": priors,
                        "cfg": {"backbone": CFG["backbone"], "img": TRAIN_RES},
                        "loss": args.loss, "metrics": m, "selected_on": args.select}, run / "best.pt")
            np.savez(run / "val_predictions.npz", probs=probs, targets=targets,
                     image_id=va["image_id"].values.astype(str),
                     study=va["study"].values.astype(str),
                     z=va["z"].values.astype(np.float32))
            print(f"   -> best {args.select}={best:.4f} saved")
        else:
            bad += 1

        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                    "best": best, "hist": hist, "bad": bad, "select": args.select}, last_path)

        if bad >= CFG["patience"]:
            print(f"early stop: no {args.select} gain for {bad} epochs"); break

    print(f"DONE best {args.select}={best:.4f} -> {run}")


if __name__ == "__main__":
    main()