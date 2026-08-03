#!/usr/bin/env python3
"""
03_train_stage2_sala.py  —  ICH pipeline, step 3 of N   (THE NOVELTY)

Stage-2: study-level classifier with SALA — Spatially-Adaptive Logit Adjustment.

--------------------------------------------------------------------------------
THE NOVEL OBJECT — SALA
--------------------------------------------------------------------------------
Menon et al. (ICLR 2021) logit adjustment corrects a classifier's decision
boundary for class imbalance by a FIXED offset:      z_c  ->  z_c + tau*log(pi_c)
The offset is the same for every input x. That is optimal only if the class prior
is constant over the input space — but for ICH it is NOT: a skull-adjacent bright
crescent locally raises the epidural prior; a mid-brain slice lowers it.

SALA makes the offset INPUT-CONDITIONAL via a weakly-supervised attention map:

        z_c(x)  ->  z_c(x) + tau * log(pi_c) * g_c(a_c(x))
        g_c(a)  =  1 - lambda_c * a_c(x),     lambda_c in [0,1] learnable

  * a_c(x) in [0,1] : attention confidence that class-c evidence is present in x
  * where evidence is ABSENT (a->0): g->1, FULL prior correction (stay skeptical)
  * where evidence is PRESENT (a->1): g->1-lambda, DAMPENED correction (trust it)

Properties (all unit-tested, see methods section):
  P1. lambda_c = 0  =>  SALA reduces EXACTLY to Menon logit adjustment  (clean limit)
  P2. attention dampens the prior penalty precisely where lesion evidence localizes
  P3. fully differentiable in {lambda_c, attention params} -> trained end-to-end

This unifies "logit adjustment" and "attention refinement" into ONE operator, and
the ablation (fixed-LA vs attention-only vs SALA) is the paper's core result.

--------------------------------------------------------------------------------
STRUCTURE
--------------------------------------------------------------------------------
Stage-1 (02_) gave per-slice features/logits. Here we:
  1. build per-study slice sequences (ordered by z),
  2. an attention module produces per-slice, per-class attention a_c,
  3. attention-pool slices -> study representation,
  4. classify, and apply SALA at the study level,
  5. evaluate study-level (comparable to literature + CQ500).

Ablation flag --method:
    sala          : full novel operator                       [DEFAULT, contribution]
    fixed_la      : Menon fixed logit adjustment (lambda=0)    [baseline]
    attn_only     : attention pooling, NO logit adjustment     [baseline]
    mean_pool     : plain mean pooling, no attn, no LA         [baseline]

Usage:
    python 03_train_stage2_sala.py --fold 0 --method sala
    python 03_train_stage2_sala.py --fold 0 --method fixed_la
    python 03_train_stage2_sala.py --fold 0 --method attn_only
    python 03_train_stage2_sala.py --fold 0 --method mean_pool
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import cv2, pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import timm
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================= #
ROOT = Path(os.environ.get("ICH_ROOT", "/path/to/rsna-intracranial-hemorrhage-detection"))
MAN_DIR  = ROOT / "manifests"
OUT_DIR  = ROOT / "stage2_runs"
SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
ALL_COLS = SUBTYPES + ["any"]
WINDOWS  = [(40, 80), (80, 200), (600, 2800)]
SEED     = 42
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD  = np.array([0.229, 0.224, 0.225], np.float32)

PROFILE = "laptop"
CFG = {
    "laptop": dict(backbone="convnext_nano.in12k_ft_in1k", img=224, seq_len=24,
                   batch=4, workers=4, subset_studies=400, epochs=3, amp=True),
    "workstation": dict(backbone="convnext_tiny.in12k_ft_in1k", img=384, seq_len=32,
                        batch=8, workers=12, subset_studies=None, epochs=10, amp=True),
}[PROFILE]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)


# ============================================================================= #
# DATA — per-study slice sequences
# ============================================================================= #
def load_windowed(path, img):
    try:
        ds = pydicom.dcmread(path)
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    except Exception:
        return np.zeros((img, img, 3), np.float32)
    ch = []
    for c, w in WINDOWS:
        lo, hi = c - w / 2, c + w / 2
        ch.append(np.clip((a - lo) / (hi - lo + 1e-6), 0, 1))
    im = np.stack(ch, -1)
    if im.shape[:2] != (img, img):
        im = cv2.resize(im, (img, img), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)


class StudyDS(Dataset):
    """One item = one study: a sequence of up to seq_len slices + study labels."""
    def __init__(self, slice_df, study_df, img, seq_len, train=True):
        self.img, self.seq_len, self.train = img, seq_len, train
        self.study_df = study_df.reset_index(drop=True)
        # group slice paths by study, ordered by z
        g = slice_df.sort_values("z").groupby("study")
        self.by_study = {s: sub for s, sub in g}

    def __len__(self):
        return len(self.study_df)

    def __getitem__(self, i):
        row = self.study_df.iloc[i]
        sub = self.by_study[row["study"]]
        paths = sub["path"].tolist()
        # uniformly sample seq_len slices across the study
        n = len(paths)
        if n >= self.seq_len:
            idx = np.linspace(0, n - 1, self.seq_len).astype(int)
        else:
            idx = list(range(n)) + [n - 1] * (self.seq_len - n)  # pad with last
        mask = np.array([1.0 if j < n else 0.0 for j in range(self.seq_len)], np.float32)
        mask[:min(n, self.seq_len)] = 1.0

        imgs = []
        for j in idx:
            im = load_windowed(paths[j], self.img)
            if self.train and np.random.rand() < 0.5:
                im = im[:, ::-1, :].copy()
            im = (im - _MEAN) / _STD
            imgs.append(torch.from_numpy(im).permute(2, 0, 1).float())
        x = torch.stack(imgs)                        # (S,3,H,W)
        y = torch.from_numpy(row[ALL_COLS].values.astype(np.float32))
        m = torch.from_numpy(mask)
        return x, y, m


# ============================================================================= #
# MODEL — backbone -> per-slice feats -> attention -> SALA head
# ============================================================================= #
class SALAHead(nn.Module):
    """Attention pooling over slices + Spatially-Adaptive Logit Adjustment."""
    def __init__(self, feat_dim, n_cls, priors, method="sala", tau=1.0):
        super().__init__()
        self.method, self.tau, self.n_cls = method, tau, n_cls
        # per-class attention scorer: feat -> per-class attention logit per slice
        self.attn = nn.Linear(feat_dim, n_cls)
        # per-class classifier from pooled feature
        self.cls = nn.Linear(feat_dim, n_cls)
        # SALA gate strength, one per class, in [0,1] via sigmoid(raw)
        self.lam_raw = nn.Parameter(torch.zeros(n_cls))   # start at 0.5 after sigmoid
        priors = torch.tensor(priors, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        self.register_buffer("log_prior", torch.log(priors / (1 - priors)))
        # prior-init the classifier bias for a stable start
        with torch.no_grad():
            self.cls.bias.copy_(self.log_prior.clone())

    def forward(self, feats, mask):
        # feats: (B,S,D)  mask: (B,S)
        B, S, D = feats.shape
        attn_logits = self.attn(feats)               # (B,S,C)  per-slice per-class
        # mask padded slices out of the softmax
        m = mask.unsqueeze(-1)                        # (B,S,1)
        attn_logits = attn_logits.masked_fill(m == 0, -1e4)

        if self.method in ("sala", "attn_only", "fixed_la"):
            attn_w = torch.softmax(attn_logits, dim=1)     # (B,S,C) over slices
        else:  # mean_pool
            attn_w = (m / m.sum(1, keepdim=True).clamp(min=1)).expand(-1, -1, self.n_cls)

        # per-class "evidence present" signal a_c(x) in [0,1].
        # Use the PEAK pre-softmax attention logit over valid slices, squashed by sigmoid.
        # A strong peak (some slice strongly indicates class c) -> a_c -> 1 (evidence present);
        # a flat/weak response -> a_c -> 0 (no localized evidence). This gives lambda a real,
        # well-scaled gradient (unlike max-of-softmax, which saturates near 1/S).
        peak_logit = attn_logits.masked_fill(m == 0, -1e4).max(dim=1).values   # (B,C)
        a_c = torch.sigmoid(peak_logit)                # (B,C) in (0,1)

        # attention-pooled feature per class -> but classifier is shared over feat,
        # so pool feats with the class-agnostic mean of attention for representation,
        # and keep a_c for the SALA gate.
        pooled = torch.einsum("bsc,bsd->bcd", attn_w, feats) / \
                 attn_w.sum(1, keepdim=True).clamp(min=1e-6).transpose(1, 2)  # (B,C,D)
        # per-class logit: dot pooled_c with classifier row c
        W = self.cls.weight                           # (C,D)
        b = self.cls.bias                             # (C,)
        logits = torch.einsum("bcd,cd->bc", pooled, W) + b      # (B,C)

        # ---- SALA / logit adjustment ----
        if self.method == "sala":
            lam = torch.sigmoid(self.lam_raw)         # (C,) in [0,1]
            gate = 1.0 - lam.unsqueeze(0) * a_c       # (B,C)
            logits = logits + self.tau * self.log_prior.unsqueeze(0) * gate
        elif self.method == "fixed_la":
            logits = logits + self.tau * self.log_prior.unsqueeze(0)  # Menon
        # attn_only / mean_pool: no logit adjustment
        return logits, attn_w, a_c


class Stage2Net(nn.Module):
    def __init__(self, backbone, n_cls, priors, method):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        self.feat_dim = self.enc.num_features
        self.head = SALAHead(self.feat_dim, n_cls, priors, method=method)

    def forward(self, x, mask):
        B, S, C, H, W = x.shape
        f = self.enc(x.view(B * S, C, H, W)).view(B, S, -1)   # (B,S,D)
        return self.head(f, mask)


# ============================================================================= #
def evaluate(probs, targets):
    out, aucs, aps = {}, [], []
    for i, name in enumerate(ALL_COLS):
        gt = targets[:, i]
        if gt.sum() == 0 or gt.sum() == len(gt):
            auc = ap = float("nan")
        else:
            auc = roc_auc_score(gt, probs[:, i]); ap = average_precision_score(gt, probs[:, i])
        out[name] = dict(auc=auc, ap=ap, pos=int(gt.sum()))
        if not np.isnan(auc): aucs.append(auc); aps.append(ap)
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_ap"]  = float(np.mean(aps)) if aps else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--method", default="sala",
                    choices=["sala", "fixed_la", "attn_only", "mean_pool"])
    args = ap.parse_args()

    run = OUT_DIR / f"stage2_fold{args.fold}_{args.method}_{PROFILE}"
    run.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"Stage-2 SALA | profile={PROFILE} fold={args.fold} method={args.method}")
    print(f"backbone={CFG['backbone']} img={CFG['img']} seq_len={CFG['seq_len']} device={DEVICE}")
    print("=" * 70)

    slice_man = pd.read_parquet(MAN_DIR / "slice_manifest.parquet")
    study_man = pd.read_parquet(MAN_DIR / "study_manifest.parquet")
    folds     = pd.read_parquet(MAN_DIR / "folds.parquet")
    study_man = study_man.merge(folds[["study", "fold"]], on="study", how="left")

    tr_std = study_man[study_man.fold != args.fold].copy()
    va_std = study_man[study_man.fold == args.fold].copy()
    if CFG["subset_studies"]:
        pos = tr_std[tr_std[SUBTYPES].sum(1) > 0]
        neg = tr_std[tr_std[SUBTYPES].sum(1) == 0]
        tr_std = pd.concat([pos.sample(min(len(pos), CFG["subset_studies"] // 2), random_state=SEED),
                            neg.sample(min(len(neg), CFG["subset_studies"] // 2), random_state=SEED)])
        va_std = va_std.sample(min(len(va_std), CFG["subset_studies"] // 2), random_state=SEED)
        print(f"[laptop subset] train studies={len(tr_std)}  val studies={len(va_std)}")
    else:
        print(f"train studies={len(tr_std):,}  val studies={len(va_std):,}")

    priors = [max(tr_std[c].mean(), 1e-6) for c in ALL_COLS]
    print("study priors:", {c: round(p, 4) for c, p in zip(ALL_COLS, priors)})

    tr_ds = StudyDS(slice_man, tr_std, CFG["img"], CFG["seq_len"], train=True)
    va_ds = StudyDS(slice_man, va_std, CFG["img"], CFG["seq_len"], train=False)
    tr_ld = DataLoader(tr_ds, batch_size=CFG["batch"], shuffle=True,
                       num_workers=CFG["workers"], pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=CFG["batch"], shuffle=False,
                       num_workers=CFG["workers"], pin_memory=True)

    model = Stage2Net(CFG["backbone"], 6, priors, args.method).to(DEVICE)
    # SALA's lambda (and the attention scorer) govern the contribution and must train
    # FAST — like temperature params — not at backbone speed. Separate high-LR group.
    sala_params, base_params = [], []
    for n, p in model.named_parameters():
        if "lam_raw" in n or "head.attn" in n:
            sala_params.append(p)
        else:
            base_params.append(p)
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": 1e-4, "weight_decay": 1e-4},
        {"params": sala_params, "lr": 1e-2, "weight_decay": 0.0},   # 100x for lambda/attn
    ])
    steps = max(1, len(tr_ld) * CFG["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[1e-4, 1e-2], total_steps=steps, pct_start=0.1)
    scaler = GradScaler("cuda", enabled=CFG["amp"])
    # loss: SALA/fixed_la already fold the prior into logits, so plain BCE here
    bce = nn.BCEWithLogitsLoss()

    best_auc, hist = -1, []
    for ep in range(CFG["epochs"]):
        model.train(); rl, t0 = 0.0, time.time()
        for x, y, m in tqdm(tr_ld, desc=f"E{ep+1}/{CFG['epochs']} train"):
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=CFG["amp"]):
                logits, _, _ = model(x, m)
                loss = bce(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            rl += loss.item()

        model.eval(); P, T = [], []
        with torch.no_grad():
            for x, y, m in tqdm(va_ld, desc=f"E{ep+1}/{CFG['epochs']} val"):
                x, m = x.to(DEVICE), m.to(DEVICE)
                with autocast("cuda", enabled=CFG["amp"]):
                    logits, _, _ = model(x, m)
                P.append(torch.sigmoid(logits).float().cpu().numpy()); T.append(y.numpy())
        probs, targets = np.concatenate(P), np.concatenate(T)
        me = evaluate(probs, targets)
        print(f"\nEpoch {ep+1}: loss={rl/len(tr_ld):.4f}  macroAUC={me['macro_auc']:.4f}  "
              f"EDH_AUC={me['epidural']['auc']:.4f}  EDH_AP={me['epidural']['ap']:.4f}  "
              f"({time.time()-t0:.0f}s)")
        for c in ALL_COLS:
            print(f"    {c:18s} AUC={me[c]['auc']:.4f}  AP={me[c]['ap']:.4f}  pos={me[c]['pos']}")
        # SALA introspection: show learned lambda (evidence-trust) per class
        if args.method == "sala":
            lam = torch.sigmoid(model.head.lam_raw).detach().cpu().numpy()
            print("    SALA lambda (evidence trust):",
                  {c: round(float(l), 3) for c, l in zip(ALL_COLS, lam)})
        hist.append(dict(epoch=ep+1, macro_auc=me["macro_auc"],
                         **{f"{c}_auc": me[c]["auc"] for c in ALL_COLS},
                         **{f"{c}_ap": me[c]["ap"] for c in ALL_COLS}))
        if me["macro_auc"] > best_auc:
            best_auc = me["macro_auc"]
            torch.save({"model": model.state_dict(), "priors": priors,
                        "method": args.method, "metrics": me}, run / "best.pt")
            np.savez(run / "val_predictions.npz", probs=probs, targets=targets,
                     study=va_std["study"].values)
            print(f"    -> best (macroAUC={best_auc:.4f}) saved")

    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    print("=" * 70)
    print(f"DONE. method={args.method}  best macroAUC={best_auc:.4f}  -> {run}")
    print("=" * 70)


if __name__ == "__main__":
    main()