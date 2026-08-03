#!/usr/bin/env python3
"""
10_wsgsa_baseline.py — WsGSA (Zhang et al., IEEE TCDS 2023) re-implemented and
trained under OUR patient-disjoint protocol.

WHY THIS EXISTS
---------------
WsGSA reports slice-level EDH F1 = 0.467 at a fixed 0.5 threshold. Our pipeline
reports ~0.13-0.22. Those numbers are not comparable: their split is never
described as patient-disjoint, and slice-level evaluation on a leaky split
inflates rare-class F1 substantially. Citing their number next to ours is a
weakness a reviewer will attack from either direction.

This script removes the confound. It trains THEIR architecture on OUR folds and
evaluates at slice level with threshold 0.5 -- their protocol in every respect
except the split. The resulting EDH F1 is the pivotal number for Paper 1.

FIDELITY: what is faithful, what is substituted
-----------------------------------------------
FAITHFUL (fully specified in the paper):
  * WGSA = position attention (Eqs. 2-3) + channel attention (Eqs. 4-5).
    C/8 compression for PAM query/key, C/4 for value, W/2 x H/2 downsample in
    the channel branch, lambda_p and lambda_c learned from init 0.
  * Element-wise sum of the two attention maps, 1x1 conv, sigmoid -> attention
    map (Eq. 6).
  * Classification branch: attention (x) MSFE features -> 1x1 conv to 6
    channels -> GAP -> sigmoid (Eqs. 8-9).
  * Loss: L = BCE + 0.25 * MSE(attention_map, weak_label)  (Eq. 12, lambda=0.25
    stated explicitly in the paper).
  * WKEM merge rule: CAM (threshold 0.5) UNION unsupervised blood mask,
    6x6 Gaussian smoothed.
  * Two-pass training: a plain soft-attention model produces the CAMs that
    WKEM needs before WsGSA itself can be trained.

SUBSTITUTED (unpublished in the original, documented here and in the paper):
  * WKEM's region-growing HU ranges were "provided by the doctor" and never
    published. We substitute the standard acute-hemorrhage range (50-80 HU),
    implemented directly on the stored uint8 channels:
        brain ch0 (WL 40/WW 80)  : HU [0,80]     -> uint8 = HU * 3.1875
                                   blood 50-80 HU -> 159-255
        bone  ch2 (WL 600/WW2800): HU [-800,2000]-> uint8 = (HU+800)*0.0911
                                   blood 50-80 HU ->  77-80
                                   cortical bone  -> 164-246
    So: (ch0 >= 159) AND (ch2 <= 110) selects blood-density voxels while
    excluding skull, with ~80 uint8 levels of margin. Verified: the
    pre-windowed guard in windowing.py fires on 0/3000 sampled RSNA slices,
    so every slice carries genuine HU semantics.
  * MSFE backbone is unspecified ("a CNN-based network"). We use ConvNeXt-tiny,
    MATCHING our own pipeline, so the comparison isolates the head/attention
    design rather than the backbone.
  * Windowing held constant with our pipeline (brain 40/80, subdural 80/200,
    bone 600/2800). The original's third window is [40,380]. Holding
    preprocessing constant is the controlled choice; it also gives WKEM a
    cleaner skull discriminator than the original had, so if anything this
    favours the baseline.
  * JFM (their 3D smoother) is NOT implemented here. It is a post-hoc smoother
    over slice predictions; our 05_sequence_smooth.py is the equivalent and is
    applied to both methods or neither.

USAGE
-----
    # pass 1 -- soft-attention model that supplies CAMs for WKEM
    python 10_wsgsa_baseline.py --phase cam --fold 0

    # pass 2 -- WsGSA proper, guided by weak knowledge
    python 10_wsgsa_baseline.py --phase wsgsa --fold 0

    # evaluate at THEIR protocol (slice level, threshold 0.5)
    python 10_wsgsa_baseline.py --phase eval --fold 0

Run fold 0 end-to-end and read the EDH F1 before committing folds 1-4.
"""
import argparse, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import roc_auc_score, average_precision_score

from ich_config import (ROOT, MAN_DIR, MEMMAP_DIR, S1_DIR, ALL_COLS, SUBTYPES,
                        TRAIN_RES, DEVICE, setup_hardware)

warnings.filterwarnings("ignore")
setup_hardware()

OUT_DIR = ROOT / "wsgsa_runs"; OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)

CFG = dict(batch=48, epochs=10, lr_head=3e-4, lr_backbone=3e-5,
           workers=8, lam_mse=0.25, cam_thresh=0.5, seq_res=24)

# Blood-mask constants, derived from WINDOWS = [(40,80),(80,200),(600,2800)].
BLOOD_CH0_MIN = 159      # 50 HU in the brain window
BLOOD_CH2_MAX = 110      # excludes skull with ~80 levels of margin

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================ DATA
class MemmapSlices(Dataset):
    """Reads the shared uint8 memmap. Returns raw uint8 so the blood mask can be
    computed from true channel values before ImageNet normalisation."""

    def __init__(self, idx_df, meta, train):
        self.rows = idx_df["row"].values.astype(np.int64)
        self.y = idx_df[ALL_COLS].values.astype(np.float32)
        self.train = train
        self.meta = meta
        self.mm = None       # opened lazily, per worker

    def __len__(self):
        return len(self.rows)

    def _open(self):
        m = self.meta
        self.mm = np.memmap(MEMMAP_DIR / m["dat"], dtype=np.uint8, mode="r",
                            shape=(m["n"], m["c"], m["h"], m["w"]))

    def __getitem__(self, i):
        if self.mm is None:
            self._open()
        x = np.asarray(self.mm[self.rows[i]])          # (3,H,W) uint8
        return torch.from_numpy(x.copy()), torch.from_numpy(self.y[i])


def gpu_prep(x_u8, train):
    """uint8 -> normalised float, plus the raw uint8 kept for masking.
    Augmentation is geometric only, applied identically to image and mask."""
    x = x_u8.to(DEVICE, non_blocking=True)
    if train:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])
    xf = x.float().div_(255.0)
    if TRAIN_RES != xf.shape[-1]:
        xf = F.interpolate(xf, size=(TRAIN_RES, TRAIN_RES),
                           mode="bilinear", align_corners=False)
    xn = (xf - IMNET_MEAN.to(DEVICE)) / IMNET_STD.to(DEVICE)
    return xn.contiguous(memory_format=torch.channels_last), x


def blood_mask(x_u8, out_hw):
    """WKEM's unsupervised branch: blood-density AND not-bone, 6x6 smoothed.

    Substitutes the original's unpublished clinician-provided HU ranges with
    the standard acute-hemorrhage window, applied to the stored channels.
    """
    ch0 = x_u8[:, 0:1].float()
    ch2 = x_u8[:, 2:3].float()
    m = ((ch0 >= BLOOD_CH0_MIN) & (ch2 <= BLOOD_CH2_MAX)).float()
    # 6x6 Gaussian, as specified in the paper
    k = torch.tensor([1., 5., 10., 10., 5., 1.], device=m.device)
    k = (k / k.sum()).view(1, 1, 1, 6)
    m = F.conv2d(F.pad(m, (2, 3, 0, 0), mode="replicate"), k)
    m = F.conv2d(F.pad(m, (0, 0, 2, 3), mode="replicate"), k.transpose(2, 3))
    return F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False).clamp(0, 1)


# ============================================================ MODULES
class MSFE(nn.Module):
    """Multi-scale feature extraction: take features from several backbone
    stages, project to a common width, resample to a common grid, and fuse."""

    def __init__(self, backbone="convnext_tiny", out_ch=256, grid=24):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=True,
                                     features_only=True, out_indices=(1, 2, 3))
        chs = self.net.feature_info.channels()
        self.grid = grid
        self.proj = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in chs])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * len(chs), out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.out_ch = out_ch

    def forward(self, x):
        feats = self.net(x)
        ups = [F.interpolate(p(f), size=(self.grid, self.grid),
                             mode="bilinear", align_corners=False)
               for p, f in zip(self.proj, feats)]
        return self.fuse(torch.cat(ups, 1))


class PositionAttention(nn.Module):
    """Eqs. 2-3. Query/key compressed to C/8, value to C/4, lambda_p from 0."""

    def __init__(self, c):
        super().__init__()
        self.q = nn.Conv2d(c, c // 8, 1)
        self.k = nn.Conv2d(c, c // 8, 1)
        self.v = nn.Conv2d(c, c // 4, 1)
        self.out = nn.Conv2d(c // 4, c, 1)
        self.lam = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).permute(0, 2, 1)      # B, HW, C/8
        k = self.k(x).flatten(2)                       # B, C/8, HW
        att = torch.softmax(torch.bmm(q, k), dim=-1)   # B, HW, HW
        v = self.v(x).flatten(2)                       # B, C/4, HW
        o = torch.bmm(v, att.permute(0, 2, 1)).view(B, C // 4, H, W)
        return self.lam * self.out(o) + x


class ChannelAttention(nn.Module):
    """Eqs. 4-5. Operates on a W/2 x H/2 grid, lambda_c from 0."""

    def __init__(self, c):
        super().__init__()
        self.out = nn.Conv2d(c, c, 1)
        self.lam = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        xd = F.avg_pool2d(x, 2)
        Bd, Cd, Hd, Wd = xd.shape
        f = xd.flatten(2)                              # B, C, HW/4
        att = torch.softmax(torch.bmm(f, f.permute(0, 2, 1)), dim=-1)
        o = torch.bmm(att, f).view(B, C, Hd, Wd)
        o = F.interpolate(self.out(o), size=(H, W), mode="bilinear", align_corners=False)
        return self.lam * o + x


class WGSA(nn.Module):
    """Weakly-guided soft attention: PAM + channel attention, summed, 1x1 conv,
    sigmoid -> a single-channel attention map supervised by weak knowledge."""

    def __init__(self, c):
        super().__init__()
        self.pam = PositionAttention(c)
        self.cam = ChannelAttention(c)
        self.pc = nn.Conv2d(c, c, 3, padding=1)
        self.cc = nn.Conv2d(c, c, 3, padding=1)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, f):
        a = self.pc(self.pam(f)) + self.cc(self.cam(f))
        return torch.sigmoid(self.head(a))             # B,1,H,W in [0,1]


class SoftAttnModel(nn.Module):
    """Pass-1 model: ordinary (unguided) soft attention. Its 6-channel class map
    is the CAM that WKEM consumes."""

    def __init__(self, n_cls=6):
        super().__init__()
        self.msfe = MSFE(grid=CFG["seq_res"])
        c = self.msfe.out_ch
        self.attn = nn.Sequential(nn.Conv2d(c, c, 3, padding=1),
                                  nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, 1, 1), nn.Sigmoid())
        self.cls = nn.Conv2d(c, n_cls, 1)

    def forward(self, x, return_cam=False):
        f = self.msfe(x)
        a = self.attn(f)
        cmap = self.cls(f * a)                         # B,6,H,W  <- the CAM
        logits = cmap.mean((2, 3))
        if return_cam:
            return logits, cmap
        return logits


class WsGSA(nn.Module):
    """The full model: MSFE -> WGSA -> guided classification branch."""

    def __init__(self, n_cls=6):
        super().__init__()
        self.msfe = MSFE(grid=CFG["seq_res"])
        c = self.msfe.out_ch
        self.wgsa = WGSA(c)
        self.integrate = nn.Conv2d(c, c, 1)
        self.cls = nn.Conv2d(c, n_cls, 1)

    def forward(self, x):
        f = self.msfe(x)
        a = self.wgsa(f)                               # B,1,h,w
        g = self.integrate(f * a)
        logits = self.cls(g).mean((2, 3))              # GAP -> Eq. 8
        return logits, a


# ============================================================ WEAK KNOWLEDGE
@torch.no_grad()
def weak_label(cam_model, x_norm, x_u8, hw):
    """WKEM: CAM(>=0.5) UNION unsupervised blood mask -> target for the
    attention branch."""
    _, cmap = cam_model(x_norm, return_cam=True)
    c = cmap[:, :5].amax(1, keepdim=True)              # max over the 5 subtypes
    c = c - c.amin(dim=(2, 3), keepdim=True)
    c = c / (c.amax(dim=(2, 3), keepdim=True) + 1e-6)
    cam_bin = (c >= CFG["cam_thresh"]).float()
    bm = blood_mask(x_u8, hw)
    return torch.clamp(cam_bin + bm, 0, 1)


# ============================================================ TRAIN / EVAL
def build_loaders(fold, train_bs=None):
    idx = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    folds = pd.read_parquet(MAN_DIR / "folds.parquet")
    # inner join: held-out test studies are absent from folds.parquet and are
    # therefore excluded structurally, exactly as in 02_train_stage1.py
    idx = idx[idx["ok"]].merge(folds[["study", "fold"]], on="study", how="inner")
    tr = idx[idx.fold != fold].copy(); va = idx[idx.fold == fold].copy()
    print(f"  train={len(tr):,} val={len(va):,} EDH_train={int(tr['epidural'].sum())}")
    meta = json.load(open(MEMMAP_DIR / "memmap_meta.json"))

    w = np.where(tr[SUBTYPES].sum(1).values > 0, 3.0, 1.0)
    w[tr["epidural"].values > 0] = 8.0
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.float32),
                                    len(tr), replacement=True)
    bs = train_bs or CFG["batch"]
    trl = DataLoader(MemmapSlices(tr, meta, True), batch_size=bs, sampler=sampler,
                     num_workers=CFG["workers"], pin_memory=True, drop_last=True)
    val = DataLoader(MemmapSlices(va, meta, False), batch_size=bs, shuffle=False,
                     num_workers=CFG["workers"], pin_memory=True)
    return trl, val, va


def make_opt(model):
    bb, hd = [], []
    for n, p in model.named_parameters():
        (bb if "msfe.net" in n else hd).append(p)
    return torch.optim.AdamW(
        [{"params": bb, "lr": CFG["lr_backbone"]},
         {"params": hd, "lr": CFG["lr_head"]}], weight_decay=1e-4)


def run_cam_phase(fold):
    print("=" * 70); print(f"WsGSA pass 1/2 | soft-attention (CAM source) | fold={fold}")
    print("=" * 70)
    trl, val, _ = build_loaders(fold)
    model = SoftAttnModel().to(DEVICE).to(memory_format=torch.channels_last)
    opt = make_opt(model)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[CFG["lr_backbone"], CFG["lr_head"]],
        total_steps=CFG["epochs"] * len(trl))
    bce = nn.BCEWithLogitsLoss()
    for ep in range(CFG["epochs"]):
        model.train(); t0 = time.time(); tot = 0.0
        for i, (xb, yb) in enumerate(trl):
            xn, _ = gpu_prep(xb, True); yb = yb.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = bce(model(xn), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
            if i % 200 == 0:
                print(f"   E{ep+1} {i}/{len(trl)} loss={tot/(i+1):.4f}", flush=True)
        print(f"  epoch {ep+1}/{CFG['epochs']} loss={tot/len(trl):.4f} "
              f"({time.time()-t0:.0f}s)")
    d = OUT_DIR / f"wsgsa_cam_fold{fold}"; d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, d / "cam_model.pt")
    print(f"[saved] {d/'cam_model.pt'}")


def run_wsgsa_phase(fold):
    print("=" * 70); print(f"WsGSA pass 2/2 | guided attention | fold={fold}")
    print("=" * 70)
    ck = OUT_DIR / f"wsgsa_cam_fold{fold}" / "cam_model.pt"
    if not ck.exists():
        raise SystemExit(f"missing {ck} — run --phase cam first")
    cam_model = SoftAttnModel().to(DEVICE).to(memory_format=torch.channels_last)
    cam_model.load_state_dict(torch.load(ck, map_location=DEVICE)["model"])
    cam_model.eval()
    for p in cam_model.parameters():
        p.requires_grad_(False)

    trl, val, va_df = build_loaders(fold)
    model = WsGSA().to(DEVICE).to(memory_format=torch.channels_last)
    opt = make_opt(model)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[CFG["lr_backbone"], CFG["lr_head"]],
        total_steps=CFG["epochs"] * len(trl))
    bce = nn.BCEWithLogitsLoss(); mse = nn.MSELoss()
    hist = []

    for ep in range(CFG["epochs"]):
        model.train(); t0 = time.time(); lc = lm = 0.0
        for i, (xb, yb) in enumerate(trl):
            xn, xu = gpu_prep(xb, True); yb = yb.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, attn = model(xn)
                wk = weak_label(cam_model, xn, xu, attn.shape[-2:])
                l_cls = bce(logits, yb)
                l_att = mse(attn.float(), wk.float())
                loss = l_cls + CFG["lam_mse"] * l_att      # Eq. 12
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            lc += l_cls.item(); lm += l_att.item()
            if i % 200 == 0:
                print(f"   E{ep+1} {i}/{len(trl)} bce={lc/(i+1):.4f} "
                      f"mse={lm/(i+1):.4f}", flush=True)
        print(f"  epoch {ep+1}/{CFG['epochs']} bce={lc/len(trl):.4f} "
              f"mse={lm/len(trl):.4f} ({time.time()-t0:.0f}s)")
        hist.append(dict(epoch=ep + 1, bce=lc / len(trl), mse=lm / len(trl)))

    d = OUT_DIR / f"wsgsa_fold{fold}"; d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, d / "wsgsa.pt")
    pd.DataFrame(hist).to_csv(d / "history.csv", index=False)

    # inference on the validation fold
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for xb, yb in val:
            xn, _ = gpu_prep(xb, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(xn)
            P.append(torch.sigmoid(logits.float()).cpu().numpy())
            Y.append(yb.numpy())
    P = np.concatenate(P); Y = np.concatenate(Y)
    np.savez_compressed(d / "val_predictions.npz", probs=P, targets=Y,
                        study=va_df["study"].values.astype(str))
    print(f"[saved] {d/'val_predictions.npz'}  ({len(P):,} slices)")
    evaluate(P, Y, fold)


def evaluate(P, Y, fold):
    """WsGSA's own protocol: SLICE level, fixed threshold 0.5."""
    rows = []
    for i, name in enumerate(ALL_COLS):
        y, p = Y[:, i].astype(int), P[:, i]
        if y.sum() == 0: continue
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-12)
        rows.append(dict(cls=name, pos=int(y.sum()),
                         AUC=float(roc_auc_score(y, p)),
                         AP=float(average_precision_score(y, p)),
                         sensitivity=sens, precision=prec, F1=f1,
                         specificity=tn / max(tn + fp, 1),
                         accuracy=(tp + tn) / len(y), TP=tp, FP=fp, FN=fn, TN=tn))
    df = pd.DataFrame(rows)
    avg = {c: (float(df[c].mean()) if df[c].dtype != object else "") for c in df.columns}
    avg["cls"] = "AVERAGE"; avg["pos"] = int(df["pos"].sum())
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    out = RESULTS / f"wsgsa_slice_thr05_fold{fold}.csv"
    df.to_csv(out, index=False)

    edh = df[df.cls == "epidural"]["F1"].iloc[0]
    print("\n" + "=" * 70)
    print("WsGSA under OUR protocol — slice level, threshold 0.5")
    print("=" * 70)
    print(df[["cls", "pos", "AUC", "AP", "sensitivity", "precision", "F1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  EDH F1 = {edh:.4f}    (WsGSA reported 0.467)")
    print(f"  avg F1 = {avg['F1']:.4f}    (WsGSA reported 0.746)")
    print(f"\n[logged] {out}")
    print("\nDECISION GATE (master plan section 7):")
    if edh < 0.25:
        print("  EDH F1 < 0.25 -> published 0.467 does not survive patient-disjoint")
        print("  evaluation. Protocol-dependence confirmed. Proceed to folds 1-4.")
    elif edh < 0.35:
        print("  EDH F1 in [0.25,0.35) -> partial confirmation, substantial")
        print("  protocol-dependent degradation. Proceed to folds 1-4.")
    elif edh < 0.40:
        print("  EDH F1 in [0.35,0.40) -> mild degradation. Report honestly;")
        print("  the protocol-dependence headline weakens. Proceed.")
    else:
        print("  EDH F1 >= 0.40 -> the base method genuinely outperforms ours on")
        print("  rare-class F1 under identical conditions. REPORT THIS HONESTLY.")
        print("  Paper 1 pivots to the ceiling + rigor + external-validation claims.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["cam", "wsgsa", "eval"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    if a.batch:  CFG["batch"] = a.batch
    if a.epochs: CFG["epochs"] = a.epochs

    if a.phase == "cam":
        run_cam_phase(a.fold)
    elif a.phase == "wsgsa":
        run_wsgsa_phase(a.fold)
    else:
        d = OUT_DIR / f"wsgsa_fold{a.fold}" / "val_predictions.npz"
        if not d.exists():
            raise SystemExit(f"missing {d}")
        z = np.load(d, allow_pickle=True)
        evaluate(z["probs"], z["targets"], a.fold)


if __name__ == "__main__":
    main()