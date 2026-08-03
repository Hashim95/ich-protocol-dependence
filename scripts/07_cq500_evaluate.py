#!/usr/bin/env python3
"""
07_cq500_evaluate.py — EXTERNAL validation on CQ500 (5-fold Stage-1 ensemble).

Runs all 5 RSNA-trained Stage-1 models on CQ500 scans, averages their per-slice
predictions (ensemble), aggregates to scan level (high percentile), evaluates vs
3-radiologist consensus. Uses the shared windowing authority so preprocessing is
pixel-identical to RSNA. Logs to results/.

    python 07_cq500_evaluate.py --folds 0 1 2 3 4 --pct 95
"""
import argparse, warnings, os, datetime
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch.amp import autocast
import timm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

from ich_config import (S1_DIR, CQ_ROOT, ROOT, AMP_DTYPE, DEVICE, TRAIN_RES, setup_hardware)
from windowing import window_dicom
warnings.filterwarnings("ignore")
setup_hardware()

RESULTS = ROOT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
OUR = ["epidural","intraparenchymal","intraventricular","subarachnoid","subdural","any"]
RNG = np.random.default_rng(42)


def _stamp(): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_model(run):
    ck = torch.load(S1_DIR / run / "best.pt", map_location="cpu", weights_only=False)
    backbone = ck["cfg"]["backbone"]
    model = timm.create_model(backbone, pretrained=False, num_classes=6)
    sd = {k[4:] if k.startswith("net.") else k: v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def scan_slice_paths(dicom_dir):
    fs = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm")]
    return sorted(fs)


def boot_ci(y, p, metric, n=1000):
    idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s): continue
        try: v.append(metric(y[s], p[s]))
        except Exception: pass
    if not v: return float("nan"), float("nan"), float("nan")
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--pct", type=float, default=95.0)
    a = ap.parse_args()

    models = [load_model(f"stage1_fold{f}_logit_adjusted_workstation") for f in a.folds]
    print(f"loaded {len(models)} fold models for ensemble")

    man = pd.read_parquet(CQ_ROOT / "cq500_manifest.parquet")
    print(f"CQ500 scans: {len(man)}")

    scan_probs, scan_labels = [], []
    for _, r in tqdm(man.iterrows(), total=len(man), desc="CQ500"):
        fs = scan_slice_paths(r["dicom_dir"])
        if not fs: continue
        batch = []
        for f in fs:
            im = window_dicom(f, TRAIN_RES, normalize=True)   # identical to RSNA windowing
            batch.append(im)
        if not batch: continue
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().to(DEVICE)
        # average sigmoid over the 5 models
        preds = np.zeros((len(x), 6), np.float32)
        with torch.no_grad():
            for m in models:
                out = []
                for i in range(0, len(x), 64):
                    with autocast("cuda", dtype=AMP_DTYPE):
                        out.append(torch.sigmoid(m(x[i:i+64]).float()).cpu().numpy())
                preds += np.concatenate(out)
        preds /= len(models)
        scan_probs.append(np.percentile(preds, a.pct, axis=0))
        scan_labels.append(r[OUR].values.astype(float))

    P = np.array(scan_probs); Y = np.array(scan_labels)
    print(f"\nEvaluated {len(P)} scans (5-fold ensemble, agg={a.pct:.0f}th pct)")
    print("=" * 74)
    print(f"{'class':18s} {'AUC [95% CI]':24s} {'AP':7s} {'F1@.5':7s} {'pos':>5s}")
    print("-" * 74)
    rows, aucs = [], []
    for i, name in enumerate(OUR):
        y = Y[:, i].astype(int); p = P[:, i]
        if np.isnan(y).any() or y.sum() == 0:
            print(f"{name:18s} (no positive consensus labels)"); continue
        auc, lo, hi = boot_ci(y, p, roc_auc_score)
        apv = average_precision_score(y, p)
        f1 = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
        aucs.append(auc)
        rows.append(dict(cls=name, pos=int(y.sum()), auc=auc, auc_lo=lo, auc_hi=hi, ap=apv, f1=f1))
        print(f"{name:18s} {auc:.3f} [{lo:.3f},{hi:.3f}]   {apv:.3f}   {f1:.3f}   {int(y.sum()):>5}")
    print("-" * 74)
    print(f"{'MACRO AUC':18s} {np.nanmean(aucs):.4f}")
    print("=" * 74)
    out = RESULTS / f"cq500_external_ensemble_{_stamp()}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[logged] {out}")
    print("External validation on independent cohort — the differentiator vs WsGSA.")
    print("Note: EDH has 12 positives; its CI is wide by necessity, report honestly.")


if __name__ == "__main__":
    main()