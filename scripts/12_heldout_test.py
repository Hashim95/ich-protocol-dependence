#!/usr/bin/env python3
"""
12_heldout_test.py — FINAL EVALUATION. Run exactly once.

WHAT THIS IS
------------
The 4,348-study held-out test partition (70 EDH) has not been read by any
training, feature-extraction, model-selection, threshold-selection, or figure
script since it was carved on 26 July 2026. This script scores it.

It is the only unbiased estimate in the paper. Every other number — including
the pooled out-of-fold results — comes from data that participated, however
indirectly, in some design decision: which loss, which architecture, which
threshold policy, whether to smooth, how many epochs.

WHY IT RUNS ONCE
----------------
A held-out set retains its meaning only while it remains unseen. If a result
here is disappointing and the pipeline is adjusted in response, the partition
silently becomes a validation set and the paper's central methodological claim
is undermined by the paper's own conduct. This script therefore refuses to
overwrite an existing result unless --force is passed, and records a receipt
noting when it ran and against which configuration.

If you find yourself wanting to change something after seeing these numbers:
report them as they are, and put the change in future work.

WHAT IT EVALUATES
-----------------
The complete pipeline as specified in the paper:
  Stage-1 ConvNeXt-tiny (5 folds, final-epoch, logit-adjusted)
    -> frozen features
    -> Stage-2 conditional head (5 folds, final-epoch)
    -> ensemble by averaging the five folds' study-level probabilities
    -> fixed 0.5 threshold

Reporting the ensemble is the honest choice: it is what a deployment would use,
and it is what the released weights constitute. Per-fold test scores are also
logged so the ensemble gain is visible.

USAGE
    python 12_heldout_test.py --dry-run     # verify wiring, score nothing
    python 12_heldout_test.py               # THE run
"""
import argparse, datetime, glob, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)

from ich_config import (ROOT, MAN_DIR, MEMMAP_DIR, S1_DIR, S2_DIR, FEAT_DIR,
                        ALL_COLS, TRAIN_RES, AMP_DTYPE, DEVICE, PROFILE,
                        setup_hardware)

setup_hardware()

RESULTS = ROOT / "results"
RECEIPT = RESULTS / "HELDOUT_TEST_RECEIPT.json"
SEQ_LEN = 32
NICE = {"epidural": "EDH", "intraparenchymal": "IPH", "intraventricular": "IVH",
        "subarachnoid": "SAH", "subdural": "SDH", "any": "Any"}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- model defs
class CondHead(nn.Module):
    """Must match 03c_conditional_subtype.py exactly."""

    def __init__(self, fd, priors, method="conditional"):
        super().__init__()
        self.method = method
        self.attn = nn.Linear(fd, 6)
        self.detect = nn.Linear(fd, 1)
        self.subtype = nn.Linear(fd, 5)
        self.joint = nn.Linear(fd, 6)

    def _pool(self, feats, mask):
        al = self.attn(feats).masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        aw = torch.softmax(al, dim=1)
        return torch.einsum("bsc,bsd->bcd", aw, feats) / \
               aw.sum(1, keepdim=True).clamp(min=1e-6).transpose(1, 2)

    def forward(self, feats, mask):
        pooled = self._pool(feats, mask)
        if self.method == "joint":
            return torch.einsum("bcd,cd->bc", pooled, self.joint.weight) + self.joint.bias
        any_feat = pooled[:, 5, :]; sub_feat = pooled[:, :5, :]
        d = self.detect(any_feat).squeeze(-1)
        s = torch.einsum("bcd,cd->bc", sub_feat, self.subtype.weight) + self.subtype.bias
        return d, s


# ---------------------------------------------------------------- stage 1
def stage1_features(fold, rows_df, meta, batch=128):
    """Run one fold's frozen Stage-1 backbone over the test slices."""
    import timm
    ck = S1_DIR / f"stage1_fold{fold}_logit_adjusted_{PROFILE}" / "best.pt"
    if not ck.exists():
        raise SystemExit(f"missing Stage-1 checkpoint: {ck}")
    d = torch.load(ck, map_location="cpu", weights_only=False)
    enc = timm.create_model(d["cfg"]["backbone"], pretrained=False,
                            num_classes=0, global_pool="avg")
    sd = {k.replace("net.", "", 1): v for k, v in d["model"].items()
          if k.startswith("net.")}
    sd = {k: v for k, v in sd.items() if not k.startswith("head.fc")}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"    (unexpected keys ignored: {len(unexpected)})")
    enc = enc.to(DEVICE).eval().to(memory_format=torch.channels_last)

    mm = np.memmap(MEMMAP_DIR / meta["dat"], dtype=np.uint8, mode="r",
                   shape=(meta["n"], meta["c"], meta["h"], meta["w"]))
    rows = rows_df["row"].values.astype(np.int64)
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            x = torch.from_numpy(np.asarray(mm[rows[s:s + batch]]))
            x = x.to(DEVICE).float().div_(255.0)
            if x.shape[-1] != TRAIN_RES:
                x = F.interpolate(x, size=(TRAIN_RES, TRAIN_RES),
                                  mode="bilinear", align_corners=False)
            x = ((x - _MEAN.to(DEVICE)) / _STD.to(DEVICE)).contiguous(
                memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                f = enc(x)
            out.append(f.float().cpu().numpy())
            if (s // batch) % 50 == 0:
                print(f"      {s:,}/{len(rows):,}", flush=True)
    return np.concatenate(out)


# ---------------------------------------------------------------- stage 2
def stage2_predict(fold, feats, s2r, studies, fd):
    ck = S2_DIR / f"stage2_cond_fold{fold}_conditional_{PROFILE}" / "best.pt"
    if not ck.exists():
        raise SystemExit(f"missing Stage-2 checkpoint: {ck}")
    d = torch.load(ck, map_location="cpu", weights_only=False)
    head = CondHead(fd, None, method="conditional")
    head.load_state_dict(d["model"])
    head = head.to(DEVICE).eval()

    P = np.zeros((len(studies), 6), np.float32)
    with torch.no_grad():
        for i in range(0, len(studies), 64):
            chunk = studies[i:i + 64]
            fb, mb = [], []
            for st in chunk:
                r = s2r[st]; n = len(r)
                idx = (np.linspace(0, n - 1, SEQ_LEN).astype(int) if n >= SEQ_LEN
                       else list(range(n)) + [n - 1] * (SEQ_LEN - n))
                fb.append(feats[[r[j] for j in idx]])
                m = np.zeros(SEQ_LEN, np.float32); m[:min(n, SEQ_LEN)] = 1.0
                mb.append(m)
            f = torch.from_numpy(np.stack(fb)).float().to(DEVICE)
            m = torch.from_numpy(np.stack(mb)).to(DEVICE)
            dlog, slog = head(f, m)
            p_any = torch.sigmoid(dlog).unsqueeze(1)
            p = torch.cat([p_any * torch.sigmoid(slog), p_any], dim=1)
            P[i:i + 64] = p.cpu().numpy()
    return P


# ---------------------------------------------------------------- metrics
def score(P, Y, label):
    rows = []
    rng = np.random.default_rng(31337)
    for i, c in enumerate(ALL_COLS):
        y, p = Y[:, i].astype(int), P[:, i]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        auc = float(roc_auc_score(y, p)); ap = float(average_precision_score(y, p))
        a, b = [], []
        for _ in range(1000):
            s = rng.choice(len(y), len(y), replace=True)
            if 0 < y[s].sum() < len(s):
                a.append(roc_auc_score(y[s], p[s]))
                b.append(average_precision_score(y[s], p[s]))
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        se = tp / max(tp + fn, 1); pr = tp / max(tp + fp, 1)
        rows.append(dict(cls=NICE[c], pos=int(y.sum()), AUC=auc,
                         AUC_lo=float(np.percentile(a, 2.5)) if a else np.nan,
                         AUC_hi=float(np.percentile(a, 97.5)) if a else np.nan,
                         AP=ap,
                         AP_lo=float(np.percentile(b, 2.5)) if b else np.nan,
                         AP_hi=float(np.percentile(b, 97.5)) if b else np.nan,
                         brier=float(brier_score_loss(y, p)),
                         sensitivity=se, specificity=tn / max(tn + fp, 1),
                         precision=pr, F1=2 * pr * se / max(pr + se, 1e-12),
                         TP=tp, FP=fp, FN=fn, TN=tn))
    df = pd.DataFrame(rows)
    mac = df.select_dtypes("number").mean().to_dict()
    mac["cls"] = "MACRO"; mac["pos"] = int(df["pos"].sum())
    df = pd.concat([df, pd.DataFrame([mac])], ignore_index=True)
    print(f"\n{label}")
    print(df[["cls", "pos", "AUC", "AUC_lo", "AUC_hi", "AP", "brier",
              "sensitivity", "precision", "F1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-score an already-scored test set (think hard first)")
    a = ap.parse_args()

    if RECEIPT.exists() and not a.force and not a.dry_run:
        r = json.loads(RECEIPT.read_text())
        raise SystemExit(
            f"\nThe held-out test set was already scored on {r['scored_at']}.\n"
            f"Re-scoring after seeing the result converts it into a validation\n"
            f"set and undermines the paper's own methodological argument.\n"
            f"Existing result: {RESULTS/'heldout_test_metrics.csv'}\n"
            f"Pass --force only if the earlier run was invalid for a reason\n"
            f"unrelated to its numbers (e.g. it crashed, or used wrong weights).")

    test = pd.read_parquet(MAN_DIR / "heldout_test.parquet")
    idx = pd.read_parquet(MEMMAP_DIR / "memmap_index.parquet")
    idx = idx[idx["ok"]].merge(test[["study"]], on="study", how="inner")
    idx = idx.sort_values(["study", "z"], kind="stable").reset_index(drop=True)

    folds = pd.read_parquet(MAN_DIR / "folds.parquet")
    leak = set(test["study"]) & set(folds["study"])
    assert not leak, f"FATAL: {len(leak)} test studies appear in folds.parquet"

    studies = idx["study"].unique()
    s2r = {s: g.index.tolist() for s, g in idx.groupby("study", sort=False)}
    Y = idx.groupby("study", sort=False)[ALL_COLS].max().loc[studies].values

    print("=" * 70)
    print("HELD-OUT TEST EVALUATION — this partition has never been scored")
    print("=" * 70)
    print(f"studies : {len(studies):,}")
    print(f"slices  : {len(idx):,}")
    for i, c in enumerate(ALL_COLS):
        print(f"  {c:<18} {int(Y[:, i].sum()):>5} positive")
    print(f"leak check: 0 test studies in folds.parquet — OK")

    if a.dry_run:
        print("\n--dry-run: wiring verified, nothing scored")
        return

    meta = json.load(open(MEMMAP_DIR / "memmap_meta.json"))
    fold_P, per_fold = [], []
    for f in range(5):
        print(f"\n[fold {f}] Stage-1 features over {len(idx):,} test slices")
        feats = stage1_features(f, idx, meta)
        print(f"[fold {f}] Stage-2 conditional head")
        P = stage2_predict(f, feats, s2r, studies, feats.shape[1])
        fold_P.append(P)
        d = score(P, Y, f"--- fold {f} alone ---")
        d["fold"] = f; per_fold.append(d)
        del feats

    P_ens = np.mean(fold_P, axis=0)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ens = score(P_ens, Y, "=== HELD-OUT TEST — 5-FOLD ENSEMBLE (REPORTED) ===")

    ens.to_csv(RESULTS / "heldout_test_metrics.csv", index=False)
    pd.concat(per_fold).to_csv(RESULTS / "heldout_test_per_fold.csv", index=False)
    np.savez_compressed(RESULTS / "heldout_test_predictions.npz",
                        probs=P_ens, targets=Y, study=studies.astype(str))
    RECEIPT.write_text(json.dumps({
        "scored_at": stamp, "n_studies": int(len(studies)),
        "n_slices": int(len(idx)),
        "positives": {c: int(Y[:, i].sum()) for i, c in enumerate(ALL_COLS)},
        "pipeline": "ConvNeXt-tiny logit-adjusted -> conditional head, 5-fold ensemble",
        "threshold": 0.5,
        "note": "Scored once. Do not re-score."}, indent=2))

    print(f"\n[logged] heldout_test_metrics.csv, heldout_test_per_fold.csv")
    print(f"[receipt] {RECEIPT}")
    print("\nReport these numbers as they are. If they differ from the")
    print("development estimates, that difference is itself a finding.")


if __name__ == "__main__":
    main()