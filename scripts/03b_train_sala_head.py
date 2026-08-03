#!/usr/bin/env python3
"""
03b_train_sala_head.py — Stage-2 SALA head on cached frozen features (03a).

SALA — Spatially-Adaptive Logit Adjustment:
    z_c(x) -> z_c(x) + tau*log(pi_c) * (1 - lambda_c * a_c(x))
  lambda_c in [0,1] learnable; lambda=0 recovers Menon (ICLR'21) exactly.

FIX vs the original: the evidence signal a_c and the attention-pooling weights are
now produced by SEPARATE heads (self.evid vs self.attn), so the gate can move
independently of the pooled feature it gates. This makes "attention dampens the
prior where evidence localizes" a clean, defensible claim and keeps lambda's
gradient well-conditioned.

--method: sala [contribution] | fixed_la | attn_only | mean_pool  (ablation)

    python 03b_train_sala_head.py --features stage1_fold0_logit_adjusted_workstation \
                                  --fold 0 --method sala
"""
import argparse, time, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from ich_config import (MAN_DIR, FEAT_DIR, S2_DIR, ALL_COLS, PROFILE, SEED, DEVICE)
warnings.filterwarnings("ignore")
torch.manual_seed(SEED); np.random.seed(SEED)

CFG = {"workstation": dict(seq_len=32, batch=32, epochs=25),
       "laptop":      dict(seq_len=24, batch=16, epochs=15),
       "cpu":         dict(seq_len=24, batch=16, epochs=10)}[PROFILE]


def load_features(tag):
    shards = sorted(glob.glob(str(FEAT_DIR / tag / "shard_*.npz")))
    assert shards, f"no feature shards in {FEAT_DIR/tag}"
    feats, labels, studies, zs = [], [], [], []
    for s in shards:
        d = np.load(s, allow_pickle=True)
        feats.append(d["feat"]); labels.append(d["labels"])
        studies.append(d["study"]); zs.append(d["z"])
    feat = np.concatenate(feats); labels = np.concatenate(labels)
    study = np.concatenate(studies); z = np.concatenate(zs)
    meta = (FEAT_DIR / tag / "meta.txt").read_text()
    fd = int([l for l in meta.splitlines() if l.startswith("feat_dim")][0].split("=")[1])
    return feat, labels, study, z, fd


class StudyFeatDS(Dataset):
    def __init__(self, study_list, s2r, feat, labels, seq_len):
        self.st = study_list; self.s2r = s2r; self.feat = feat
        self.labels = labels; self.seq = seq_len
    def __len__(self): return len(self.st)
    def __getitem__(self, i):
        rows = self.s2r[self.st[i]]; n = len(rows)
        idx = np.linspace(0, n - 1, self.seq).astype(int) if n >= self.seq \
              else list(range(n)) + [n - 1] * (self.seq - n)
        sel = [rows[j] for j in idx]
        f = torch.from_numpy(self.feat[sel].astype(np.float32))
        m = torch.zeros(self.seq); m[:min(n, self.seq)] = 1.0
        y = torch.from_numpy(self.labels[rows].max(0).astype(np.float32))
        return f, y, m


class SALAHead(nn.Module):
    def __init__(self, feat_dim, n_cls, priors, method="sala", tau=1.0):
        super().__init__()
        self.method, self.tau, self.n_cls = method, tau, n_cls
        self.attn = nn.Linear(feat_dim, n_cls)     # pooling attention
        self.evid = nn.Linear(feat_dim, n_cls)     # SEPARATE evidence detector (fix)
        self.cls  = nn.Linear(feat_dim, n_cls)
        self.lam_raw = nn.Parameter(torch.zeros(n_cls))
        pr = torch.tensor(priors, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        self.register_buffer("log_prior", torch.log(pr / (1 - pr)))
        with torch.no_grad(): self.cls.bias.copy_(self.log_prior.clone())

    def forward(self, feats, mask):
        B, S, D = feats.shape
        mfill = mask.unsqueeze(-1) == 0
        al = self.attn(feats).masked_fill(mfill, -1e4)             # (B,S,C)
        if self.method in ("sala", "attn_only", "fixed_la"):
            aw = torch.softmax(al, dim=1)
        else:  # mean_pool
            m = mask.unsqueeze(-1)
            aw = (m / m.sum(1, keepdim=True).clamp(min=1)).expand(-1, -1, self.n_cls)

        # evidence signal from the SEPARATE head, peak over valid slices
        ev = self.evid(feats).masked_fill(mfill, -1e4)
        a_c = torch.sigmoid(ev.max(dim=1).values)                  # (B,C), independent of aw

        pooled = torch.einsum("bsc,bsd->bcd", aw, feats) / \
                 aw.sum(1, keepdim=True).clamp(min=1e-6).transpose(1, 2)   # (B,C,D)
        logits = torch.einsum("bcd,cd->bc", pooled, self.cls.weight) + self.cls.bias

        if self.method == "sala":
            lam = torch.sigmoid(self.lam_raw)
            logits = logits + self.tau * self.log_prior.unsqueeze(0) * (1 - lam.unsqueeze(0) * a_c)
        elif self.method == "fixed_la":
            logits = logits + self.tau * self.log_prior.unsqueeze(0)
        return logits, a_c


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--method", default="sala", choices=["sala", "fixed_la", "attn_only", "mean_pool"])
    ap.add_argument("--select", default="edh_ap", choices=["edh_ap", "macro_auc"])
    a = ap.parse_args()

    run = S2_DIR / f"stage2_sala_fold{a.fold}_{a.method}_{PROFILE}"; run.mkdir(parents=True, exist_ok=True)
    print("=" * 70); print(f"SALA head | fold={a.fold} method={a.method} select={a.select}")

    feat, labels, study, z, fd = load_features(a.features)
    print(f"loaded {len(feat):,} slice features, dim={fd}")
    folds = pd.read_parquet(MAN_DIR / "folds.parquet"); fo = dict(zip(folds["study"], folds["fold"]))

    order = np.argsort(z, kind="stable"); s2r = {}
    for r in order: s2r.setdefault(study[r], []).append(int(r))
    all_st = np.array(list(s2r.keys())); st_fold = np.array([fo.get(s, -1) for s in all_st])
    tr = all_st[st_fold != a.fold]; va = all_st[st_fold == a.fold]
    print(f"train studies={len(tr):,} val studies={len(va):,}")

    trl_lab = np.stack([labels[s2r[s]].max(0) for s in tr])
    priors = [max(trl_lab[:, i].mean(), 1e-6) for i in range(6)]

    trd = StudyFeatDS(tr, s2r, feat, labels, CFG["seq_len"])
    vad = StudyFeatDS(va, s2r, feat, labels, CFG["seq_len"])
    trl = DataLoader(trd, batch_size=CFG["batch"], shuffle=True, num_workers=4, drop_last=True)
    val = DataLoader(vad, batch_size=CFG["batch"], shuffle=False, num_workers=4)

    model = SALAHead(fd, 6, priors, method=a.method).to(DEVICE)
    fast = [p for n, p in model.named_parameters() if "lam_raw" in n or "attn" in n or "evid" in n]
    base = [p for n, p in model.named_parameters() if not ("lam_raw" in n or "attn" in n or "evid" in n)]
    opt = torch.optim.AdamW([{"params": base, "lr": 1e-3, "weight_decay": 1e-4},
                             {"params": fast, "lr": 1e-2, "weight_decay": 0.0}])
    bce = nn.BCEWithLogitsLoss()

    best, hist = -1, []
    for ep in range(CFG["epochs"]):
        model.train(); rl = 0.0; t0 = time.time()
        for f, y, m in trl:
            f, y, m = f.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(f, m); loss = bce(logits, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            rl += loss.item()
        model.eval(); P, T = [], []
        with torch.no_grad():
            for f, y, m in val:
                f, m = f.to(DEVICE), m.to(DEVICE)
                logits, _ = model(f, m)
                P.append(torch.sigmoid(logits).cpu().numpy()); T.append(y.numpy())
        probs, targets = np.concatenate(P), np.concatenate(T); me = evaluate(probs, targets)
        print(f"E{ep+1:2d} loss={rl/len(trl):.4f} macroAUC={me['macro_auc']:.4f} "
              f"macroAP={me['macro_ap']:.4f} EDH_AUC={me['epidural']['auc']:.4f} "
              f"EDH_AP={me['epidural']['ap']:.4f} ({time.time()-t0:.1f}s)")
        if a.method == "sala":
            lam = torch.sigmoid(model.lam_raw).detach().cpu().numpy()
            print("     lambda:", {c: round(float(l), 3) for c, l in zip(ALL_COLS, lam)})
        hist.append(dict(epoch=ep+1, macro_auc=me["macro_auc"], macro_ap=me["macro_ap"],
                         **{f"{c}_auc": me[c]["auc"] for c in ALL_COLS},
                         **{f"{c}_ap": me[c]["ap"] for c in ALL_COLS}))
        score = (me["epidural"]["ap"] if a.select == "edh_ap" else me["macro_auc"])
        if not np.isnan(score) and score > best:
            best = score
            torch.save({"model": model.state_dict(), "priors": priors, "method": a.method,
                        "metrics": me}, run / "best.pt")
            np.savez(run / "val_predictions.npz", probs=probs, targets=targets, study=va)
    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    print(f"DONE method={a.method} best {a.select}={best:.4f} -> {run}")


if __name__ == "__main__":
    main()