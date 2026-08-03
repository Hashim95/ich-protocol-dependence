#!/usr/bin/env python3
"""
03c_conditional_subtype.py — Stage-2 CONDITIONAL detect-then-subtype head (03a features).

Motivated by diagnose_minority_fp.py: if EDH false-positives concentrate on
OTHER-hemorrhage studies (subtype confusion), the bottleneck is disambiguation,
not detection. So:
    detect  d(x)   -> P(any hemorrhage)          trained on ALL studies
    subtype s_c(x) -> P(subtype c | hemorrhage)  trained on any-positive studies
    final   p_c = sigmoid(d) * sigmoid(s_c)      (chain rule)

--method: conditional [contribution] | joint [baseline] | conditional_margin [ablation]

    python 03c_conditional_subtype.py --features stage1_fold0_logit_adjusted_workstation \
                                      --fold 0 --method conditional
"""
import argparse, time, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from ich_config import (MAN_DIR, FEAT_DIR, S2_DIR, ALL_COLS, EDH_I, PROFILE, SEED, DEVICE)
warnings.filterwarnings("ignore")
torch.manual_seed(SEED); np.random.seed(SEED)

CFG = {"workstation": dict(seq_len=32, batch=32, epochs=10),
       "laptop":      dict(seq_len=24, batch=16, epochs=20),
       "cpu":         dict(seq_len=24, batch=16, epochs=12)}[PROFILE]


def load_features(tag):
    shards = sorted(glob.glob(str(FEAT_DIR / tag / "shard_*.npz")))
    assert shards, f"no shards in {FEAT_DIR/tag}"
    F_, L, S, Z = [], [], [], []
    for s in shards:
        d = np.load(s, allow_pickle=True)
        F_.append(d["feat"]); L.append(d["labels"]); S.append(d["study"]); Z.append(d["z"])
    feat = np.concatenate(F_); labels = np.concatenate(L)
    study = np.concatenate(S); z = np.concatenate(Z)
    meta = (FEAT_DIR / tag / "meta.txt").read_text()
    fd = int([l for l in meta.splitlines() if l.startswith("feat_dim")][0].split("=")[1])
    return feat, labels, study, z, fd


class StudyDS(Dataset):
    def __init__(self, studies, s2r, feat, labels, seq_len):
        self.st = studies; self.s2r = s2r; self.feat = feat; self.labels = labels; self.seq = seq_len
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


class CondHead(nn.Module):
    def __init__(self, fd, priors, method="conditional"):
        super().__init__()
        self.method = method
        self.attn = nn.Linear(fd, 6)
        self.detect = nn.Linear(fd, 1)
        self.subtype = nn.Linear(fd, 5)
        self.joint = nn.Linear(fd, 6)
        with torch.no_grad():
            pr = torch.tensor(priors, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            lp = torch.log(pr / (1 - pr))
            self.detect.bias.copy_(lp[5:6]); self.joint.bias.copy_(lp)
            cond = (pr[:5] / pr[5]).clamp(1e-6, 1 - 1e-6)
            self.subtype.bias.copy_(torch.log(cond / (1 - cond)))
    def _pool(self, feats, mask):
        al = self.attn(feats).masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        aw = torch.softmax(al, dim=1)
        return torch.einsum("bsc,bsd->bcd", aw, feats) / \
               aw.sum(1, keepdim=True).clamp(min=1e-6).transpose(1, 2)
    def forward(self, feats, mask):
        pooled = self._pool(feats, mask)
        if self.method == "joint":
            return (torch.einsum("bcd,cd->bc", pooled, self.joint.weight) + self.joint.bias), None
        any_feat = pooled[:, 5, :]; sub_feat = pooled[:, :5, :]
        d_logit = self.detect(any_feat).squeeze(-1)
        s_logit = torch.einsum("bcd,cd->bc", sub_feat, self.subtype.weight) + self.subtype.bias
        return (d_logit, s_logit), None


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
    ap.add_argument("--features", required=True); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--method", default="conditional",
                    choices=["conditional", "joint", "conditional_margin"])
    ap.add_argument("--margin", type=float, default=1.0)
    a = ap.parse_args()

    # tag the run with the feature source, or ablation Stage-2 runs would
    # overwrite the headline results (same fold + method, different features)
    _ft = a.features.replace(f"stage1_fold{a.fold}_", "").replace(f"_{PROFILE}", "")
    _suffix = "" if _ft == "logit_adjusted" else f"_{_ft}"
    run = S2_DIR / f"stage2_cond_fold{a.fold}_{a.method}{_suffix}_{PROFILE}"
    run.mkdir(parents=True, exist_ok=True)
    print("=" * 70); print(f"CONDITIONAL | fold={a.fold} method={a.method}")

    feat, labels, study, z, fd = load_features(a.features)
    folds = pd.read_parquet(MAN_DIR / "folds.parquet"); fo = dict(zip(folds["study"], folds["fold"]))
    order = np.argsort(z, kind="stable"); s2r = {}
    for r in order: s2r.setdefault(study[r], []).append(int(r))
    _all = list(s2r.keys())
    st = np.array([s for s in _all if s in fo])          # drop held-out test studies
    stf = np.array([fo[s] for s in st])
    if len(st) < len(_all):
        print(f"  excluded {len(_all)-len(st):,} studies not in folds.parquet (held-out test)")
    tr = st[stf != a.fold]; va = st[stf == a.fold]
    print(f"train={len(tr):,} val={len(va):,}")

    trl_lab = np.stack([labels[s2r[s]].max(0) for s in tr])
    priors = [max(trl_lab[:, i].mean(), 1e-6) for i in range(6)]

    trd = StudyDS(tr, s2r, feat, labels, CFG["seq_len"]); vad = StudyDS(va, s2r, feat, labels, CFG["seq_len"])
    trl = DataLoader(trd, batch_size=CFG["batch"], shuffle=True, num_workers=4, drop_last=True)
    val = DataLoader(vad, batch_size=CFG["batch"], shuffle=False, num_workers=4)

    model = CondHead(fd, priors, method=a.method).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best, hist = -1, []
    for ep in range(CFG["epochs"]):
        model.train(); rl = 0.0; t0 = time.time()
        for f, y, m in trl:
            f, y, m = f.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out, _ = model(f, m)
            if a.method == "joint":
                loss = F.binary_cross_entropy_with_logits(out, y)
            else:
                d_logit, s_logit = out
                y_any = y[:, 5]; y_sub = y[:, :5]
                L_det = F.binary_cross_entropy_with_logits(d_logit, y_any)
                pos = y_any > 0
                L_sub = F.binary_cross_entropy_with_logits(s_logit[pos], y_sub[pos]) if pos.any() \
                        else torch.tensor(0.0, device=DEVICE)
                loss = L_det + L_sub
                if a.method == "conditional_margin":
                    ep_pos = y_sub[:, EDH_I] > 0
                    if ep_pos.any():
                        edh_l = s_logit[ep_pos, EDH_I]
                        sib = s_logit[ep_pos][:, [1, 2, 3, 4]].max(1).values
                        loss = loss + F.relu(a.margin - (edh_l - sib)).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            rl += loss.item()

        model.eval(); P, T = [], []
        with torch.no_grad():
            for f, y, m in val:
                f, m = f.to(DEVICE), m.to(DEVICE); out, _ = model(f, m)
                if a.method == "joint":
                    p = torch.sigmoid(out)
                else:
                    d_logit, s_logit = out
                    p_any = torch.sigmoid(d_logit).unsqueeze(1)
                    p_final = p_any * torch.sigmoid(s_logit)
                    p = torch.cat([p_final, p_any], dim=1)
                P.append(p.cpu().numpy()); T.append(y.numpy())
        probs, targets = np.concatenate(P), np.concatenate(T); me = evaluate(probs, targets)
        print(f"E{ep+1:2d} loss={rl/len(trl):.4f} macroAUC={me['macro_auc']:.4f} "
              f"macroAP={me['macro_ap']:.4f} EDH_AUC={me['epidural']['auc']:.4f} "
              f"EDH_AP={me['epidural']['ap']:.4f} ({time.time()-t0:.1f}s)")
        hist.append(dict(epoch=ep+1, macro_auc=me["macro_auc"], macro_ap=me["macro_ap"],
                         **{f"{c}_auc": me[c]["auc"] for c in ALL_COLS},
                         **{f"{c}_ap": me[c]["ap"] for c in ALL_COLS}))
        score = me["epidural"]["ap"] if not np.isnan(me["epidural"]["ap"]) else -1
        if score > best:
            best = score          # tracked for logging only; NOT used for selection

    # PROTOCOL: fixed schedule, final-epoch model reported (matches Stage-1).
    # Selecting the checkpoint on this fold's own validation data would bias
    # every pooled OOF metric derived from it.
    torch.save({"model": model.state_dict(), "method": a.method, "metrics": me},
               run / "best.pt")
    np.savez(run / "val_predictions.npz", probs=probs, targets=targets, study=va)
    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    print(f"PROTOCOL: final-epoch model saved (best EDH_AP seen: {best:.4f})")
    print(f"DONE method={a.method} -> {run}")


if __name__ == "__main__":
    main()