#!/usr/bin/env python3
"""
02c_train_stage1_ft.py  —  ICH pipeline, step 2c   (RECOVER STABLE EDH)

Diagnostic finding: the current Stage-1 hits epidural AUC ~0.945 at EPOCH 1 then
OVERFITS it away (0.945 -> 0.92 -> 0.87), while the old NeuroFusion code held EDH
AUC 0.91-0.93 STABLY for many epochs. Two things the old code did that the current
Stage-1 does not:

  A. GRADUAL UNFREEZE — backbone frozen for the first few epochs (train head only),
     then unfrozen progressively. Acts as a regularizer that prevents the rare class
     from being overfit away early.
  B. EPIDURAL MODEL SELECTION — keep the checkpoint at the epoch where EDH is best
     (on the VALIDATION fold, AUC = threshold-free = no leakage), instead of keeping
     the last / best-macro epoch (which discards the EDH peak).

Ablation flag --mode isolates each:
    baseline     : all params trainable from epoch 1, select on macro AUC  (== current 02)
    unfreeze     : gradual unfreeze, select on macro AUC
    select_edh   : all trainable, select on validation EDH AUC
    both         : gradual unfreeze + EDH selection            [expected best]

Laptop-testable. Same windowing / logit-adjusted loss / honest eval as everything else.

Usage:
    python 02c_train_stage1_ft.py --fold 0 --mode both
"""
import os, argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2, pydicom
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import timm
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
warnings.filterwarnings("ignore")

import os; ROOT = Path(os.environ.get("ICH_ROOT", "/path/to/rsna-intracranial-hemorrhage-detection"))
MAN_DIR=ROOT/"manifests"; OUT_DIR=ROOT/"stage1_runs"
SUBTYPES=["epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]
ALL_COLS=SUBTYPES+["any"]; WINDOWS=[(40,80),(80,200),(600,2800)]
_MEAN=np.array([0.485,0.456,0.406],np.float32); _STD=np.array([0.229,0.224,0.225],np.float32)
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)

PROFILE="workstation"
CFG={"laptop":dict(backbone="convnext_nano.in12k_ft_in1k",img=224,batch=48,workers=8,
                   subset=40000,epochs=8,amp=True),
     "workstation":dict(backbone="convnext_tiny.in12k_ft_in1k",img=384,batch=24,workers=16,
                        subset=None,epochs=10,amp=True)}[PROFILE]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gradual unfreeze schedule (convnext has 4 stages: stages.0-3). Head always trainable.
# epoch < FREEZE_UNTIL: head only. then unfreeze from the top down.
FREEZE_UNTIL = 2       # epochs 0..1 head-only
UNFREEZE_STAGES = {2:["stages.3"], 3:["stages.2"], 4:["stages.1"], 5:["stages.0","stem"]}


def load_windowed(path,img):
    try:
        ds=pydicom.dcmread(path); a=ds.pixel_array.astype(np.float32)
        a=a*float(getattr(ds,"RescaleSlope",1.0))+float(getattr(ds,"RescaleIntercept",0.0))
    except Exception: return np.zeros((img,img,3),np.float32)
    ch=[]
    for c,w in WINDOWS:
        lo,hi=c-w/2,c+w/2; ch.append(np.clip((a-lo)/(hi-lo+1e-6),0,1))
    im=np.stack(ch,-1)
    if im.shape[:2]!=(img,img): im=cv2.resize(im,(img,img))
    return im.astype(np.float32)


class SliceDS(Dataset):
    def __init__(self,df,img,train=True): self.df=df.reset_index(drop=True); self.img=img; self.train=train
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=load_windowed(r["path"],self.img)
        if self.train and np.random.rand()<0.5: im=im[:,::-1,:].copy()
        im=(im-_MEAN)/_STD
        return torch.from_numpy(im).permute(2,0,1).float(), torch.from_numpy(r[ALL_COLS].values.astype(np.float32))


class Net(nn.Module):
    def __init__(self,backbone,priors):
        super().__init__()
        self.net=timm.create_model(backbone,pretrained=True,num_classes=6)
        with torch.no_grad():
            m=None
            for mod in self.net.modules():
                if isinstance(mod,nn.Linear): m=mod
            if m is not None:
                pr=torch.tensor(priors).clamp(1e-6,1-1e-6); m.bias.copy_(torch.log(pr/(1-pr)).float())
    def forward(self,x): return self.net(x)


class LogitAdjustedBCE(nn.Module):
    def __init__(self,priors,tau=1.0):
        super().__init__(); pr=torch.tensor(priors).clamp(1e-6,1-1e-6)
        self.register_buffer("lp",torch.log(pr/(1-pr))); self.tau=tau
    def forward(self,logits,targets):
        return F.binary_cross_entropy_with_logits(logits+self.tau*self.lp.unsqueeze(0),targets)


def set_backbone_frozen(model, frozen=True):
    """Freeze/unfreeze all backbone params EXCEPT the final classifier head."""
    # find the classifier linear (last Linear) to keep trainable
    linears=[m for m in model.net.modules() if isinstance(m,nn.Linear)]
    head=linears[-1] if linears else None
    for n,p in model.net.named_parameters():
        p.requires_grad = not frozen
    if head is not None:
        for p in head.parameters(): p.requires_grad=True


def unfreeze_matching(model, keys):
    for n,p in model.net.named_parameters():
        if any(k in n for k in keys): p.requires_grad=True


def evaluate(probs,targets):
    out,aucs,aps={},[],[]
    for i,name in enumerate(ALL_COLS):
        gt=targets[:,i]
        if gt.sum()==0 or gt.sum()==len(gt): auc=ap=float("nan")
        else: auc=roc_auc_score(gt,probs[:,i]); ap=average_precision_score(gt,probs[:,i])
        out[name]=dict(auc=auc,ap=ap,pos=int(gt.sum()))
        if not np.isnan(auc): aucs.append(auc); aps.append(ap)
    out["macro_auc"]=float(np.mean(aucs)) if aucs else float("nan")
    out["macro_ap"]=float(np.mean(aps)) if aps else float("nan")
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,default=0)
    ap.add_argument("--mode",default="both",choices=["baseline","unfreeze","select_edh","both"])
    a=ap.parse_args()
    do_unfreeze = a.mode in ("unfreeze","both")
    select_on_edh = a.mode in ("select_edh","both")

    run=OUT_DIR/f"stage1_ft_fold{a.fold}_{a.mode}_{PROFILE}"; run.mkdir(parents=True,exist_ok=True)
    print("="*70); print(f"Stage-1 FT | fold={a.fold} mode={a.mode} "
                          f"(unfreeze={do_unfreeze} select_edh={select_on_edh})")

    sm=pd.read_parquet(MAN_DIR/"slice_manifest.parquet")
    folds=pd.read_parquet(MAN_DIR/"folds.parquet"); sm=sm.merge(folds[["study","fold"]],on="study",how="left")
    tr=sm[sm.fold!=a.fold].copy(); va=sm[sm.fold==a.fold].copy()
    if CFG["subset"]:
        pos=tr[tr[SUBTYPES].sum(1)>0]; neg=tr[tr[SUBTYPES].sum(1)==0]
        tr=pd.concat([pos.sample(min(len(pos),CFG["subset"]//2),random_state=SEED),
                      neg.sample(min(len(neg),CFG["subset"]//2),random_state=SEED)])
        va=va.sample(min(len(va),CFG["subset"]//3),random_state=SEED)
    print(f"train={len(tr):,} val={len(va):,} EDH_train={int(tr['epidural'].sum())}")
    priors=[max(tr[c].mean(),1e-6) for c in ALL_COLS]

    w=np.where(tr[SUBTYPES].sum(1).values>0,3.0,1.0)
    w[tr["epidural"].values>0]=8.0
    sampler=WeightedRandomSampler(torch.tensor(w,dtype=torch.float32),num_samples=len(tr),replacement=True)
    trl=DataLoader(SliceDS(tr,CFG["img"],True),batch_size=CFG["batch"],sampler=sampler,
                   num_workers=CFG["workers"],pin_memory=True,drop_last=True)
    val=DataLoader(SliceDS(va,CFG["img"],False),batch_size=CFG["batch"],shuffle=False,
                   num_workers=CFG["workers"],pin_memory=True)

    model=Net(CFG["backbone"],priors).to(DEVICE)
    crit=LogitAdjustedBCE(priors).to(DEVICE)
    if do_unfreeze:
        set_backbone_frozen(model,frozen=True)   # start head-only
        print(f"  gradual unfreeze ON: backbone frozen until epoch {FREEZE_UNTIL}")
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=2e-4,weight_decay=1e-4)
    scaler=GradScaler("cuda",enabled=CFG["amp"])

    best_score,best_tag=-1,("macro_auc" if not select_on_edh else "epidural_auc")
    hist=[]
    for ep in range(CFG["epochs"]):
        # apply unfreeze schedule
        if do_unfreeze and ep in UNFREEZE_STAGES:
            unfreeze_matching(model, UNFREEZE_STAGES[ep])
            # rebuild optimizer to include newly-trainable params
            opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=1e-4,weight_decay=1e-4)
            n_train=sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [epoch {ep}] unfroze {UNFREEZE_STAGES[ep]} -> {n_train/1e6:.1f}M trainable")

        model.train(); rl=0; t0=time.time()
        for x,y in tqdm(trl,desc=f"E{ep+1}/{CFG['epochs']} {a.mode}"):
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda",enabled=CFG["amp"]):
                loss=crit(model(x),y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); rl+=loss.item()

        model.eval(); P,T=[],[]
        with torch.no_grad():
            for x,y in val:
                x=x.to(DEVICE)
                with autocast("cuda",enabled=CFG["amp"]):
                    logits=model(x)
                P.append(torch.sigmoid(logits).float().cpu().numpy()); T.append(y.numpy())
        probs,targets=np.concatenate(P),np.concatenate(T); me=evaluate(probs,targets)
        print(f"E{ep+1}: loss={rl/len(trl):.4f} macroAUC={me['macro_auc']:.4f} "
              f"EDH_AUC={me['epidural']['auc']:.4f} EDH_AP={me['epidural']['ap']:.4f} ({time.time()-t0:.0f}s)")
        hist.append(dict(epoch=ep+1,macro_auc=me["macro_auc"],
                         **{f"{c}_auc":me[c]["auc"] for c in ALL_COLS},
                         **{f"{c}_ap":me[c]["ap"] for c in ALL_COLS}))

        score = me["epidural"]["auc"] if select_on_edh else me["macro_auc"]
        if not np.isnan(score) and score>best_score:
            best_score=score
            torch.save({"model":model.state_dict(),"priors":priors,"cfg":CFG,
                        "mode":a.mode,"metrics":me,"selected_on":best_tag},run/"best.pt")
            print(f"   -> saved (best {best_tag}={best_score:.4f} | EDH_AUC={me['epidural']['auc']:.4f})")
    pd.DataFrame(hist).to_csv(run/"history.csv",index=False)
    print(f"DONE mode={a.mode} best {best_tag}={best_score:.4f} -> {run}"); print("="*70)


if __name__=="__main__":
    main()
