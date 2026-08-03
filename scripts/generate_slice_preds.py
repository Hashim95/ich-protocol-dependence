#!/usr/bin/env python3
"""
generate_slice_preds.py  —  make val_slice_predictions.npz from a trained checkpoint

The 7-hour training already saved best.pt but the per-slice predictions weren't
written. This loads best.pt and runs ONE inference pass over the fold's validation
slices to produce the npz that 04c_slice_metrics.py needs. ~10 min, no training.

Usage:
    python generate_slice_preds.py --run stage1_ft_fold0_both_workstation --fold 0
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2, pydicom
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import timm
from tqdm import tqdm
warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("ICH_ROOT", "/path/to/rsna-intracranial-hemorrhage-detection"))
MAN_DIR=ROOT/"manifests"; OUT_DIR=ROOT/"stage1_runs"
SUBTYPES=["epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]
ALL_COLS=SUBTYPES+["any"]; WINDOWS=[(40,80),(80,200),(600,2800)]
_MEAN=np.array([0.485,0.456,0.406],np.float32); _STD=np.array([0.229,0.224,0.225],np.float32)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self,df,img): self.df=df.reset_index(drop=True); self.img=img
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=load_windowed(r["path"],self.img); im=(im-_MEAN)/_STD
        return torch.from_numpy(im).permute(2,0,1).float(), torch.from_numpy(r[ALL_COLS].values.astype(np.float32))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True); ap.add_argument("--fold",type=int,default=0)
    a=ap.parse_args()
    run=OUT_DIR/a.run
    ckpt=torch.load(run/"best.pt", map_location="cpu", weights_only=False)
    backbone=ckpt["cfg"]["backbone"]; img=ckpt["cfg"]["img"]
    print(f"loaded checkpoint: backbone={backbone} img={img} "
          f"(best {ckpt.get('selected_on','?')})")

    model=timm.create_model(backbone,pretrained=False,num_classes=6)
    # 02c wraps the backbone as self.net -> strip the "net." prefix
    sd = {k[4:] if k.startswith("net.") else k: v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd); model=model.to(DEVICE).eval()

    sm=pd.read_parquet(MAN_DIR/"slice_manifest.parquet")
    folds=pd.read_parquet(MAN_DIR/"folds.parquet")
    sm=sm.merge(folds[["study","fold"]],on="study",how="left")
    va=sm[sm.fold==a.fold].copy()
    print(f"val slices: {len(va):,}")

    ld=DataLoader(SliceDS(va,img),batch_size=64,shuffle=False,num_workers=16,pin_memory=True)
    P,T=[],[]
    with torch.no_grad():
        for x,y in tqdm(ld,desc="inference"):
            x=x.to(DEVICE,non_blocking=True)
            with autocast("cuda",enabled=True):
                logits=model(x)
            P.append(torch.sigmoid(logits).float().cpu().numpy()); T.append(y.numpy())
    probs,targets=np.concatenate(P),np.concatenate(T)
    np.savez(run/"val_slice_predictions.npz",
             probs=probs,targets=targets,
             image_id=va["image_id"].values.astype(str),
             study=va["study"].values.astype(str))
    print(f"wrote {run/'val_slice_predictions.npz'}  ({len(probs):,} slices)")
    print("now run: python 04c_slice_metrics.py --run", a.run)


if __name__=="__main__":
    main()
