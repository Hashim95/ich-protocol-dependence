#!/usr/bin/env python3
"""
06_cq500_prepare.py  —  build CQ500 scan-level consensus labels + locate scan folders

CQ500 (Qure.ai) gives 3-radiologist reads per scan. We build a consensus label per
scan by majority vote (>=2 of 3) for each finding, map our model's 6 outputs to
CQ500's, and locate each scan's DICOM directory on disk.

CQ500 -> our label mapping:
    ICH  <-> any            EDH <-> epidural
    IPH  <-> intraparenchymal   IVH <-> intraventricular
    SDH  <-> subdural       SAH <-> subarachnoid

Output: cq500_manifest.parquet  (scan_id, dicom_dir, + 6 consensus labels)

Usage:
    python 06_cq500_prepare.py
"""
import re, warnings
import os
from pathlib import Path
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")

CQ=Path(os.environ.get("ICH_CQ", ""))
OUT=CQ/"cq500_manifest.parquet"
# our-order labels + the CQ500 column suffix each maps to
OUR=["epidural","intraparenchymal","intraventricular","subarachnoid","subdural","any"]
CQ_SUFFIX={"epidural":"EDH","intraparenchymal":"IPH","intraventricular":"IVH",
           "subarachnoid":"SAH","subdural":"SDH","any":"ICH"}


def consensus(df):
    """majority vote (>=2 of 3 readers) per finding -> dict of our-label: 0/1"""
    out={}
    for our,suf in CQ_SUFFIX.items():
        cols=[f"R{r}:{suf}" for r in (1,2,3) if f"R{r}:{suf}" in df.columns]
        if not cols: out[our]=np.nan; continue
        votes=df[cols].fillna(0).astype(float).sum(axis=1)
        out[our]=(votes>=2).astype(int)
    return out


def ct_number(s):
    """extract the scan's CT number: 'CQ500-CT-5' or 'CQ500CT5 CQ500CT5' -> '5'"""
    m=re.search(r"CT[-_ ]?0*(\d+)", str(s), re.I)
    return m.group(1) if m else None


def find_scan_dirs():
    """map CT number -> deepest dir containing DICOMs."""
    scan_dirs={}
    for qct in sorted(CQ.glob("qct*")):
        if not qct.is_dir(): continue
        for scan in qct.iterdir():
            if not scan.is_dir(): continue
            norm=ct_number(scan.name)
            if norm is None: continue
            best=None; best_n=0
            for d,_,files in __import__("os").walk(scan):
                n=sum(1 for f in files if f.lower().endswith(".dcm"))
                if n>best_n: best_n=n; best=d
            if best: scan_dirs[norm]=best
    return scan_dirs


def main():
    reads=pd.read_csv(CQ/"reads.csv")
    print(f"reads.csv: {len(reads)} rows, columns include name/Category + R1-3 findings")
    # consensus labels
    cons=consensus(reads)
    lab=pd.DataFrame(cons); lab["name"]=reads["name"].astype(str)
    lab["scan_norm"]=lab["name"].apply(ct_number)

    print("\nCQ500 consensus prevalence (majority of 3 readers):")
    for our in OUR:
        if our in lab: print(f"  {our:18s} {int(lab[our].sum()):4d} / {len(lab)}  ({100*lab[our].mean():.1f}%)")

    # locate DICOM dirs
    print("\nlocating scan DICOM directories on disk ...")
    dirs=find_scan_dirs()
    print(f"  found {len(dirs)} scan directories under qct*/")
    lab["dicom_dir"]=lab["scan_norm"].map(dirs)
    matched=lab["dicom_dir"].notna().sum()
    print(f"  matched {matched}/{len(lab)} reads rows to on-disk scans")
    if matched < len(lab):
        miss=lab[lab["dicom_dir"].isna()]["name"].head(5).tolist()
        print(f"  (unmatched examples: {miss} — name/folder mismatch; will skip these)")

    lab=lab[lab["dicom_dir"].notna()].reset_index(drop=True)
    lab.to_parquet(OUT,index=False)
    print(f"\nEDH scans available in CQ500: {int(lab['epidural'].sum())} "
          f"(scan-level; small — expected, EDH is rare)")
    print(f"wrote {OUT}  ({len(lab)} scans with labels+paths)")
    print("next: python 07_cq500_evaluate.py --run stage1_ft_fold0_both_workstation")


if __name__=="__main__":
    main()
