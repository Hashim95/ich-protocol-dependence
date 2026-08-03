# smoke_test.py
import pandas as pd, pydicom, numpy as np, glob
from pathlib import Path

ROOT = Path("/home/ivision/Documents/Hashim/RSNA ICH Dataset/rsna-intracranial-hemorrhage-detection")
csv  = ROOT / "stage_2_train.csv"
dcmdir = ROOT / "stage_2_train"

print("CSV exists:", csv.exists(), "| DICOM dir exists:", dcmdir.exists())

df = pd.read_csv(csv)
print("Raw CSV rows:", len(df), "| sample ID:", df.iloc[0]["ID"])

# split ID_<hash>_<subtype>  ->  image_id="ID_<hash>", subtype="<subtype>"
df[["a","b","subtype"]] = df["ID"].str.split("_", expand=True)
df["image_id"] = df["a"] + "_" + df["b"]
df["Label"] = df["Label"].astype("int8")

# FIX: pivot_table + max handles the duplicate (image_id, subtype) rows in stage-2 CSV
wide = df.pivot_table(index="image_id", columns="subtype", values="Label", aggfunc="max")
print("Unique slices:", len(wide))
print("Prevalence:\n", wide.mean().round(4))

# window ONE dicom
dcms = glob.glob(str(dcmdir / "*.dcm"))
print("DICOM files found:", len(dcms))
ds = pydicom.dcmread(dcms[0])
arr = ds.pixel_array.astype(np.float32)
arr = arr*float(getattr(ds,"RescaleSlope",1)) + float(getattr(ds,"RescaleIntercept",0))
def win(a,c,w):
    lo,hi=c-w/2,c+w/2; return np.clip((a-lo)/(hi-lo),0,1)
img = np.stack([win(arr,40,80), win(arr,80,200), win(arr,600,2800)], -1)
print("Windowed shape:", img.shape, "| HU range:", round(float(arr.min())), "to", round(float(arr.max())),
      "| windowed range:", round(float(img.min()),3), round(float(img.max()),3))
print("\n✅ Smoke test passed — data path, labels, and windowing all work.")