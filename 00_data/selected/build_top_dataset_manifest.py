import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent
manifest_path = root / "dataset_manifest.tmp.csv"
out_dir = root / "data"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "top_dataset_manifest.tmp.csv"

lr = {
    "agaricus_lepiota", "allbp", "allhyper", "allrep", "analcatdata_authorship",
    "chess", "clean1", "clean2", "coil2000", "dermatology", "dis", "dna",
    "kr_vs_kp", "labor", "mfeat_factors", "mfeat_karhunen", "mfeat_pixel",
    "mofn_3_7_10", "mushroom", "optdigits", "penguins", "prnn_crabs",
    "segmentation", "shuttle", "soybean", "spambase", "texture", "tokyo1",
    "twonorm", "wine_recognition"
}

rf = {
    "agaricus_lepiota", "allbp", "allhyper", "allrep", "analcatdata_authorship",
    "analcatdata_creditscore", "analcatdata_lawsuit", "ann_thyroid", "chess",
    "clean1", "clean2", "collins", "corral", "dermatology", "dis",
    "hypothyroid", "irish", "kr_vs_kp", "monk3", "mushroom", "mux6",
    "new_thyroid", "page_blocks", "pendigits", "penguins", "segmentation",
    "shuttle", "threeOf9", "wine_recognition", "xd6"
}

xgb = {
    "agaricus_lepiota", "allbp", "allhyper", "allhypo", "allrep",
    "analcatdata_bankruptcy", "analcatdata_creditscore", "analcatdata_cyyoung9302",
    "analcatdata_lawsuit", "appendicitis", "backache", "breast_cancer",
    "churn", "clean1", "clean2", "coil2000", "confidence", "dis",
    "hypothyroid", "irish", "labor", "mfeat_karhunen", "mfeat_pixel", "monk3",
    "mushroom", "penguins", "prnn_crabs", "prnn_synth", "spectf", "tokyo1"
}

top_dataset = sorted(lr | rf | xgb)

df = pd.read_csv(manifest_path)
# dataset_manifest uses file names like "name.tsv.gz"; normalize for matching.
df["dataset_base"] = df["dataset"].astype(str).str.replace(".tsv.gz", "", regex=False)

out = df[df["dataset_base"].isin(top_dataset)].copy()
out["in_lr"] = out["dataset_base"].isin(lr)
out["in_rf"] = out["dataset_base"].isin(rf)
out["in_xgb"] = out["dataset_base"].isin(xgb)
out["model_count"] = out[["in_lr", "in_rf", "in_xgb"]].sum(axis=1)
out["top_dataset"] = out["dataset_base"]

ordered_cols = [
    "top_dataset", "dataset", "dataset_base", "in_lr", "in_rf", "in_xgb", "model_count"
] + [c for c in out.columns if c not in {"top_dataset", "dataset", "dataset_base", "in_lr", "in_rf", "in_xgb", "model_count"}]
out = out[ordered_cols].sort_values(["model_count", "top_dataset"], ascending=[False, True])

out.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(f"Rows: {len(out)}")
