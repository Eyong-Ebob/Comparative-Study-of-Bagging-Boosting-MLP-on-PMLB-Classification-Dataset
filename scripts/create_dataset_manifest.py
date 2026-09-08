import os
from glob import glob
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


def detect_target_column(df):
    if "class" in df.columns:
        return "class"
    if "target" in df.columns:
        return "target"
    return df.columns[-1]


def summarize_dataset(path):
    df = pd.read_csv(path, compression="gzip", sep="\t")
    target_col = detect_target_column(df)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    class_counts = y.value_counts(dropna=False)
    n_classes = int(class_counts.shape[0])

    # Squared-distance imbalance from perfect class balance.
    # 0.0 means perfectly balanced; values approach 1.0 with extreme imbalance.
    if n_classes > 0 and len(y) > 0:
        class_proportions = class_counts / float(len(y))
        perfect_balance = 1.0 / n_classes
        class_imbalance = float(((class_proportions - perfect_balance) ** 2).sum())
    else:
        class_imbalance = 0.0

    numeric_features = 0
    categorical_features = 0
    for col in feature_cols:
        if is_numeric_dtype(X[col]):
            numeric_features += 1
        else:
            categorical_features += 1

    if numeric_features and categorical_features:
        feature_type = "both"
    elif numeric_features:
        feature_type = "numeric"
    elif categorical_features:
        feature_type = "categorical"
    else:
        feature_type = "unknown"

    missing_values_count = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_values_ratio = float(missing_values_count / total_cells) if total_cells > 0 else 0.0

    return {
        "dataset": os.path.basename(path),
        "target_column": target_col,
        "rows": int(df.shape[0]),
        "features": int(len(feature_cols)),
        "classes": n_classes,
        "class_imbalance": class_imbalance,
        "missing_values_count": missing_values_count,
        "missing_values_ratio": missing_values_ratio,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_type": feature_type,
    }


def main():
    dataset_paths = sorted(glob("data/*.tsv.gz"))

    if not dataset_paths:
        raise FileNotFoundError("No datasets found in data/*.tsv.gz")

    records = []
    for i, path in enumerate(dataset_paths, start=1):
        print(f"[{i}/{len(dataset_paths)}] {os.path.basename(path)}")
        try:
            records.append(summarize_dataset(path))
        except Exception as exc:
            records.append(
                {
                    "dataset": os.path.basename(path),
                    "target_column": "",
                    "rows": "",
                    "features": "",
                    "classes": "",
                    "class_imbalance": "",
                    "missing_values_count": "",
                    "missing_values_ratio": "",
                    "numeric_features": "",
                    "categorical_features": "",
                    "feature_type": "error",
                    "error": str(exc),
                }
            )

    manifest = pd.DataFrame(records)
    manifest = manifest.sort_values(by="dataset").reset_index(drop=True)
    output_path = Path("dataset_manifest.csv")
    temp_output_path = Path("dataset_manifest.tmp.csv")

    manifest.to_csv(temp_output_path, index=False)
    os.replace(temp_output_path, output_path)

    print(f"\nWrote dataset_manifest.csv with {len(manifest)} rows")


if __name__ == "__main__":
    main()
