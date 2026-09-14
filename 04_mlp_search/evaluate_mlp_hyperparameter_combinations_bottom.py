from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data"
PARAM_FILE = ROOT / "results" / "mlp_hyperparameter_combinations.csv"
BOTTOM_CSV = BASE / "bottom_dataset.csv"
BOTTOM_DATASETS_DIR = BASE / "bottom_datasets"
OUTPUT_FILENAME = "mlp_hyperparameter_combinations.csv"
CV_N_SPLITS = 10
CV_RANDOM_STATE = 90483257

ACTIVATION_MAP = {
    "relu": "relu",
    "tanh": "tanh",
    "sigmoid": "logistic",
    "logistic": "logistic",
    "identity": "identity",
    "gelu": "relu",
    "swish": "relu",
}

METRIC_COLUMNS = ["Balanced_Accuracy", "Accuracy", "Macro F1 Score", "Time(s)"]


def load_param_grid() -> pd.DataFrame:
    if not PARAM_FILE.exists():
        raise FileNotFoundError(f"Hyperparameter CSV not found at {PARAM_FILE}")
    df = pd.read_csv(PARAM_FILE)
    if not all(col in df.columns for col in ["Hidden_Neurons", "Learning_Rate", "Batch_Size", "Activation_Function", "Epochs"]):
        raise ValueError("Parameter CSV is missing required hyperparameter columns")
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_bottom_dataset_list() -> list[str]:
    if not BOTTOM_CSV.exists():
        raise FileNotFoundError(f"Bottom dataset manifest not found at {BOTTOM_CSV}")
    bottom = pd.read_csv(BOTTOM_CSV)
    if "dataset" not in bottom.columns:
        raise ValueError("bottom_dataset.csv must contain a 'dataset' column")
    return bottom["dataset"].astype(str).tolist()


def find_label_column(df: pd.DataFrame) -> str:
    if "class" in df.columns:
        return "class"
    if "target" in df.columns:
        return "target"
    raise ValueError("Could not find a label column named 'class' or 'target'")


def dataset_folder(dataset_name: str) -> Path:
    return BOTTOM_DATASETS_DIR / dataset_name.replace(".tsv.gz", "")


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    data_path = BASE / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    df = pd.read_csv(data_path, compression="gzip", sep="\t")
    label_col = find_label_column(df)
    X = df.drop(columns=[label_col]).values.astype(float)
    y = df[label_col].values
    return X, y


def build_classifier(hidden_neurons: int, activation: str, learning_rate: float, batch_size: int, epochs: int) -> MLPClassifier:
    act = ACTIVATION_MAP.get(str(activation).strip().lower(), "relu")
    return MLPClassifier(
        hidden_layer_sizes=(hidden_neurons,),
        activation=act,
        solver="adam",
        learning_rate_init=learning_rate,
        batch_size=batch_size,
        max_iter=epochs,
        early_stopping=True,
        random_state=324089,
    )


def evaluate_combo(X: np.ndarray, y: np.ndarray, hidden_neurons: int, activation: str, learning_rate: float, batch_size: int, epochs: int, n_jobs: int = -1) -> tuple[float, float, float, float]:
    pipeline = make_pipeline(RobustScaler(), build_classifier(hidden_neurons, activation, learning_rate, batch_size, epochs))
    cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    start = time.perf_counter()
    pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=n_jobs)
    elapsed = time.perf_counter() - start
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    ba = balanced_accuracy_score(y, pred)
    return ba, acc, f1, elapsed


def safe_save(out_path: Path, df: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def process_dataset(dataset_name: str, param_grid: pd.DataFrame, force: bool = False, n_jobs: int = -1) -> None:
    folder = dataset_folder(dataset_name)
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / OUTPUT_FILENAME
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if all(col in existing.columns for col in METRIC_COLUMNS) and not existing[METRIC_COLUMNS].isna().any().any() and not force:
            print(f"Skipping {dataset_name}: already fully evaluated")
            return
        df = existing
        # preserve columns and add any missing metric columns
        for col in METRIC_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
    else:
        df = param_grid.copy()

    X, y = load_dataset(dataset_name)
    print(f"Evaluating {dataset_name} ({len(df)} combinations) on {X.shape[0]} rows, {X.shape[1]} features")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for idx, row in df.iterrows():
        if not force and not row[METRIC_COLUMNS].isna().any():
            continue
        try:
            ba, acc, f1, elapsed = evaluate_combo(
                X,
                y,
                int(row["Hidden_Neurons"]),
                row["Activation_Function"],
                float(row["Learning_Rate"]),
                int(row["Batch_Size"]),
                int(row["Epochs"]),
                n_jobs=n_jobs,
            )
            df.at[idx, "Balanced_Accuracy"] = ba
            df.at[idx, "Accuracy"] = acc
            df.at[idx, "Macro F1 Score"] = f1
            df.at[idx, "Time(s)"] = elapsed
        except Exception as exc:
            print(f"  Combo {idx+1}/{len(df)} failed for {dataset_name}: {exc}")
            df.at[idx, "Balanced_Accuracy"] = np.nan
            df.at[idx, "Accuracy"] = np.nan
            df.at[idx, "Macro F1 Score"] = np.nan
            df.at[idx, "Time(s)"] = np.nan
        if (idx + 1) % 25 == 0:
            safe_save(out_path, df)
    safe_save(out_path, df)
    print(f"Finished {dataset_name}: results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MLP hyperparameter combinations on bottom datasets")
    parser.add_argument("--datasets", nargs="*", help="Optional specific dataset names to evaluate")
    parser.add_argument("--datasets-file", type=Path, help="Optional text file with dataset names, one per line")
    parser.add_argument("--exclude", nargs="*", default=[], help="Optional dataset names to skip")
    parser.add_argument("--force", action="store_true", help="Recompute all combinations even if output exists")
    args = parser.parse_args()

    param_grid = load_param_grid()
    bottom_datasets = load_bottom_dataset_list()

    if args.datasets_file:
        if not args.datasets_file.exists():
            raise FileNotFoundError(f"Datasets file not found: {args.datasets_file}")
        with args.datasets_file.open("r", encoding="utf-8") as f:
            file_datasets = [line.strip() for line in f if line.strip()]
        targets = file_datasets
    elif args.datasets:
        targets = args.datasets
    else:
        targets = bottom_datasets

    if args.exclude:
        targets = [ds for ds in targets if ds not in args.exclude]

    missing = [ds for ds in targets if ds not in bottom_datasets]
    if missing:
        raise ValueError(f"Requested datasets not found in bottom_dataset.csv: {missing}")

    for dataset_name in targets:
        process_dataset(dataset_name, param_grid, force=args.force)


if __name__ == "__main__":
    main()
