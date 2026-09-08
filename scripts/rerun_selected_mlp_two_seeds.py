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
SELECTED_DIR = ROOT / "results" / "Selected dataset"
TOP_SELECTED_CANDIDATES = [
    SELECTED_DIR / "top_datasets_mlp_best.csv",
    SELECTED_DIR / "top_datasets_mlp.csv",
]
BOTTOM_SELECTED_CANDIDATES = [
    SELECTED_DIR / "bottom_datasets_mlp_best.csv",
    SELECTED_DIR / "bottom_datasets_mlp.csv",
]

TOP_BASE = ROOT / "results" / "MLP Testing Final" / "Top_Datasets"
BOTTOM_BASE = ROOT / "data" / "bottom_datasets"
BOTTOM_ALT_BASE = ROOT / "results" / "MLP Testing Final" / "Bottom_Datasets"

CV_N_SPLITS = 10
METRIC_COLUMNS = ["Balanced_Accuracy", "Accuracy", "Macro F1 Score", "Time(s)"]

# Randomly chosen new seeds for this rerun batch.
DEFAULT_SEEDS = [1685701598, 807355633]

ACTIVATION_MAP = {
    "relu": "relu",
    "tanh": "tanh",
    "sigmoid": "logistic",
    "logistic": "logistic",
    "identity": "identity",
    "gelu": "relu",
    "swish": "relu",
    "softmax": "logistic",
}


def normalize_dataset_name(name: str) -> str:
    s = Path(str(name).strip()).name
    if s.endswith(".tsv.gz"):
        return s[: -len(".tsv.gz")]
    return s


def find_label_column(df: pd.DataFrame) -> str:
    if "class" in df.columns:
        return "class"
    if "target" in df.columns:
        return "target"
    raise ValueError("Could not find a label column named 'class' or 'target'")


def load_selected_datasets() -> tuple[list[str], list[str]]:
    top_path = next((p for p in TOP_SELECTED_CANDIDATES if p.exists()), None)
    bottom_path = next((p for p in BOTTOM_SELECTED_CANDIDATES if p.exists()), None)

    if top_path is None:
        raise FileNotFoundError(
            f"Missing selected top datasets file. Tried: {TOP_SELECTED_CANDIDATES}"
        )
    if bottom_path is None:
        raise FileNotFoundError(
            f"Missing selected bottom datasets file. Tried: {BOTTOM_SELECTED_CANDIDATES}"
        )

    top_df = pd.read_csv(top_path)
    bot_df = pd.read_csv(bottom_path)

    top_col = "Dataset" if "Dataset" in top_df.columns else "dataset" if "dataset" in top_df.columns else None
    bottom_col = "Dataset" if "Dataset" in bot_df.columns else "dataset" if "dataset" in bot_df.columns else None

    if top_col is None:
        raise ValueError(f"Expected 'Dataset' or 'dataset' column in {top_path}")
    if bottom_col is None:
        raise ValueError(f"Expected 'Dataset' or 'dataset' column in {bottom_path}")

    top = sorted({normalize_dataset_name(v) for v in top_df[top_col].dropna().astype(str)})
    bottom = sorted({normalize_dataset_name(v) for v in bot_df[bottom_col].dropna().astype(str)})
    return top, bottom


def resolve_top_data_path(dataset: str) -> Path:
    candidates = [
        TOP_BASE / dataset / f"{dataset}.tsv.gz",
        ROOT / "data" / "top_datasets" / f"{dataset}.tsv.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Top dataset file not found for {dataset}: tried {candidates}")


def resolve_bottom_data_path(dataset: str) -> Path:
    candidates = [
        ROOT / "data" / f"{dataset}.tsv.gz",
        BOTTOM_BASE / f"{dataset}.tsv.gz",
        BOTTOM_ALT_BASE / dataset / f"{dataset}.tsv.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Bottom dataset file not found for {dataset}: tried {candidates}")


def resolve_param_csv(dataset_type: str, dataset: str) -> Path:
    if dataset_type == "top":
        candidates = [
            TOP_BASE / dataset / "mlp_hyperparameter_combinations.csv",
        ]
    else:
        candidates = [
            BOTTOM_BASE / dataset / "mlp_hyperparameter_combinations.csv",
            BOTTOM_ALT_BASE / dataset / "mlp_hyperparameter_combinations.csv",
        ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Parameter CSV not found for {dataset_type}:{dataset}: tried {candidates}")


def load_dataset(dataset_type: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset_type == "top":
        data_path = resolve_top_data_path(dataset)
    else:
        data_path = resolve_bottom_data_path(dataset)

    df = pd.read_csv(data_path, compression="gzip", sep="\t")
    label_col = find_label_column(df)
    X = df.drop(columns=[label_col]).values.astype(float)
    y = df[label_col].values
    return X, y


def build_classifier(hidden_neurons: int, activation: str, learning_rate: float, batch_size: int, epochs: int, model_seed: int) -> MLPClassifier:
    act = ACTIVATION_MAP.get(str(activation).strip().lower(), "relu")
    return MLPClassifier(
        hidden_layer_sizes=(hidden_neurons,),
        activation=act,
        solver="adam",
        learning_rate_init=learning_rate,
        batch_size=batch_size,
        max_iter=epochs,
        early_stopping=True,
        random_state=model_seed,
    )


def evaluate_combo(
    X: np.ndarray,
    y: np.ndarray,
    hidden_neurons: int,
    activation: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    cv_seed: int,
    model_seed: int,
    n_jobs: int,
) -> tuple[float, float, float, float]:
    pipeline = make_pipeline(
        RobustScaler(),
        build_classifier(hidden_neurons, activation, learning_rate, batch_size, epochs, model_seed=model_seed),
    )
    cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=cv_seed)
    start = time.perf_counter()
    pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=n_jobs)
    elapsed = time.perf_counter() - start
    ba = balanced_accuracy_score(y, pred)
    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    return ba, acc, macro_f1, elapsed


def evaluate_dataset_for_seed(
    dataset_type: str,
    dataset: str,
    seed: int,
    force: bool,
    n_jobs: int,
) -> Path:
    in_csv = resolve_param_csv(dataset_type, dataset)
    out_csv = in_csv.with_name(f"mlp_hyperparameter_combinations_seed_{seed}.csv")

    if out_csv.exists() and not force:
        out_df = pd.read_csv(out_csv)
        if all(c in out_df.columns for c in METRIC_COLUMNS) and not out_df[METRIC_COLUMNS].isna().any().any():
            print(f"Skipping {dataset_type}:{dataset} seed={seed}; output already complete")
            return out_csv

    # Start from the baseline hyperparameter grid and clear metric columns so
    # each seed run recomputes results instead of reusing prior metrics.
    df = pd.read_csv(in_csv)
    for col in METRIC_COLUMNS:
        df[col] = np.nan

    if out_csv.exists() and not force:
        existing = pd.read_csv(out_csv)
        for col in METRIC_COLUMNS:
            if col not in existing.columns:
                existing[col] = np.nan
        if len(existing) == len(df):
            df = existing

    X, y = load_dataset(dataset_type, dataset)
    print(f"Evaluating {dataset_type}:{dataset} for seed={seed} with {len(df)} combinations")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Use a different model initialization seed to avoid identical model starts.
    model_seed = seed + 1

    for idx, row in df.iterrows():
        if not force and not row[METRIC_COLUMNS].isna().any():
            continue
        try:
            ba, acc, macro_f1, elapsed = evaluate_combo(
                X=X,
                y=y,
                hidden_neurons=int(row["Hidden_Neurons"]),
                activation=row["Activation_Function"],
                learning_rate=float(row["Learning_Rate"]),
                batch_size=int(row["Batch_Size"]),
                epochs=int(row["Epochs"]),
                cv_seed=seed,
                model_seed=model_seed,
                n_jobs=n_jobs,
            )
            df.at[idx, "Balanced_Accuracy"] = ba
            df.at[idx, "Accuracy"] = acc
            df.at[idx, "Macro F1 Score"] = macro_f1
            df.at[idx, "Time(s)"] = elapsed
        except Exception as exc:
            print(f"  Failed {dataset_type}:{dataset} seed={seed} combo {idx + 1}/{len(df)}: {exc}")
            df.at[idx, "Balanced_Accuracy"] = np.nan
            df.at[idx, "Accuracy"] = np.nan
            df.at[idx, "Macro F1 Score"] = np.nan
            df.at[idx, "Time(s)"] = np.nan

        if (idx + 1) % 25 == 0:
            df.to_csv(out_csv, index=False)

    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun selected top/bottom dataset MLP hyperparameter evaluations for two new seeds"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=2,
        default=DEFAULT_SEEDS,
        help=f"Exactly two seeds (default: {DEFAULT_SEEDS[0]} {DEFAULT_SEEDS[1]})",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["top", "bottom", "both"],
        default="both",
        help="Which selected dataset group to evaluate",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset names to limit evaluation to a subset",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs passed to cross_val_predict",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all combinations even if seed output CSV exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    top_selected, bottom_selected = load_selected_datasets()

    if args.datasets:
        wanted = {normalize_dataset_name(v) for v in args.datasets}
        top_selected = [d for d in top_selected if d in wanted]
        bottom_selected = [d for d in bottom_selected if d in wanted]

    selected_jobs: list[tuple[str, str]] = []
    if args.dataset_type in {"top", "both"}:
        selected_jobs.extend(("top", d) for d in top_selected)
    if args.dataset_type in {"bottom", "both"}:
        selected_jobs.extend(("bottom", d) for d in bottom_selected)

    print(f"Using seeds: {args.seeds[0]}, {args.seeds[1]}")
    print(f"Top datasets loaded: {len(top_selected)}")
    print(f"Bottom datasets loaded: {len(bottom_selected)}")
    print(f"Selected datasets to process: {len(selected_jobs)}")
    print(f"Top datasets: {sorted(top_selected)}")
    print(f"Bottom datasets: {sorted(bottom_selected)}")

    failed: list[tuple[str, str, int, str]] = []
    for dataset_type, dataset in selected_jobs:
        for seed in args.seeds:
            try:
                evaluate_dataset_for_seed(
                    dataset_type=dataset_type,
                    dataset=dataset,
                    seed=seed,
                    force=args.force,
                    n_jobs=args.n_jobs,
                )
            except Exception as exc:
                msg = str(exc)
                failed.append((dataset_type, dataset, seed, msg))
                print(f"ERROR {dataset_type}:{dataset} seed={seed}: {msg}")

    if failed:
        fail_df = pd.DataFrame(failed, columns=["dataset_type", "dataset", "seed", "error"])
        fail_path = SELECTED_DIR / "selected_seed_rerun_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        print(f"Completed with failures. Details written to {fail_path}")
    else:
        print("Completed successfully for all selected datasets and both seeds.")


if __name__ == "__main__":
    main()
