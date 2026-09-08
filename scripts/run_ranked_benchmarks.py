from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RANDOM_SEARCH_DIR = ROOT / "Baseline_model_code" / "random_search"
RESULTS_DIR = ROOT / "results"

CLASSIFIERS = [
    "XGBClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
]


def run_classifier_on_dataset(classifier: str, dataset_path: Path, num_combinations: int, seed: int, timeout_seconds: int) -> dict | None:
    command = [
        sys.executable,
        f"{classifier}.py",
        str(dataset_path),
        str(num_combinations),
        str(seed),
    ]
    completed = subprocess.run(
        command,
        cwd=RANDOM_SEARCH_DIR,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout_seconds,
    )

    if completed.returncode != 0 and not completed.stdout.strip():
        return None

    rows = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 6:
            continue
        dataset_name, classifier_name, params, accuracy, macro_f1, balanced_accuracy = parts
        try:
            rows.append(
                {
                    "dataset": dataset_name,
                    "classifier": classifier_name,
                    "params": params,
                    "accuracy": float(accuracy),
                    "macro_f1": float(macro_f1),
                    "balanced_accuracy": float(balanced_accuracy),
                }
            )
        except ValueError:
            continue

    if not rows:
        return None

    best_row = max(rows, key=lambda row: row["accuracy"])
    best_row["source_file"] = dataset_path.name
    return best_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run three classifiers sequentially and rank datasets by best accuracy.")
    parser.add_argument("--num-combinations", type=int, default=5, help="Random search combinations per dataset per classifier.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of datasets to run.")
    parser.add_argument("--per-run-timeout", type=int, default=900, help="Timeout in seconds for one classifier on one dataset.")
    args = parser.parse_args()

    dataset_paths = sorted(DATA_DIR.glob("*.tsv.gz"))
    if args.limit is not None:
        dataset_paths = dataset_paths[: args.limit]

    if not dataset_paths:
        print("No datasets found in data/", file=sys.stderr)
        return 1

    RESULTS_DIR.mkdir(exist_ok=True)

    for classifier_index, classifier in enumerate(CLASSIFIERS):
        print(f"Running {classifier} on {len(dataset_paths)} datasets...")
        records = []
        skipped_datasets = []
        for dataset_index, dataset_path in enumerate(dataset_paths, start=1):
            print(f"  [{dataset_index}/{len(dataset_paths)}] {dataset_path.name}", flush=True)
            try:
                result = run_classifier_on_dataset(
                    classifier=classifier,
                    dataset_path=dataset_path,
                    num_combinations=args.num_combinations,
                    seed=args.seed + classifier_index,
                    timeout_seconds=args.per_run_timeout,
                )
            except subprocess.TimeoutExpired:
                skipped_datasets.append({"dataset": dataset_path.name, "reason": "timeout"})
                continue
            if result is not None:
                records.append(result)
            else:
                skipped_datasets.append({"dataset": dataset_path.name, "reason": "no_valid_result"})

        ranked = pd.DataFrame(records)
        if ranked.empty:
            print(f"No successful runs for {classifier}", file=sys.stderr)
            continue

        ranked = ranked.sort_values(["accuracy", "balanced_accuracy", "macro_f1"], ascending=[False, False, False]).reset_index(drop=True)
        ranked.insert(0, "rank", ranked.index + 1)

        ranked_path = RESULTS_DIR / f"{classifier}_ranked_by_accuracy.csv"
        ranked.to_csv(ranked_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved ranked results to {ranked_path}")

        if skipped_datasets:
            skipped_path = RESULTS_DIR / f"{classifier}_skipped_datasets.csv"
            pd.DataFrame(skipped_datasets).to_csv(skipped_path, index=False)
            print(f"Saved skipped dataset log to {skipped_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())