from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_mlp_hyperparameter_combinations_bottom import (
    load_bottom_dataset_list,
    load_param_grid,
    process_dataset,
)

CURRENT_PID = None


def pid_exists(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


def wait_for_pid(pid: int, interval: int = 10) -> None:
    print(f"Waiting for PID {pid} to exit...")
    while pid_exists(pid):
        time.sleep(interval)
    print(f"PID {pid} has exited.")


def load_dataset_list(dataset_file: Path | None) -> list[str]:
    if dataset_file:
        if not dataset_file.exists():
            raise FileNotFoundError(f"Datasets file not found: {dataset_file}")
        with dataset_file.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return load_bottom_dataset_list()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bottom dataset MLP evaluations sequentially until complete")
    parser.add_argument("--wait-pid", type=int, help="Wait for this PID to exit before starting")
    parser.add_argument("--datasets-file", type=Path, help="Optional text file with dataset names, one per line")
    parser.add_argument("--force", action="store_true", help="Recompute all combinations even if output exists")
    args = parser.parse_args()

    if args.wait_pid:
        try:
            import psutil
        except ImportError:
            raise RuntimeError("psutil is required for --wait-pid; install it or omit this option")
        wait_for_pid(args.wait_pid)

    param_grid = load_param_grid()
    dataset_names = load_dataset_list(args.datasets_file)

    print(f"Running sequentially on {len(dataset_names)} datasets")

    # Limit BLAS/OMP thread usage to avoid oversubscription; tune n_jobs below.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    DEFAULT_N_JOBS = 4

    for dataset_name in dataset_names:
        print(f"\n=== Processing dataset: {dataset_name} ===")
        try:
            process_dataset(dataset_name, param_grid, force=args.force, n_jobs=DEFAULT_N_JOBS)
        except Exception as exc:
            print(f"Error processing {dataset_name}: {exc}")
            print("Continuing to next dataset.")

    print("All requested bottom datasets have been processed.")


if __name__ == "__main__":
    main()
