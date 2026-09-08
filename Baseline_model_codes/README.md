# model_code — how to run classifiers and where outputs go

This folder contains the grid-search and random-search classifier scripts used by the benchmark. The scripts are standalone: they read a gzip TSV dataset, evaluate many parameter combinations with 10-fold CV, and print a tab-separated result line for each combination to stdout.

Key folders
- `model_code/random_search/` — scripts that sample parameter combinations randomly and call `evaluate_model.py`.
- `model_code/grid_search/` — scripts that enumerate parameter grids and call `evaluate_model.py`.

How to run (examples)
- Random-search (example for RandomForest):

```bash
python model_code/random_search/RandomForestClassifier.py data/breast_cancer.tsv.gz 50 1234 
# args: <dataset> <num_param_combinations> <random_seed>
```

- Grid-search (example for RandomForest):

```bash
python model_code/grid_search/RandomForestClassifier.py data/breast_cancer.tsv.gz
# grid scripts typically read hard-coded value lists and print every result to stdout
```

# Output format
- Each script prints one tab-separated line per parameter combination with fields:
  - dataset_name
  - (optional) preprocessor class name (in preprocessing variant)
  - classifier class name
  - parameter string (comma-separated key=val)
  - accuracy
  - macro_f1
  - balanced_accuracy

# Capture outputs to files
- Redirect stdout to a file and collect per-run results:

```bash
python model_code/random_search/RandomForestClassifier.py data/breast_cancer.tsv.gz 50 1234 \
  > results/random_search/randomforest_breast_cancer_seed1234.tsv
```

# Recommended workflow for reproducibility
- Always record: script invoked, full argv, random seed, git commit, and environment. Use the top-level `scripts/` wrappers or `experiments/run_experiment.py` to create a `run_manifest.json` that contains this provenance.
- Store final aggregated CSVs under `results/` and move large per-parameter outputs into `archive/` when not needed.

 # Notes and caveats
- The evaluator swallows exceptions for bad parameter combos to keep long runs alive — inspect logs if a classifier produces no output.
- Datasets must have a `class` or `target` label column and be gzip TSV with `.tsv.gz` extension.

## Important — Which scripts produced the thesis results

The three main classifier result sets used in the thesis were produced by these scripts:

- **Logistic Regression (LR):** `model_code/random_search/LogisticRegression.py` (random search).
- **Random Forest (RF):** `model_code/random_search/RandomForestClassifier.py` (random search).
- **XGBoost (XGB):** `model_code/grid_search/XGBClassifier.py` (exhaustive grid search).

Rationale: LR and RF were run with random search to efficiently sample large or
continuous hyperparameter spaces, while XGB used an exhaustive grid because its
parameter grid was small and full coverage was feasible.

Orchestration: top-level orchestrators such as `scripts/run_ranked_benchmarks.py` and `scripts/generate_evaluation_report.py` invoke those scripts, capture their stdout, and aggregate results into `results/` CSVs (for example `results/Final_Performance_Table.csv`).

# Capture a single-run output and record provenance (recommended):

```bash
python model_code/random_search/RandomForestClassifier.py data/breast_cancer.tsv.gz 50 1234 \
  > results/raw/randomsearch_randomforest_breast_cancer_seed1234.tsv
# then create a run manifest with the experiment runner:
python experiments/run_experiment.py --config experiments/configs/top40_mlp.json
```



