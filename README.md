# Comparative Study of Bagging, Boosting, and MLPs on PMLB Classification Datasets

This repository reproduces the end-to-end experimental workflow of the M.Sc research project. The goal is to evaluate when multilayer perceptrons (MLPs) are competitive with tree-based ensemble methods on tabular classification datasets from PMLB.

The study compares four model families:
- Logistic Regression
- Random Forest
- XGBoost
- MLPClassifier

The project is designed to answer a specific research question:

> Under which dataset conditions do MLPs compete with or outperform bagging and boosting methods?

This is not a generic benchmarking repository. It follows the thesis workflow: dataset profiling, baseline benchmarking, dataset subgroup selection, MLP evaluation, and final comparative analysis.

---

## 1. Research objective

The thesis focuses on model competitiveness under controlled dataset conditions rather than raw leaderboard performance.

The analysis asks whether MLPs are competitive when datasets are:
- larger
- higher-dimensional
- more numeric than categorical
- less imbalanced
- more linearly separable
- more complex in class structure

The final comparison is based on paired dataset-level evaluation, with Balanced Accuracy as the primary metric.

---

## 2. Thesis workflow implemented in code

The repository follows the exact research pipeline:

1. Download and profile PMLB classification datasets
2. Build a dataset manifest with structural metadata
3. Evaluate baseline models (Logistic Regression, Random Forest, XGBoost)
4. Extract best-performing configuration per dataset and classifier
5. Select representative top and bottom dataset groups for MLP analysis
6. Run large-scale MLP hyperparameter search
7. Aggregate outputs and compare MLP against baseline families
8. Validate results and generate final reporting artifacts

This is the implemented logic behind the thesis methodology.

---

## 3. Project structure

```text
.
├── 00_data/                     # raw data and curated dataset folders
├── 01_baseline_search/          # baseline search scripts for LR, RF, XGB
├── 02_baseline_results/         # baseline benchmark outputs and rankings
├── 03_selected_subset/          # selected dataset groups for deeper analysis
├── 04_mlp_search/               # MLP hyperparameter search scripts
├── 05_analysis/                 # comparison, validation, and aggregation scripts
├── 06_legacy/                   # historical or superseded code
├── 07_project_docs/             # thesis chapters and methodology notes
├── 08_results/                  # final tables and dataset-level outputs
├── metafeatures/                # dataset feature extraction utilities
├── notebooks/                   # exploratory and analysis notebooks
├── scripts/                     # workflow support scripts
├── requirements.txt             # project dependencies
├── LICENSE                      # project license
├── README.md                    # main project documentation
├── _report_figs/                # generated figures and reports
├── archive/                     # archived materials
└── .gitignore
```

---

## 4. Requirements

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Required packages include:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- xgboost
- pmlb
- torch
- jupyter

---

## 5. Reproducible execution workflow

Use the scripts in the order below to reproduce the thesis pipeline.

### Step 1: Prepare the dataset pool

The project assumes the PMLB dataset repository is available and accessible through the PMLB package.

Use the dataset loading and metadata scripts in the project workflow to build the dataset manifest and characterize each dataset.

Relevant project code:
- `metafeatures/`
- `scripts/download_datasets.py`
- `00_data/`

Expected output:
- dataset manifest containing metadata such as rows, features, classes, imbalance, and feature type

---

### Step 2: Run baseline model searches

Baseline experimentation is implemented in:
- `01_baseline_search/LogisticRegression.py`
- `01_baseline_search/RandomForestClassifier.py`
- `01_baseline_search/XGBClassifier.py`
- `01_baseline_search/evaluate_model.py`

Run the classifiers in the project’s benchmark workflow to produce baseline results.

Example flow:

```bash
python 01_baseline_search/LogisticRegression.py
python 01_baseline_search/RandomForestClassifier.py
python 01_baseline_search/XGBClassifier.py
```

Expected output:
- per-dataset classifier outputs
- best configuration per dataset-model pair
- baseline leaderboards in `02_baseline_results/`

---

### Step 3: Construct the comparison manifest

After the baseline results are generated, merge the best performance rows into a single dataset manifest that stores:
- model performance metrics
- dataset metadata
- condition assignments
- selection flags

The results are stored under:
- `08_results/dataset_manifest.csv`
- `08_results/Master_Performance_Table.csv`

These files form the basis for selecting the top and bottom dataset groups used in the MLP experiments.

---

### Step 4: Select dataset groups for MLP analysis

The study divides datasets into representative analysis groups based on performance and dataset conditions.

This stage is implemented through the selection and aggregation logic in:
- `03_selected_subset/`
- `02_baseline_results/`
- `08_results/`

The output should include the dataset lists used for:
- top-group MLP evaluation
- bottom-group MLP evaluation

---

### Step 5: Run the MLP hyperparameter search

The MLP search is implemented in:
- `04_mlp_search/evaluate_mlp_hyperparameter_combinations_top.py`
- `04_mlp_search/evaluate_mlp_hyperparameter_combinations_bottom.py`
- `04_mlp_search/mlp_hyperparameter_combinations.csv`

These scripts evaluate a fixed MLP search space across the selected datasets using stratified 10-fold cross-validation.

Run the top-group experiments:

```bash
python 04_mlp_search/evaluate_mlp_hyperparameter_combinations_top.py
```

Run the bottom-group experiments:

```bash
python 04_mlp_search/evaluate_mlp_hyperparameter_combinations_bottom.py
```

Expected outputs:
- per-dataset MLP evaluation files
- Balanced Accuracy, Accuracy, Macro F1, and runtime metrics
- results stored under the dataset-specific folders and summary output tables

---

### Step 6: Aggregate and validate results

Use the analysis stage to combine evaluation outputs and validate consistency.

Relevant scripts:
- `05_analysis/compare_central.py`
- `05_analysis/validate_central.py`
- `05_analysis/generate_evaluation_report.py`

These scripts check that results are correctly consolidated and generate summary files for the final analysis.

Example:

```bash
python 05_analysis/compare_central.py
python 05_analysis/validate_central.py
python 05_analysis/generate_evaluation_report.py
```

Expected output:
- validated comparative tables
- ranked MLP vs baseline comparisons
- final reporting artifacts for the thesis analysis

---

## 6. Main output files

The code generates the following research artifacts:

- `08_results/dataset_manifest.csv`
- `08_results/Master_Performance_Table.csv`
- `08_results/LogisticRegression_ranked_by_balanced_accuracy.csv`
- `08_results/RandomForestClassifier_ranked_by_balanced_accuracy.csv`
- `08_results/XGBClassifier_ranked_by_balanced_accuracy.csv`
- `08_results/mlp_best_results_all_seeds.csv`
- dataset-specific MLP search outputs under `08_results/`

These files are the main evidence used to compare model families and support the thesis conclusions.

---

## 7. Key methodological rules

The project follows strict methodological controls:

- stratified 10-fold cross-validation
- same data splits across model families
- preprocessing inside each fold to prevent leakage
- Balanced Accuracy as the primary performance metric
- Macro F1 and Accuracy as secondary metrics
- model seed control for reproducibility
- checkpointing and validation for large MLP runs

This ensures that comparisons reflect model behavior rather than procedural artifacts.

---

## 8. Reproducibility checklist

To reproduce the thesis workflow from scratch, complete the following sequence:

1. install dependencies
2. load PMLB datasets
3. generate the dataset manifest
4. run baseline searches for LR, RF, and XGB
5. merge best baseline results into the master manifest
6. select the representative dataset groups
7. run MLP hyperparameter search on those groups
8. aggregate results
9. validate the final comparison tables
10. prepare final analysis and thesis report


## 09. License

This project is released under the repository license included in the root `LICENSE` file.

---

## 10. Practical note

This repository is meant to be run as a complete experimental pipeline rather than as isolated notebooks. The code is the implementation of the thesis solution, and the workflow should be followed in the order above to reproduce the research findings.
