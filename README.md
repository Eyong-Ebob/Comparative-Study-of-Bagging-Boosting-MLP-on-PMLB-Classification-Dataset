<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
## Downloading the benchmark data

Please refer to [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks) to gain access to the curated datasets from this study. PMLB provides an easy-to-use Python interface to download the datasets.

## Dataset Evaluation for MLP Competitiveness Study

This iteration of the project focuses on determining **under what dataset conditions MLP classifiers are competitive with tree-based ensemble methods** on public tabular classification datasets.

### Key Question
The dissertation does not aim to beat SOTA. Instead, it asks: **"On which dataset conditions do MLPs compete with or outperform tree ensembles?"**

Not: "Model X got accuracy Y."
But: "Are MLPs competitive when datasets are larger, more continuous, less categorical, more linearly separable, less imbalanced, or higher dimensional?"

### Dataset Selection Strategy

**Purpose**: Maximize diversity across dataset conditions (size, dimensionality, feature type, class balance) while ensuring all datasets are learnable and thoroughly evaluated.

**Selection Process**:
1. Filter to datasets successfully evaluated by all three baseline classifiers (XGBoost, RandomForest, LogisticRegression) → 128 usable datasets
2. Stratify across five condition dimensions:
   - **Size**: small (low, medium, large 
   - **Dimensionality**: low, mid, high
   - **Feature type**: numeric, categorical, mixed
   - **Class imbalance**: balanced, moderate, high
   - **Classes**: binary, multiclass-few, multiclass-many.
3. Within each non-empty stratum, select the dataset with best composite performance score
4. Fill remainder with highest-scoring runners-up
5. Result: datasets** covering condition space

### Column Meanings (dataset_manifest.csv)

#### Dataset Properties (Conditions Under Test)
- `rows`: Training samples. **Why**: MLPs scale better on large data than greedy tree splits.
- `features`: Dimensionality. **Why**: Trees struggle in high-d (curse of dimensionality); MLPs may compete.
- `feature_type` (numeric | categorical | mixed): **Why**: Trees naturally handle categorical; MLPs require encoding (overhead).
- `classes`: Classification complexity. **Why**: Loss scales with log(classes); affects both methods differently.
- `class_imbalance` [0, 1): Squared distance from perfect balance. **Why**: Trees/ensembles handle imbalance well; MLPs require loss weighting/sampling.
- `missing_values_ratio`: **Why**: Trees are inherently robust; MLPs require imputation (overhead).

#### Stratification Bins
- `size_bin`: small | medium | large
- `dim_bin`: low_dim | mid_dim | high_dim
- `imb_bin`: balanced | moderate_imb | high_imb
- `class_bin`: binary | multiclass_few | multiclass_many
- `strata`: Full condition assignment (e.g., `small+low_dim+numeric+balanced+binary`)

#### Baseline Classifier Performance (Fair Comparison)
- `xgb_accuracy`, `rf_accuracy`, `logreg_accuracy`: Best accuracy per baseline
- `xgb_macro_f1`, `rf_macro_f1`, `logreg_macro_f1`: Class-balanced F1 per baseline
- `xgb_balanced_accuracy`, `rf_balanced_accuracy`, `logreg_balanced_accuracy`: Balanced accuracy per baseline
- `n_models_success`: Count of successful classifiers (3 = all; selection requirement)

#### Multi-Model Consensus & Selection
- `mean_accuracy_3models`: Average accuracy benchmark for MLPs to beat
- `mean_macro_f1_3models`, `mean_balanced_accuracy_3models`: Consensus class-balanced metrics
- `std_accuracy_3models`: Model disagreement (low = clean signals; high = method-dependent)
- `cond_selection_score`: Composite score (40% accuracy, 30% consensus, 30% balanced accuracy) within each stratum
- `selection_rank_top40`: Position in final 40 (1-40) for MLP evaluation

#### Justifications & Selection Metadata
- `selection_justification_condition_based`: Natural language reason linking dataset to hypothesis (e.g., "high-dimensional numeric dataset where MLPs may compete with trees")
- `selected_for_mlp_eval`: Boolean flag (True for top 40; False for others)

### Expected Condition Coverage (Top 40)
| Dimension | Expected | Count |
|-----------|----------|-------|
| **Small** (<1k rows) | Overfitting tests | ~13 |
| **Medium** (1k-20k) | Balanced regime | ~18 |
| **Large** (>20k) | Scalability tests | ~9 |
| **Low-dim** (≤10) | Tree advantages | ~11 |
| **Mid-dim** (11-100) | Balanced | ~19 |
| **High-dim** (>100) | NN potential | ~10 |
| **Numeric-only** | NN sweet spot | ~40 |
| **Well-balanced** | Clean signals | ~14 |
| **Moderate imbalance** | Ensemble stress | ~13 |
| **Severe imbalance** | Tree advantage | ~13 |
| **Binary** | Simpler | ~21 |
| **Multiclass** (3-4) | Medium | ~12 |
| **Many-class** (5+) | Complex | ~7 |

### Output Files
- `results/final_40_datasets_for_mlp_condition_based.csv`: Metadata for all selected datasets + condition assignments
- `dataset_manifest.csv`: Full 130-dataset metadata with selection fields; filter via `selected_for_mlp_eval`

### Dissertation Analysis Framework
With this stratified selection, you can answer:
1. **"Are MLPs competitive on large-scale datasets?"** → Compare across `size_bin`
2. **"Do MLPs handle high-dimensional data?"** → Compare across `dim_bin`
3. **"Can MLPs beat trees on numeric data?"** → Subset `feature_type == numeric`
4. **"Does imbalance hurt MLPs more?"** → Compare across `imb_bin`, track metric degradation
5. **"Which conditions favor MLPs vs trees?"** → Stratify by `strata`, compute win rates per condition

### Search spaces used for the baseline classifiers

The parameter ranges below are the actual search spaces used to generate the benchmark results. They are defined in the corresponding scripts under `Baseline_model_codes/` and are therefore the source of truth for the reported experiments.

| Model | Hyperparameter | Search values / sampling rule | Source |
|---|---|---|---|
| Logistic Regression | `C` | `np.random.uniform(low=1e-10, high=10.0, size=num_param_combinations)` | `Baseline_model_codes/random_search/LogisticRegression.py` |
|  | `penalty` | `['l1', 'l2']` sampled randomly | `Baseline_model_codes/random_search/LogisticRegression.py` |
|  | `fit_intercept` | `True` or `False` sampled randomly | `Baseline_model_codes/random_search/LogisticRegression.py` |
|  | `dual` | `False` when `penalty != 'l2'`; otherwise sampled randomly from `True` or `False` | `Baseline_model_codes/random_search/LogisticRegression.py` |
|  | `solver` | fixed at `liblinear` | `Baseline_model_codes/random_search/LogisticRegression.py` |
| Random Forest | `n_estimators` | sampled from `range(50, 1001, 50)` | `Baseline_model_codes/random_search/RandomForestClassifier.py` |
|  | `min_impurity_decrease` | sampled from `np.random.exponential(scale=0.01, size=num_param_combinations)` | `Baseline_model_codes/random_search/RandomForestClassifier.py` |
|  | `max_features` | sampled from `np.arange(0.01, 1.0, 0.01)` plus `['sqrt', 'log2', None]` | `Baseline_model_codes/random_search/RandomForestClassifier.py` |
|  | `criterion` | `['gini', 'entropy']` sampled randomly | `Baseline_model_codes/random_search/RandomForestClassifier.py` |
|  | `max_depth` | sampled from `range(1, 51)` plus `None` | `Baseline_model_codes/random_search/RandomForestClassifier.py` |
| XGBoost | `n_estimators` | `[10, 50, 100, 500]` | `Baseline_model_codes/grid_search/XGBClassifier.py` |
|  | `learning_rate` | `[0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]` | `Baseline_model_codes/grid_search/XGBClassifier.py` |
|  | `gamma` | `np.arange(0.0, 0.51, 0.05)` | `Baseline_model_codes/grid_search/XGBClassifier.py` |
|  | `max_depth` | `[1, 2, 3, 4, 5, 10, 20, 50, None]` | `Baseline_model_codes/grid_search/XGBClassifier.py` |
|  | `subsample` | `np.arange(0.0, 1.01, 0.1)` | `Baseline_model_codes/grid_search/XGBClassifier.py` |

Note: Logistic Regression and Random Forest were evaluated with random search; XGBoost used exhaustive grid search. 



=======
# Comparative-Study-of-Bagging-Boosting-MLP-on-PMLB-Classification-Dataset
>>>>>>> 7990d02227c7a2a676a64334ff3e6d506c8215b5
=======
# Comparative-Study-of-Bagging-Boosting-MLP-on-PMLB-Classification-Dataset
>>>>>>> 7990d02227c7a2a676a64334ff3e6d506c8215b5
=======
# Comparative-Study-of-Bagging-Boosting-MLP-on-PMLB-Classification-Dataset
>>>>>>> 7990d02227c7a2a676a64334ff3e6d506c8215b5
