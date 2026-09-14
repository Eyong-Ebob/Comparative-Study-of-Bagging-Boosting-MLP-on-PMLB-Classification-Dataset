# CHAPTER THREE

## METHODOLOGY

## 3.1 Introduction

This chapter describes the experimental methodology used to compare bagging, boosting, and multi-layer perceptron (MLP) models on diverse PMLB classification datasets. The objective is not only to report raw performance, but to build a controlled pipeline that makes model comparison fair, reproducible, and interpretable across varying dataset conditions (sample size, dimensionality, class structure, and imbalance).

The methodology follows a staged pipeline:

1. Dataset acquisition and profiling.
2. Baseline model experimentation (Logistic Regression, Random Forest, XGBoost).
3. Best-result extraction and manifest construction.
4. Dataset subgrouping for MLP competitiveness analysis.
5. MLP hyperparameter experimentation on top and bottom dataset groups.
6. Aggregation, ranking, and reproducibility validation.

## 3.2 Research Design and Experimental Philosophy

This study uses a quantitative comparative benchmarking design. All models are evaluated under the same cross-validation protocol and metric framework to isolate the effect of model class rather than data split artifacts.

Three principles guided the design:

1. Fairness: each classifier is evaluated with equivalent data exposure and fold assignment.
2. Robustness: performance is measured using out-of-fold predictions over the full dataset.
3. Reproducibility: random seeds and pipeline constants are fixed and documented.

The experimentation is fully script-driven to reduce manual intervention and improve repeatability.

## 3.3 Data Source and Dataset Construction

The datasets are sourced from PMLB tabular classification collections and stored as compressed TSV files. A manifest-building stage parses each dataset and records structural metadata used later for stratified analysis and subgroup selection.

For each dataset, the pipeline extracts:

1. Number of rows (samples).
2. Number of features.
3. Number of classes.
4. Class imbalance score.
5. Missing-value count and ratio.
6. Feature-type counts (numeric vs categorical).

The label column is detected as class, target, or fallback to the last column when needed. Features are cast to numeric arrays for downstream compatibility.

At the time of this experiment snapshot, the consolidated manifest contains 130 datasets.

## 3.4 Baseline Model Evaluation Pipeline

### 3.4.1 Baseline models

The baseline comparative family includes:

1. Logistic Regression (linear baseline).
2. Random Forest (bagging family).
3. XGBoost (boosting family).

These provide complementary inductive biases and form the reference against which MLP competitiveness is assessed.

### 3.4.2 Preprocessing and pipeline encapsulation

Each baseline configuration is evaluated inside a scikit-learn pipeline with RobustScaler followed by the classifier. Embedding scaling inside the cross-validation pipeline ensures preprocessing is re-fitted only on each training fold, preventing leakage from test folds.

### 3.4.3 Hyperparameter search protocol

Hyperparameters are generated via scripted search definitions:

1. Logistic Regression: random sampling of C, penalty, fit_intercept, and dual (with validity guards), using liblinear solver.
2. Random Forest: random sampling of n_estimators, min_impurity_decrease, max_features, criterion, and max_depth.
3. XGBoost: random sampling over n_estimators, learning_rate, gamma, max_depth, and subsample.

Invalid or unstable combinations are skipped through exception handling so large-scale benchmark runs can continue without full-process failure.

### 3.4.4 Cross-validation strategy

All baseline runs use the same stratified 10-fold cross-validation protocol:

1. n_splits = 10
2. shuffle = True
3. random_state = 90483257

Stratification preserves class proportions per fold, which is critical for imbalanced datasets. The pipeline uses cross_val_predict rather than fold-score averaging, producing one out-of-fold prediction for every sample. This supports whole-dataset metric computation on predictions where each sample is tested exactly once.

### 3.4.5 Metric computation

Three metrics are computed per evaluated configuration:

1. Accuracy.
2. Macro F1.
3. Balanced Accuracy.

Macro F1 is computed as:

$$
\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} \frac{2\,\text{Precision}_c\,\text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
$$

Balanced Accuracy follows the TPOT one-vs-rest formulation used in the codebase:

$$
\text{BalancedAccuracy} = \frac{1}{|C|} \sum_{c \in C} \frac{\text{Sensitivity}_c + \text{Specificity}_c}{2}
$$

This metric is emphasized because it better reflects minority-class performance than plain accuracy.

### 3.4.6 Best-configuration extraction

Raw outputs contain multiple rows per dataset and classifier (one per hyperparameter configuration). A post-processing stage reduces these to one best row per dataset per classifier, using highest accuracy as the selection key. The associated Macro F1 and Balanced Accuracy from that same best-accuracy row are retained.

This yields classifier coverage statistics in the manifest snapshot:

1. Logistic Regression valid on 128 datasets.
2. Random Forest valid on 125 datasets.
3. XGBoost valid on 99 datasets.
4. Datasets with valid results from all three models: 97.

## 3.5 Manifest Enrichment and Dataset Grouping

The manifest is enriched with multi-model summary fields, including number of successful models, model-wise mean scores, and disagreement terms. These support dataset selection for deeper MLP analysis.

In this project state, dataset groups used for MLP experimentation are:

1. Top dataset group: 48 datasets.
2. Bottom dataset group: 44 datasets.

These groups are used to study where MLP behaves competitively versus where it underperforms under the same evaluation mechanics.

## 3.6 MLP Experimentation Pipeline

### 3.6.1 Model and preprocessing

MLP experiments use MLPClassifier wrapped in a RobustScaler pipeline, matching the baseline evaluation style and leakage controls.

Core fixed settings include:

1. Solver: Adam.
2. Early stopping: enabled.
3. Model random_state: fixed for reproducibility.

Activation labels from spreadsheet inputs are mapped to sklearn-compatible activations where necessary (for example, Sigmoid to logistic, and unsupported labels mapped safely).

### 3.6.2 MLP hyperparameter search space

The shared MLP hyperparameter table contains 2,500 combinations, generated from the Cartesian product of:

1. Hidden neurons: {8, 16, 32, 64, 128}
2. Learning rate: {0.0001, 0.001, 0.005, 0.01}
3. Batch size: {8, 16, 32, 64, 128}
4. Activation function labels: {ReLU, Tanh, Sigmoid, GELU, Softmax}
5. Epochs: {50, 100, 300, 500, 1000}

For each dataset, every combination is evaluated and results are incrementally checkpointed to CSV to support recovery from interruptions.

### 3.6.3 MLP evaluation protocol

Each hyperparameter combination is evaluated using the same cross-validation setting as the baselines:

1. StratifiedKFold with 10 folds.
2. Shuffling enabled.
3. CV seed fixed at 90483257 (or alternate seeds in seed-sensitivity reruns).
4. Out-of-fold prediction generation via cross_val_predict.

Per combination, the pipeline records:

1. Balanced Accuracy.
2. Accuracy.
3. Macro F1 Score.
4. Execution Time in seconds.

### 3.6.4 Orchestration for top and bottom groups

The workflow supports:

1. Full-force reruns.
2. Dataset-subset execution from list files.
3. Sequential bottom-group processing for stability.
4. Resume behavior (skip complete outputs, fill only missing evaluations).

This ensures the full search remains tractable even when evaluating many datasets with thousands of combinations each.

### 3.6.5 Multi-seed robustness reruns

An additional script reruns selected datasets using two new random seeds. This sensitivity analysis evaluates whether top-performing MLP settings remain stable under different fold assignments and model initializations, reducing over-reliance on single-seed conclusions.

## 3.7 Result Aggregation and Output Artifacts

After per-dataset evaluation, outputs are aggregated into central files for analysis and reporting:

1. Per-dataset hyperparameter result tables.
2. Group-level ranked outputs.
3. Consolidated workbook sheets for bottom-dataset runs.
4. Summary manifests for completed and failed tasks.

Aggregation scripts sanitize dataset names for spreadsheet constraints and continuously watch for new completed results, enabling incremental consolidation during long runs.

## 3.8 Reproducibility and Quality Controls

The pipeline includes multiple reproducibility safeguards:

1. Fixed global CV seed for fold consistency.
2. Fixed model seeds for comparable stochastic behavior.
3. Deterministic dataset loading and sorting behavior.
4. Explicit logging of skipped, failed, or timed-out runs.
5. Intermediate checkpoint writes during long hyperparameter loops.

Leakage prevention controls include:

1. Scaling inside fold-specific training pipelines.
2. Strict train-test separation by cross_val_predict.
3. Whole-dataset out-of-fold metric computation.

## 3.9 Algorithmic Summary of the End-to-End Pipeline

The complete process can be summarized as follows:

1. Load all PMLB datasets and compute structural metadata.
2. Run baseline LR, RF, and XGB searches with pipeline-based CV evaluation.
3. Compute Accuracy, Macro F1, and Balanced Accuracy per configuration.
4. Select best baseline configuration per dataset-model pair.
5. Merge best baseline outputs into a central dataset manifest.
6. Split datasets into top and bottom groups for MLP competitiveness testing.
7. For each dataset in each group, evaluate all 2,500 MLP combinations with 10-fold stratified CV.
8. Store metrics and runtime per combination; checkpoint regularly.
9. Aggregate per-dataset outputs into group-level comparative artifacts.
10. Re-run selected datasets with additional seeds for robustness checks.
11. Produce ranked and consolidated outputs for downstream statistical analysis and dissertation discussion.

## 3.10 Statistical Analysis Plan

To answer the research question in a defensible way, analysis is performed at dataset level using paired comparisons between model families under identical folds and metrics.

### 3.10.1 Primary endpoint

The primary endpoint is Balanced Accuracy, evaluated out-of-fold for each dataset and model. For any model comparison A versus B on dataset i, the paired performance difference is:

$$
d_i = BA_{i,A} - BA_{i,B}
$$

where positive values indicate superiority of model A on dataset i.

### 3.10.2 Secondary endpoints

Secondary endpoints are Accuracy and Macro F1, used to verify whether conclusions from Balanced Accuracy remain directionally consistent.

### 3.10.3 Inferential testing

For paired model comparisons across datasets, the Wilcoxon signed-rank test is recommended because normality of performance differences is not guaranteed. A significance level of $\alpha = 0.05$ is used.

When performing multiple pairwise comparisons (for example MLP vs RF, MLP vs XGB, and MLP vs LR), p-values should be adjusted using Holm-Bonferroni control.

### 3.10.4 Effect size reporting

Statistical significance is complemented with practical magnitude reporting:

1. Median paired difference in Balanced Accuracy.
2. Win rate, tie rate, and loss rate across datasets.
3. Interquartile range of paired differences.

This prevents over-reliance on p-values and provides interpretable competitiveness statements.

### 3.10.5 Condition-based subgroup analysis

To align with the dissertation objective, paired differences are additionally stratified by dataset conditions stored in the manifest, such as:

1. Size bins.
2. Dimensionality bins.
3. Class-count bins.
4. Imbalance bins.

This supports statements of the form: MLP is competitive under specific conditions rather than universally superior.

### 3.10.6 Robustness checks

The two-seed rerun procedure is used as a stability check. A result is considered robust when directionality of paired differences is preserved across both seeds and confidence summaries remain close.

## 3.11 Methodological Limitations

The current methodology is strong for controlled benchmarking, but several limitations remain:

1. Best baseline extraction is keyed by accuracy, which may not always align with balanced-accuracy-optimal settings on highly imbalanced data.
2. Large hyperparameter spaces can induce selection optimism (best-of-many effect).
3. Some model-dataset combinations fail due to algorithm constraints or runtime limits, reducing complete overlap across classifiers.
4. The chapter focuses on internal benchmark comparison and does not claim out-of-distribution generalization beyond studied PMLB tasks.

These limitations are acknowledged and partially mitigated through stratified CV, multi-metric reporting, and seed-based reruns.

## 3.12 Conclusion

This methodology establishes a reproducible and leakage-aware experimental pipeline for comparing bagging, boosting, and MLP models across diverse tabular classification datasets. By combining standardized cross-validation, broad hyperparameter exploration, and structured manifest-driven aggregation, the pipeline enables condition-based analysis of where MLP is competitive relative to strong tree-ensemble baselines.
