"""
Convert MLP notebooks from 70-30 train-test split to 10-fold Stratified Cross-Validation.
Aligns MLP evaluation with LR/RF/XGB protocols using seed 90483257.
"""
import os
import json
import re
import pathlib

BASE = r"c:\Users\EYONGEBOB\Desktop\PMLB Project\sklearn-benchmarks"
TEMPLATE_PATH = os.path.join(BASE, "results/MLP Testing Final/Breast_Cancer/BreastCancer.ipynb")
BOTTOM_DS_PATH = os.path.join(BASE, "results/MLP Testing Final/Bottom_Datasets")

def create_cv_template():
    """Generate a modified template notebook using 10-fold CV instead of 70-30 split."""
    
    nb = json.loads(pathlib.Path(TEMPLATE_PATH).read_text(encoding='utf-8'))
    
    # Update cell 1 (imports) to include StratifiedKFold
    import_cell = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'from sklearn.model_selection import train_test_split' in ''.join(cell['source']):
            import_cell = i
            break
    
    if import_cell is not None:
        old_import = """from sklearn.model_selection import train_test_split"""
        new_import = """from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict"""
        src = ''.join(nb['cells'][import_cell]['source'])
        src = src.replace(old_import, new_import)
        nb['cells'][import_cell]['source'] = src.split('\n')
    
    # Update cell 2 (User Settings) - remove TEST_SIZE, add CV_SEED and update dataset settings
    settings_cell = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'DATASET_FILE' in ''.join(cell['source']) and 'TEST_SIZE' in ''.join(cell['source']):
            settings_cell = i
            break
    
    if settings_cell is not None:
        old_settings = """# =========================
# User-configurable settings
# =========================

DATASET_FILE = "breast_cancer.tsv"

HIDDEN_NEURONS = 24

# Choose one of:
# "relu", "leaky_relu", "gelu", "tanh", "sigmoid"
HIDDEN_ACTIVATION = "gelu"

EPOCHS = 2048
BATCH_SIZE = 16
LEARNING_RATE = 0.00025

RANDOM_SEED = 435342

TEST_SIZE = 0.20"""

        new_settings = """# =========================
# User-configurable settings
# =========================

DATASET_FILE = "breast_cancer.tsv"

HIDDEN_NEURONS = 24

# Choose one of:
# "relu", "leaky_relu", "gelu", "tanh", "sigmoid"
HIDDEN_ACTIVATION = "gelu"

EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 0.00025

RANDOM_SEED = 90483257  # Same seed as LR/RF/XGB for consistency
CV_N_SPLITS = 10  # 10-fold cross-validation
CV_SEED = 90483257  # StratifiedKFold seed (same as LR/RF/XGB)"""

        src = ''.join(nb['cells'][settings_cell]['source'])
        src = src.replace(old_settings, new_settings)
        nb['cells'][settings_cell]['source'] = src.split('\n')
    
    # Now rebuild the train/test logic cell to use CV
    # Find the cell with train_test_split call
    split_cell = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'X_train, X_test, y_train, y_test = train_test_split' in ''.join(cell['source']):
            split_cell = i
            break
    
    if split_cell is not None:
        # Replace the entire train/test split cell with CV setup
        new_cv_setup = """# Initialize 10-fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_SEED)

# We will collect predictions from all folds
all_fold_predictions = []
all_fold_true_labels = []
all_fold_accuracies = []
all_fold_balanced_accuracies = []
all_fold_macro_f1s = []

print(f"Starting 10-fold Cross-Validation with seed {CV_SEED}")
print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {np.bincount(y.astype(int))}")
print("=" * 60)"""
        
        nb['cells'][split_cell]['source'] = new_cv_setup.split('\n')
    
    # The following cells (7-14) need restructuring:
    # - Cell 7 (tensor creation) should be inside fold loop
    # - Cell 11 (training) stays similar
    # - Need to wrap everything in a fold loop
    
    # Find tensor creation cell
    tensor_cell = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'X_train_tensor = torch.tensor' in ''.join(cell['source']):
            tensor_cell = i
            break
    
    if tensor_cell is not None:
        # Replace tensor creation with CV loop
        new_fold_loop = """# Process each fold
fold_num = 0
for train_indices, test_indices in cv.split(X, y):
    fold_num += 1
    print(f"\\nFold {fold_num}/{CV_N_SPLITS}")
    print("-" * 60)
    
    # Split data for this fold
    X_train_fold = X[train_indices]
    X_test_fold = X[test_indices]
    y_train_fold = y[train_indices]
    y_test_fold = y[test_indices]
    
    print(f"  Training samples: {X_train_fold.shape[0]}")
    print(f"  Testing samples:  {X_test_fold.shape[0]}")
    
    # Standardize features using ONLY training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_fold).astype(np.float32)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_fold, dtype=torch.float32).to(device)"""
        
        nb['cells'][tensor_cell]['source'] = new_fold_loop.split('\n')
    
    # Find training loop cell and wrap it in the fold loop (close with de-indent at end)
    train_loop_cell = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'for epoch in range(1, EPOCHS + 1):' in ''.join(cell['source']):
            train_loop_cell = i
            break
    
    if train_loop_cell is not None:
        src = ''.join(nb['cells'][train_loop_cell]['source'])
        # Indent the entire training loop by 4 spaces (one level deeper for fold loop)
        lines = src.split('\n')
        indented_lines = ['    ' + line if line.strip() else line for line in lines]
        nb['cells'][train_loop_cell]['source'] = '\n'.join(indented_lines).split('\n')
    
    # Find evaluation cell and indent it + collect predictions
    eval_cell = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'final_train_acc, final_train_bal_acc' in ''.join(cell['source']):
            eval_cell = i
            break
    
    if eval_cell is not None:
        eval_src = """    # Evaluate on this fold
    final_train_acc, final_train_bal_acc, _, _ = evaluate_model(
        model,
        X_train_tensor,
        y_train_tensor
    )
    
    final_test_acc, final_test_bal_acc, y_true_test, y_pred_test = evaluate_model(
        model,
        X_test_tensor,
        y_test_tensor
    )
    
    # Collect results from this fold
    all_fold_predictions.extend(y_pred_test.cpu().numpy() if hasattr(y_pred_test, 'cpu') else y_pred_test)
    all_fold_true_labels.extend(y_test_tensor.cpu().numpy() if hasattr(y_test_tensor, 'cpu') else y_test_tensor)
    all_fold_accuracies.append(final_test_acc)
    all_fold_balanced_accuracies.append(final_test_bal_acc)
    
    print(f"  Fold {fold_num} test accuracy: {final_test_acc:.4f}")
    print(f"  Fold {fold_num} test balanced accuracy: {final_test_bal_acc:.4f}")"""
        
        nb['cells'][eval_cell]['source'] = eval_src.split('\n')
    
    # Find final results cell and rebuild it to show cross-fold statistics
    final_results_cell = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'print("Final Results")' in ''.join(cell['source']):
            final_results_cell = i
            break
    
    if final_results_cell is not None:
        final_src = """# Compute final CV metrics from all out-of-fold predictions
all_fold_predictions = np.array(all_fold_predictions)
all_fold_true_labels = np.array(all_fold_true_labels)

cv_accuracy = accuracy_score(all_fold_true_labels, all_fold_predictions)
cv_balanced_accuracy = balanced_accuracy_score(all_fold_true_labels, all_fold_predictions)

print("\\n" + "=" * 60)
print("CROSS-VALIDATION FINAL RESULTS")
print("=" * 60)
print(f"Mean test accuracy across folds:           {np.mean(all_fold_accuracies):.4f} (+/- {np.std(all_fold_accuracies):.4f})")
print(f"Mean test balanced accuracy across folds:  {np.mean(all_fold_balanced_accuracies):.4f} (+/- {np.std(all_fold_balanced_accuracies):.4f})")
print()
print(f"Overall CV Accuracy (all out-of-fold predictions):           {cv_accuracy:.4f}")
print(f"Overall CV Balanced Accuracy (all out-of-fold predictions):  {cv_balanced_accuracy:.4f}")
print()
print(f"Fold accuracies: {[f'{a:.4f}' for a in all_fold_accuracies]}")
print("=" * 60)"""
        
        nb['cells'][final_results_cell]['source'] = final_src.split('\n')
    
    return nb

# Generate the CV template
print("Creating 10-fold CV template from BreastCancer.ipynb...")
cv_template = create_cv_template()

# Save as reference template
template_output = os.path.join(BASE, "results/MLP Testing Final/BreastCancer_10fold_template.ipynb")
with open(template_output, 'w', encoding='utf-8') as f:
    json.dump(cv_template, f, indent=1)
print(f"✓ Template saved: {template_output}")

# Now regenerate all 44 bottom-dataset notebooks from this template
manifest_path = os.path.join(BASE, "dataset_manifest.csv")
import pandas as pd
manifest = pd.read_csv(manifest_path)

# Get bottom datasets (use the ones we already generated)
bottom_dirs = [d for d in os.listdir(BOTTOM_DS_PATH) if os.path.isdir(os.path.join(BOTTOM_DS_PATH, d))]
print(f"\nFound {len(bottom_dirs)} bottom-dataset folders to update")

updated_count = 0
for ds_dir in sorted(bottom_dirs)[:5]:  # Test with first 5
    nb_path = os.path.join(BOTTOM_DS_PATH, ds_dir, f"{ds_dir}.ipynb")
    if not os.path.exists(nb_path):
        continue
    
    # Load the template and customize for this dataset
    nb_new = json.loads(json.dumps(cv_template))  # Deep copy
    
    # Find the dataset file settings and update DATASET_FILE
    for i, cell in enumerate(nb_new['cells']):
        if cell['cell_type'] == 'code' and 'DATASET_FILE = ' in ''.join(cell['source']):
            src = ''.join(cell['source'])
            src = re.sub(r'DATASET_FILE = "[^"]*"', f'DATASET_FILE = "dataset.tsv.gz"', src)
            nb_new['cells'][i]['source'] = src.split('\n')
            break
    
    # Save updated notebook
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb_new, f, indent=1)
    
    updated_count += 1
    print(f"  ✓ Updated: {ds_dir}")

print(f"\nTest update complete: {updated_count} notebooks modified (first 5 tested)")
print("Ready to regenerate all 44. Run the full regeneration script to complete all.")
