"""Apply standardized MLP notebooks to data/Bottom_Datasets_mlp.

Each generated notebook uses stratified 10-fold CV with an inner validation split,
checkpointing, and explicit diagnostics in separate cells.
"""

import json
import pathlib

BASE = pathlib.Path(r"c:\Users\EYONGEBOB\Desktop\PMLB Project\sklearn-benchmarks")
BOTTOM = BASE / "data" / "Bottom_Datasets_mlp"


def save_json(path: pathlib.Path, obj):
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def build_notebook(dataset_file: str):
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    f"# {dataset_file} - MLP Classification Lab (10-Fold CV)",
                    "",
                    "This notebook uses stratified 10-fold CV with inner train/validation splitting and checkpoint-based model selection.",
                ],
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 0. Imports"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 1. User settings"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": f"""DATASET_FILE = \"{dataset_file}\"

RANDOM_SEED = 90483257
CV_N_SPLITS = 10
CV_SEED = 90483257
VAL_SIZE = 0.2

HIDDEN_NEURONS = 64
HIDDEN_ACTIVATION = \"gelu\"
EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 0.00025

CHECKPOINT_ROOT = Path(\"checkpoints\")
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)""".split("\n"),
            },
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": ["## 2. Reproducibility and device setup"],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 3. Load dataset and inspect metadata"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """def resolve_dataset_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p

    candidate = Path("results") / "MLP Testing Final" / "Bottom_Datasets" / Path(path).stem.replace(".tsv", "") / Path(path).name
    if candidate.exists():
        return candidate

    search_root = Path("results") / "MLP Testing Final" / "Bottom_Datasets"
    matches = list(search_root.glob(f"*/{Path(path).name}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not resolve dataset path: {path}")


def load_dataset(path: str):
    resolved_path = resolve_dataset_path(path)
    df = pd.read_csv(resolved_path, sep='\\t', compression='infer')

    if 'class' in df.columns:
        target_col = 'class'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]

    feature_cols = [c for c in df.columns if c != target_col]
    X_df = df[feature_cols]
    n_numeric = int(X_df.select_dtypes(include=[np.number]).shape[1])
    n_non_numeric = int(X_df.shape[1] - n_numeric)
    X = X_df.astype(np.float32).values
    y_raw = df[target_col].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_mapping = {str(cls): int(idx) for idx, cls in enumerate(encoder.classes_)}

    return X, y, target_col, encoder, resolved_path, feature_cols, n_numeric, n_non_numeric, class_mapping


X, y, target_col, encoder, resolved_path, feature_cols, n_numeric, n_non_numeric, class_mapping = load_dataset(DATASET_FILE)
num_features = X.shape[1]
num_classes = len(np.unique(y))
INPUT_NEURONS = int(num_features)
OUTPUT_NEURONS = 1 if num_classes == 2 else int(num_classes)

print(f"Dataset: {DATASET_FILE}")
print(f"Resolved path: {resolved_path}")
print(f"Target column: {target_col}")
print(f"Samples: {X.shape[0]}, Features: {num_features}, Classes: {num_classes}")
print(f"Feature types: numeric={n_numeric}, non_numeric={n_non_numeric}")
print(f"Model input/output (dataset-dependent): INPUT_NEURONS={INPUT_NEURONS}, OUTPUT_NEURONS={OUTPUT_NEURONS}")
print("Target encoding (LabelEncoder mapping):")
print(class_mapping)""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 4. Define the MLP model"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """def get_activation(name: str):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    if name == 'gelu':
        return nn.GELU()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'sigmoid':
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


class TabularMLP(nn.Module):
    def __init__(self, in_features: int, hidden_neurons: int, out_features: int, activation: str):
        super().__init__()
        act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_neurons),
            act,
            nn.Linear(hidden_neurons, hidden_neurons),
            act,
            nn.Linear(hidden_neurons, out_features),
        )

    def forward(self, x):
        return self.net(x)""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 5. Train/validation/test split and scaling"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """def prepare_fold_data(X_train_full, y_train_full, X_test, y_test, seed: int):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    binary = num_classes == 2
    if binary:
        y_train_t = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=device).view(-1, 1)
        y_val_t = torch.tensor(y_val.astype(np.float32), dtype=torch.float32, device=device).view(-1, 1)
        y_test_t = torch.tensor(y_test.astype(np.float32), dtype=torch.float32, device=device).view(-1, 1)
        out_dim = 1
    else:
        y_train_t = torch.tensor(y_train.astype(np.int64), dtype=torch.long, device=device)
        y_val_t = torch.tensor(y_val.astype(np.int64), dtype=torch.long, device=device)
        y_test_t = torch.tensor(y_test.astype(np.int64), dtype=torch.long, device=device)
        out_dim = num_classes

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    return {
        'X_train_t': X_train_t,
        'y_train_t': y_train_t,
        'X_val_t': X_val_t,
        'y_val_t': y_val_t,
        'X_test_t': X_test_t,
        'y_test_t': y_test_t,
        'y_val_np': y_val.astype(int),
        'y_test_np': y_test.astype(int),
        'train_loader': train_loader,
        'binary': binary,
        'out_dim': out_dim,
        'scaler': scaler,
    }""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 6. Evaluation helper"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """def evaluate_split(model, X_t, y_t, y_true_np, binary: bool):
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        if binary:
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            y_pred = (probs >= 0.5).astype(int)
            confidence = np.where(y_pred == 1, probs, 1.0 - probs)
            prob_second = probs
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(probs, axis=1).astype(int)
            confidence = probs[np.arange(len(y_pred)), y_pred]
            prob_second = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    acc = accuracy_score(y_true_np, y_pred)
    bal_acc = balanced_accuracy_score(y_true_np, y_pred)
    macro_f1 = f1_score(y_true_np, y_pred, average='macro', zero_division=0)

    return {
        'y_pred': y_pred,
        'confidence': confidence,
        'prob_second': prob_second,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
    }""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 7. Checkpoint helper"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """def checkpoint_path_for_fold(fold_idx: int) -> Path:
    return CHECKPOINT_ROOT / f"fold_{fold_idx:02d}_best.pt"


def save_checkpoint(model, fold_idx: int, epoch: int, val_bal_acc: float):
    path = checkpoint_path_for_fold(fold_idx)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'fold': fold_idx,
            'epoch': epoch,
            'val_balanced_accuracy': float(val_bal_acc),
        },
        path,
    )
    return path""".split("\n"),
            },
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": ["## 8. Validation monitoring + checkpointing loop"],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """def train_one_fold_with_validation(fold_idx, X_train_full, y_train_full, X_test, y_test):
    fold_data = prepare_fold_data(
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        seed=RANDOM_SEED + fold_idx,
    )

    model = TabularMLP(num_features, HIDDEN_NEURONS, fold_data['out_dim'], HIDDEN_ACTIVATION).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() if fold_data['binary'] else nn.CrossEntropyLoss()

    best_val_bal_acc = -1.0
    best_checkpoint = None
    training_rows = []
    checkpoint_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in fold_data['train_loader']:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        avg_train_loss = running_loss / len(fold_data['train_loader'].dataset)

        train_eval = evaluate_split(
            model,
            fold_data['X_train_t'],
            fold_data['y_train_t'],
            fold_data['y_train_t'].detach().cpu().numpy().reshape(-1).astype(int),
            fold_data['binary'],
        )
        val_eval = evaluate_split(
            model,
            fold_data['X_val_t'],
            fold_data['y_val_t'],
            fold_data['y_val_np'],
            fold_data['binary'],
        )

        saved = False
        if val_eval['balanced_accuracy'] > best_val_bal_acc:
            best_val_bal_acc = val_eval['balanced_accuracy']
            best_checkpoint = save_checkpoint(model, fold_idx, epoch, best_val_bal_acc)
            checkpoint_rows.append(
                {
                    'fold': fold_idx,
                    'epoch': epoch,
                    'val_balanced_accuracy': float(best_val_bal_acc),
                    'checkpoint_path': str(best_checkpoint),
                }
            )
            saved = True

        training_rows.append(
            {
                'fold': fold_idx,
                'epoch': epoch,
                'train_loss': float(avg_train_loss),
                'train_accuracy': float(train_eval['accuracy']),
                'train_balanced_accuracy': float(train_eval['balanced_accuracy']),
                'val_accuracy': float(val_eval['accuracy']),
                'val_balanced_accuracy': float(val_eval['balanced_accuracy']),
                'saved_checkpoint': bool(saved),
            }
        )

        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            print(
                f"[Fold {fold_idx:02d}] Epoch {epoch:4d}/{EPOCHS} | "
                f"loss={avg_train_loss:.6f} | train_bal_acc={train_eval['balanced_accuracy']:.4f} | "
                f"val_bal_acc={val_eval['balanced_accuracy']:.4f}"
            )

    checkpoint_data = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])

    test_eval = evaluate_split(
        model,
        fold_data['X_test_t'],
        fold_data['y_test_t'],
        fold_data['y_test_np'],
        fold_data['binary'],
    )

    fold_summary = {
        'fold': fold_idx,
        'accuracy': float(test_eval['accuracy']),
        'balanced_accuracy': float(test_eval['balanced_accuracy']),
        'macro_f1': float(test_eval['macro_f1']),
        'train_size': int(len(X_train_full)),
        'test_size': int(len(X_test)),
        'best_checkpoint_path': str(best_checkpoint),
        'best_checkpoint_val_bal_acc': float(best_val_bal_acc),
    }

    return {
        'training_rows': training_rows,
        'checkpoint_rows': checkpoint_rows,
        'fold_summary': fold_summary,
        'y_test_true': fold_data['y_test_np'],
        'y_test_pred': test_eval['y_pred'],
        'y_test_confidence': test_eval['confidence'],
        'y_test_prob_second': test_eval['prob_second'],
        'best_checkpoint_path': str(best_checkpoint),
        'best_checkpoint_val_bal_acc': float(best_val_bal_acc),
        'X_test_t': fold_data['X_test_t'],
        'y_test_t': fold_data['y_test_t'],
        'y_test_np': fold_data['y_test_np'],
        'binary': fold_data['binary'],
    }


cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_SEED)

oof_true = []
oof_pred = []
oof_confidence = []
oof_prob_second = []
fold_metrics = []
training_history_rows = []
checkpoint_history_rows = []

best_overall_checkpoint_path = None
best_overall_val_bal_acc = -1.0
best_overall_fold = None
best_overall_test_bundle = None

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    fold_out = train_one_fold_with_validation(
        fold_idx,
        X_train_full,
        y_train_full,
        X_test,
        y_test,
    )

    oof_true.extend(fold_out['y_test_true'].tolist())
    oof_pred.extend(fold_out['y_test_pred'].tolist())
    oof_confidence.extend(fold_out['y_test_confidence'].tolist())
    oof_prob_second.extend(fold_out['y_test_prob_second'].tolist())

    fold_metrics.append(fold_out['fold_summary'])
    training_history_rows.extend(fold_out['training_rows'])
    checkpoint_history_rows.extend(fold_out['checkpoint_rows'])

    if fold_out['best_checkpoint_val_bal_acc'] > best_overall_val_bal_acc:
        best_overall_val_bal_acc = fold_out['best_checkpoint_val_bal_acc']
        best_overall_checkpoint_path = fold_out['best_checkpoint_path']
        best_overall_fold = fold_idx
        best_overall_test_bundle = {
            'X_test_t': fold_out['X_test_t'],
            'y_test_t': fold_out['y_test_t'],
            'y_test_np': fold_out['y_test_np'],
            'binary': fold_out['binary'],
        }

    print(
        f"Fold {fold_idx:02d}: acc={fold_out['fold_summary']['accuracy']:.4f}, "
        f"bal_acc={fold_out['fold_summary']['balanced_accuracy']:.4f}, "
        f"macro_f1={fold_out['fold_summary']['macro_f1']:.4f}, "
        f"best_val_bal_acc={fold_out['best_checkpoint_val_bal_acc']:.4f}"
    )

oof_true = np.array(oof_true, dtype=int)
oof_pred = np.array(oof_pred, dtype=int)
oof_confidence = np.array(oof_confidence, dtype=float)
oof_prob_second = np.array(oof_prob_second, dtype=float)

fold_df = pd.DataFrame(fold_metrics)
training_history_df = pd.DataFrame(training_history_rows)
checkpoint_history_df = pd.DataFrame(checkpoint_history_rows)

cv_accuracy = accuracy_score(oof_true, oof_pred)
cv_balanced_accuracy = balanced_accuracy_score(oof_true, oof_pred)
cv_macro_f1 = f1_score(oof_true, oof_pred, average='macro', zero_division=0)

summary = {
    'dataset_file': DATASET_FILE,
    'samples': int(X.shape[0]),
    'features': int(num_features),
    'classes': int(num_classes),
    'cv_splits': int(CV_N_SPLITS),
    'seed': int(RANDOM_SEED),
    'oof_accuracy': float(cv_accuracy),
    'oof_balanced_accuracy': float(cv_balanced_accuracy),
    'oof_macro_f1': float(cv_macro_f1),
    'fold_accuracy_mean': float(fold_df['accuracy'].mean()),
    'fold_accuracy_std': float(fold_df['accuracy'].std(ddof=0)),
    'fold_balanced_accuracy_mean': float(fold_df['balanced_accuracy'].mean()),
    'fold_balanced_accuracy_std': float(fold_df['balanced_accuracy'].std(ddof=0)),
    'fold_macro_f1_mean': float(fold_df['macro_f1'].mean()),
    'fold_macro_f1_std': float(fold_df['macro_f1'].std(ddof=0)),
    'best_overall_checkpoint_path': str(best_overall_checkpoint_path),
    'best_overall_checkpoint_fold': int(best_overall_fold),
    'best_overall_checkpoint_val_bal_acc': float(best_overall_val_bal_acc),
}

pd.DataFrame([summary]).to_csv('cv_summary.csv', index=False)
fold_df.to_csv('cv_fold_metrics.csv', index=False)
training_history_df.to_csv('training_history.csv', index=False)
checkpoint_history_df.to_csv('checkpoint_history.csv', index=False)

print("\nSaved: cv_summary.csv, cv_fold_metrics.csv, training_history.csv, checkpoint_history.csv")
print("FINAL 10-FOLD OOF RESULTS")
print(f"Accuracy: {cv_accuracy:.4f}")
print(f"Balanced Accuracy: {cv_balanced_accuracy:.4f}")
print(f"Macro F1: {cv_macro_f1:.4f}")""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 11. Load best checkpoint after training"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """best_model = TabularMLP(
    num_features,
    HIDDEN_NEURONS,
    1 if num_classes == 2 else num_classes,
    HIDDEN_ACTIVATION,
).to(device)

best_ckpt = torch.load(best_overall_checkpoint_path, map_location=device)
best_model.load_state_dict(best_ckpt['model_state_dict'])
best_model.eval()

print(f"Loaded checkpoint: {best_overall_checkpoint_path}")
print(f"Checkpoint fold: {best_overall_fold}")
print(f"Checkpoint val balanced accuracy: {best_overall_val_bal_acc:.4f}")""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 12. Apply best checkpoint to final test set"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """final_test_eval = evaluate_split(
    best_model,
    best_overall_test_bundle['X_test_t'],
    best_overall_test_bundle['y_test_t'],
    best_overall_test_bundle['y_test_np'],
    best_overall_test_bundle['binary'],
)

y_true_test = best_overall_test_bundle['y_test_np']
y_pred_test = final_test_eval['y_pred']

print("Final test (best-checkpoint fold test split)")
print(f"Accuracy: {final_test_eval['accuracy']:.4f}")
print(f"Balanced Accuracy: {final_test_eval['balanced_accuracy']:.4f}")
print(f"Macro F1: {final_test_eval['macro_f1']:.4f}")
print("\nClassification report:")
print(classification_report(y_true_test, y_pred_test, zero_division=0))""".split("\n"),
            },
            {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": ["## 13. Final-test confusion matrix block"]},
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": """labels = list(range(num_classes))
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
ax.set_title('Final-Test Confusion Matrix (Best Checkpoint)')
plt.tight_layout()
plt.show()

print('Raw confusion matrix:')
print(cm)""".split("\n"),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    folders = sorted([d for d in BOTTOM.iterdir() if d.is_dir()])
    updated = 0
    missing = []

    for ds_dir in folders:
        ds_name = ds_dir.name
        nb_path = ds_dir / f"{ds_name}.ipynb"

        if not nb_path.exists():
            missing.append(ds_name)
            continue

        target_file = f"{ds_name}.tsv.gz"
        nb_new = build_notebook(target_file)
        save_json(nb_path, nb_new)
        updated += 1

    print(f"updated_notebooks={updated}")
    print(f"total_folders={len(folders)}")
    print(f"missing_notebook_files={len(missing)}")
    if missing:
        print("missing_examples=", ", ".join(missing[:10]))


if __name__ == "__main__":
    main()
