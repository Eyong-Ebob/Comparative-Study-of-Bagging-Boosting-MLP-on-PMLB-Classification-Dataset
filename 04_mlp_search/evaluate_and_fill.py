from pathlib import Path
import time
import pandas as pd
import shutil
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

BASE = Path('data')
BOTTOM_CSV = BASE / 'bottom_dataset.csv'
MASTER = Path('results') / 'MLP Evaluation sheet(final).xlsx'
OUT_NAME = 'MLP Evaluation sheet(final)_filled.xlsx'
CV_N_SPLITS = 10
CV_RANDOM_STATE = 90483257

# parameter grid inferred from sample filled files
hidden_options = [16, 32, 64, 128]
activations = ['relu']
learning_rates = [0.001]
batch_sizes = [16]
epochs_list = [100]

bottom = pd.read_csv(BOTTOM_CSV)
missing = []
for ds in bottom['dataset']:
    folder = BASE / 'Bottom_Datasets_mlp' / ds.replace('.tsv.gz','')
    target = folder / OUT_NAME
    if not target.exists():
        missing.append(ds)

print('Will evaluate', len(missing), 'datasets')
BASE = Path('data')
BOTTOM_CSV = BASE / 'bottom_dataset.csv'
MASTER = Path('results') / 'MLP Evaluation sheet(final).xlsx'
OUT_NAME = 'MLP Evaluation sheet(final)_filled.xlsx'
CV_N_SPLITS = 10
CV_RANDOM_STATE = 90483257


def read_parameters_from_master(master_path: Path):
    """Parse parameter combinations from the master Excel template.
    The function looks for rows where 'Hidden Neurons' header appears and
    reads following numeric rows as parameter combinations. Returns list of
    dicts with keys: hidden_neurons, activation, learning_rate, batch_size, epochs.
    If parsing fails, returns a small default grid.
    """
    params = []
    if not master_path.exists():
        return params
    try:
        df = pd.read_excel(master_path, sheet_name=0, header=None)
        # scan rows for numeric entries in column 1 with activation in col 3
        for i in range(len(df)):
            try:
                hidden = df.iat[i, 1]
            except Exception:
                hidden = None
            try:
                activation = df.iat[i, 3]
            except Exception:
                activation = None
            # accept if hidden is numeric and activation is present
            if pd.notna(hidden) and (isinstance(hidden, (int, float)) or str(hidden).isdigit()):
                if pd.isna(activation):
                    continue
                try:
                    lr = df.iat[i, 4] if pd.notna(df.iat[i, 4]) else None
                except Exception:
                    lr = None
                try:
                    bs = df.iat[i, 5] if pd.notna(df.iat[i, 5]) else None
                except Exception:
                    bs = None
                try:
                    ep = df.iat[i, 6] if pd.notna(df.iat[i, 6]) else None
                except Exception:
                    ep = None
                params.append({
                    'hidden_neurons': int(hidden),
                    'activation': str(activation).strip().lower(),
                    'learning_rate': float(lr) if lr is not None else 0.001,
                    'batch_size': int(bs) if bs is not None else 16,
                    'epochs': int(ep) if ep is not None else 100,
                })
    except Exception:
        params = []

    # dedupe
    unique = []
    seen = set()
    for p in params:
        key = (p['hidden_neurons'], p['activation'], p['learning_rate'], p['batch_size'], p['epochs'])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if unique:
        return unique
    # fallback default grid
    return [
        {'hidden_neurons': 16, 'activation': 'relu', 'learning_rate': 0.001, 'batch_size': 16, 'epochs': 100},
        {'hidden_neurons': 32, 'activation': 'relu', 'learning_rate': 0.001, 'batch_size': 16, 'epochs': 100},
        {'hidden_neurons': 64, 'activation': 'relu', 'learning_rate': 0.001, 'batch_size': 16, 'epochs': 100},
        {'hidden_neurons': 128, 'activation': 'relu', 'learning_rate': 0.001, 'batch_size': 16, 'epochs': 100},
    ]


def dataset_already_processed(folder: Path, out_name: str) -> bool:
    target = folder / out_name
    if not target.exists():
        return False
    try:
        xls = pd.ExcelFile(target)
        return 'AutoResults' in xls.sheet_names
    except Exception:
        return False


def main():
    bottom = pd.read_csv(BOTTOM_CSV)
    params = read_parameters_from_master(MASTER)
    print('Using', len(params), 'parameter combinations')

    to_run = []
    for ds in bottom['dataset']:
        folder = BASE / 'Bottom_Datasets_mlp' / ds.replace('.tsv.gz', '')
        target = folder / OUT_NAME
        if dataset_already_processed(folder, OUT_NAME):
            print('Skipping already-processed:', ds)
            continue
        to_run.append(ds)

    print('Will evaluate', len(to_run), 'datasets')

    for ds in to_run:
        row = bottom[bottom['dataset'] == ds].iloc[0]
        rows_count = int(row['rows'])
        if rows_count > 200000:
            print(f"Skipping {ds} (rows={rows_count}) — too large")
            continue
        folder = BASE / 'Bottom_Datasets_mlp' / ds.replace('.tsv.gz', '')
        folder.mkdir(parents=True, exist_ok=True)
        data_path = BASE / ds
        try:
            print('Loading', ds)
            data = pd.read_csv(data_path, compression='gzip', sep='\t')
            label_col = 'class' if 'class' in data.columns else 'target'
            X = data.drop(columns=[label_col]).values.astype(float)
            y = data[label_col].values

            results = []
            for p in params:
                h = p['hidden_neurons']
                act = p['activation']
                lr = p['learning_rate']
                bs = p['batch_size']
                ep = p['epochs']
                print(f"Evaluating {ds} h={h} act={act} lr={lr} bs={bs} ep={ep}")
                # map activation names to sklearn equivalents where possible
                activation_map = {
                    'sigmoid': 'logistic',
                    'logistic': 'logistic',
                    'tanh': 'tanh',
                    'relu': 'relu',
                    'identity': 'identity',
                    # fallback mappings for unsupported names
                    'gelu': 'relu',
                    'swish': 'relu',
                }
                act_mapped = activation_map.get(str(act).strip().lower(), 'relu')
                mlp = MLPClassifier(
                    hidden_layer_sizes=(h,),
                    activation=act_mapped,
                    solver='adam',
                    learning_rate_init=lr,
                    batch_size=bs,
                    max_iter=ep,
                    early_stopping=True,
                    random_state=324089,
                )
                pipeline = make_pipeline(RobustScaler(), mlp)
                cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
                start = time.perf_counter()
                try:
                    pred = cross_val_predict(pipeline, X, y, cv=cv)
                    elapsed = time.perf_counter() - start
                except Exception as e:
                    elapsed = None
                    print('Error during CV for', ds, e)
                    # record error row so user can see which combos failed
                    results.append({'hidden_neurons': h, 'activation': act, 'learning_rate': lr, 'batch_size': bs, 'epochs': ep, 'accuracy': None, 'macro_f1': None, 'balanced_accuracy': None, 'error': str(e), 'Time(s)': elapsed})
                    continue
                acc = accuracy_score(y, pred)
                f1 = f1_score(y, pred, average='macro', zero_division=0)
                ba = balanced_accuracy_score(y, pred)
                results.append({'hidden_neurons': h, 'activation': act, 'learning_rate': lr, 'batch_size': bs, 'epochs': ep, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': ba, 'error': None, 'Time(s)': elapsed})

            # copy master workbook into folder if not exists
            if MASTER.exists():
                shutil.copy(MASTER, folder / OUT_NAME)
            # write results to a sheet named after the dataset (sanitized)
            out_path = folder / OUT_NAME
            sheet_name = ds.replace('.tsv.gz', '')
            # sanitize sheet name (max 31 chars, remove invalid chars)
            invalid = set('[]:*?/\\')
            sheet_name = ''.join(c for c in sheet_name if c not in invalid)
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            with pd.ExcelWriter(out_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as ew:
                pd.DataFrame(results).to_excel(ew, sheet_name=sheet_name, index=False)
            print('Wrote results for', ds)
        except Exception as e:
            print('Failed', ds, e)


if __name__ == '__main__':
    main()
