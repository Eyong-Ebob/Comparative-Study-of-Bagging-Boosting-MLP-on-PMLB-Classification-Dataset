"""
Apply 10-fold CV template to all bottom-dataset notebooks.
"""
import os
import re
import json
import pathlib

BASE = r"c:\Users\EYONGEBOB\Desktop\PMLB Project\sklearn-benchmarks"
TEMPLATE = os.path.join(BASE, "results/MLP Testing Final/BreastCancer_10fold_template.ipynb")
BOTTOM   = os.path.join(BASE, "results/MLP Testing Final/Bottom_Datasets")

def load_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def save_json(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, indent=1), encoding='utf-8')

template_nb = load_json(TEMPLATE)

folders = [d for d in os.listdir(BOTTOM) if os.path.isdir(os.path.join(BOTTOM, d))]
folders = sorted(folders)

updated = 0
missing = []

for ds in folders:
    nb_path = os.path.join(BOTTOM, ds, f"{ds}.ipynb")
    if not os.path.exists(nb_path):
        missing.append(ds)
        continue

    nb_new = json.loads(json.dumps(template_nb))  # deep copy

    # Customize DATASET_FILE for generated dataset copy
    for cell in nb_new['cells']:
        if cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', []))
            if 'DATASET_FILE = ' in src:
                src = re.sub(r'DATASET_FILE\s*=\s*"[^"]*"', 'DATASET_FILE = "dataset.tsv.gz"', src)
                cell['source'] = src.split('\n')
                break

    save_json(nb_path, nb_new)
    updated += 1

print(f"updated_notebooks={updated}")
print(f"total_folders={len(folders)}")
if missing:
    print(f"missing_notebook_files={len(missing)}")
    print("missing examples:", missing[:10])
else:
    print("missing_notebook_files=0")
