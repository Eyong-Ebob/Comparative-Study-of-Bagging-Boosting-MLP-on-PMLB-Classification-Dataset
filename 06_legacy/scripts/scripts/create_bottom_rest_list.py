from pathlib import Path
import pandas as pd

root = Path('.').resolve()
bottom = pd.read_csv(root/'data'/'bottom_dataset.csv')
completed = set()
for d in (root/'data'/'bottom_datasets').iterdir():
    if d.is_dir():
        p = d/'mlp_hyperparameter_combinations.csv'
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df[['Balanced_Accuracy','Accuracy','Macro F1 Score','Time(s)']].isna().any().any():
                    completed.add(d.name)
            except Exception:
                pass

b10f = root/'bottom_10_seq.txt'
b10 = set()
if b10f.exists():
    b10 = set([line.strip().replace('.tsv.gz','') for line in b10f.read_text(encoding='utf-8').splitlines() if line.strip()])

todo = []
for ds in bottom['dataset'].astype(str).tolist():
    name = ds.replace('.tsv.gz','')
    if name not in completed and name not in b10:
        todo.append(ds)

out = root/'bottom_rest_seq.txt'
out.write_text('\n'.join(todo)+'\n', encoding='utf-8')
print('wrote', len(todo), 'datasets to', out)
