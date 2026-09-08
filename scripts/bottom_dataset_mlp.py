from pathlib import Path
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


BOTTOM_FILE = Path("data/bottom_dataset.csv")
DATA_DIR = Path("data")
OUTPUT_FILE = Path("notebooks/bottom_mlp_results_ranked.csv")

# Required CV protocol
CV_N_SPLITS = 10
CV_RANDOM_STATE = 90483257


def evaluate_bottom_datasets() -> pd.DataFrame:
    bottom_df = pd.read_csv(BOTTOM_FILE)
    print(f"Bottom CSV length: {len(bottom_df)}")

    bottom_results = []

    for idx, row in bottom_df.iterrows():
        dataset_name = row["dataset"]
        dataset_path = DATA_DIR / dataset_name

        try:
            data = pd.read_csv(dataset_path, compression="gzip", sep="\t")
            label_col = "class" if "class" in data.columns else "target"

            X = data.drop(columns=[label_col]).values.astype(float)
            y = data[label_col].values

            mlp = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                max_iter=1000,
                early_stopping=True,
                random_state=324089,
            )

            pipeline = make_pipeline(RobustScaler(), mlp)
            cv = StratifiedKFold(
                n_splits=CV_N_SPLITS,
                shuffle=True,
                random_state=CV_RANDOM_STATE,
            )
            pred = cross_val_predict(pipeline, X, y, cv=cv)

            accuracy = accuracy_score(y, pred)
            macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
            balanced_accuracy = balanced_accuracy_score(y, pred)

            bottom_results.append(
                {
                    "dataset": dataset_name,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "balanced_accuracy": balanced_accuracy,
                }
            )

            print(f"[{idx + 1}/{len(bottom_df)}] {dataset_name}: BA={balanced_accuracy:.4f}")
        except Exception as e:
            print(f"[{idx + 1}/{len(bottom_df)}] {dataset_name}: ERROR - {e}")

    return pd.DataFrame(bottom_results)


def main() -> None:
    results_df = evaluate_bottom_datasets()
    print(f"\nBottom Results ({len(results_df)} datasets):")
    print(results_df.head())

    ranked_df = results_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    ranked_df.insert(0, "rank", ranked_df.index + 1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ranked_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Bottom ranked results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
