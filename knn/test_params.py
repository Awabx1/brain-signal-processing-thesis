
import os
import numpy as np
import pandas as pd
import random
import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def load_power_band_data_for_knn(root_folder="processed", label_list=['d', 'u', 'r', 'l']):
    """
    Selects power-band columns (POW.*) as features. Each row is treated as a sample.
    """
    all_dfs = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith("_processed.csv"):
                filepath = os.path.join(subdir, file)
                label = file[0]
                if label not in label_list:
                    continue
                df = pd.read_csv(filepath)
                power_cols = [c for c in df.columns if c.startswith("POW.")]
                if not power_cols:
                    continue
                df[power_cols] = df[power_cols].fillna(0)
                df = df.assign(Label=label)
                all_dfs.append(df)

    if not all_dfs:
        print("No suitable data found for KNN.")
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)
    power_cols = [c for c in full_df.columns if c.startswith("POW.")]
    X = full_df[power_cols].values
    y = full_df["Label"].values
    print(f"Loaded KNN data: X={X.shape}, y={y.shape}")
    return X, y


def knn_parameter_search(X, y, param_grid, seed=42):
    """
    param_grid is a dict with lists of possible values, e.g.:
      {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
      }
    Performs a grid search using train/test split and returns best_params, best_acc.
    """
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=seed
    )

    best_acc = 0.0
    best_params = None

    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_combos = list(itertools.product(*param_grid.values()))
    print(f"Testing {len(param_combos)} KNN parameter combinations...")

    for combo in param_combos:
        params = dict(zip(param_keys, combo))
        # Build and train model
        knn = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            p=params['p']
        )
        knn.fit(X_train, y_train)

        # Evaluate
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Params={params}, Accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = params

    return best_params, best_acc


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    X, y = load_power_band_data_for_knn("processed")
    if X is None:
        return

    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    best_params, best_acc = knn_parameter_search(X, y, param_grid, seed=SEED)
    print(f"\nBest KNN Params: {best_params}, Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()