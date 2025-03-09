import os
import numpy as np
import pandas as pd
import random
import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def load_power_band_data_for_knn(
    root_folder="processed", 
    label_list=['d', 'u', 'r', 'l'],
    per_participant_norm=True,
    n_components=10
):
    """
    Loads power-band columns: POW.*. Then assigns "Label" to each row of df
    before slicing or appending. Optionally do per-participant normalization, then
    apply StandardScaler + PCA for dimensionality reduction.
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

                # Fill the NaNs
                df[power_cols] = df[power_cols].fillna(0)
                # Insert the Label column
                df["Label"] = label

                all_dfs.append(df)

    if not all_dfs:
        print("No suitable data found for KNN.")
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Per-participant normalization
    if per_participant_norm and 'Participant' in full_df.columns:
        def zscore_group(g):
            subset = g[power_cols]
            return (subset - subset.mean()) / (subset.std(ddof=1) + 1e-10)
        full_df[power_cols] = full_df.groupby('Participant')[power_cols].apply(zscore_group)

    X_raw = full_df[power_cols].values
    y = full_df["Label"].values

    # Global scaling (recommended for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Loaded KNN data (after PCA={n_components}): X={X_pca.shape}, y={y.shape}")
    return X_pca, y


def knn_parameter_search(X, y, param_grid, seed=42):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=seed
    )

    best_acc = 0.0
    best_params = None

    combos = list(itertools.product(*param_grid.values()))
    print(f"Testing {len(combos)} KNN parameter combinations...")

    for combo in combos:
        params = dict(zip(param_grid.keys(), combo))
        # Build model
        knn = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            metric=params['metric']
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

    X, y = load_power_band_data_for_knn(
        root_folder="processed", 
        per_participant_norm=True,
        n_components=10  
    )
    if X is None:
        return

    param_grid = {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'manhattan']  
    }

    best_params, best_acc = knn_parameter_search(X, y, param_grid, seed=SEED)
    print(f"\nBest KNN Params: {best_params}, Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()