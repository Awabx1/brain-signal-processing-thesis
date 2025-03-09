import os
import numpy as np
import pandas as pd
import random

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
    Loads power-band features and applies per-participant normalization plus PCA.
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
                df["Label"] = label
                all_dfs.append(df)

    if not all_dfs:
        print("No suitable data found for KNN.")
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)

    if per_participant_norm and 'Participant' in full_df.columns:
        def zscore_group(g):
            subset = g[power_cols]
            return (subset - subset.mean()) / (subset.std(ddof=1) + 1e-10)
        power_cols = [c for c in full_df.columns if c.startswith("POW.")]
        full_df[power_cols] = full_df.groupby('Participant')[power_cols].apply(zscore_group)

    power_cols = [c for c in full_df.columns if c.startswith("POW.")]
    X_raw = full_df[power_cols].values
    y = full_df["Label"].values

    # Global scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, y

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

    # Best KNN params found:
    # {'n_neighbors': 3, 'weights': 'distance', 'metric': 'manhattan'}
    best_knn = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric='manhattan'
    )

    # Train/test split
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=SEED
    )

    best_knn.fit(X_train, y_train)
    y_pred = best_knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[KNN Best Params] Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
