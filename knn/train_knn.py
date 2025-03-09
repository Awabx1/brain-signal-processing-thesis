
import os
import numpy as np
import pandas as pd
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

################################################################################
# KNN: uses the power-band columns (POW.*) as features.
################################################################################

def load_power_band_data_for_knn(root_folder="processed", label_list=['d', 'u', 'r', 'l']):
    """
    Loads CSV files and selects columns that are power band features (POW.*).
    Then we treat each row as a sample. This is one approach, but note that
    this may yield many samples per participant (each time frame).
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
                # Filter to only columns starting with 'POW.'
                power_cols = [c for c in df.columns if c.startswith("POW.")]
                if len(power_cols) == 0:
                    continue

                # Fill NaN with 0
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


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # 1) Load data
    X, y = load_power_band_data_for_knn("processed")
    if X is None:
        print("No data for KNN. Exiting.")
        return

    # 2) Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=SEED
    )

    # 4) KNN with smaller k to handle small dataset
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2)  
    knn.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = knn.predict(X_test)
    print("[KNN] Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()