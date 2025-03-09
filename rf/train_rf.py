import os
import numpy as np
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

################################################################################
# Random Forest: primarily uses power-band features, but can also include
# (optional) motion columns: MOT.* or contact-quality columns: CQ.* if relevant.
################################################################################

def load_power_band_data_for_rf(root_folder="processed", label_list=['d', 'u', 'r', 'l']):
    """
    Similar to load_power_band_data_for_knn, but we can optionally add
    motion columns or contact-quality columns if desired.
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
                # Power band columns
                power_cols = [c for c in df.columns if c.startswith("POW.")]
                # Example: motion columns
                # mot_cols = [c for c in df.columns if c.startswith("MOT.")]
                # Example: contact quality columns
                # cq_cols = [c for c in df.columns if c.startswith("CQ.")]
                # In practice, you may combine them:
                # feature_cols = power_cols + mot_cols + cq_cols

                if len(power_cols) == 0:
                    continue

                # Fill or drop
                df[power_cols] = df[power_cols].fillna(0)
                df = df.assign(Label=label)
                all_dfs.append(df)

    if not all_dfs:
        print("No suitable data found for Random Forest.")
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)
    power_cols = [c for c in full_df.columns if c.startswith("POW.")]

    X = full_df[power_cols].values
    y = full_df["Label"].values
    print(f"Loaded RF data: X={X.shape}, y={y.shape}")
    return X, y


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # 1) Load data
    X, y = load_power_band_data_for_rf("processed")
    if X is None:
        print("No data for RF. Exiting.")
        return

    # 2) Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=SEED
    )

    # 4) RandomForest with more trees and some max depth
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = rf.predict(X_test)
    print("[RandomForest] Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()