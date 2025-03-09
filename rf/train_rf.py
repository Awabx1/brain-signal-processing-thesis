import os
import numpy as np
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_features_for_rf(
    root_folder="processed", 
    label_list=['d', 'u', 'r', 'l'],
    use_power_bands=True,
    use_motion=True,
    use_cq=True,
    per_participant_norm=True
):
    all_dfs = []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith("_processed.csv"):
                filepath = os.path.join(subdir, file)
                label = file[0]
                if label not in label_list:
                    continue
                df = pd.read_csv(filepath)

                feature_cols = []
                if use_power_bands:
                    feature_cols += [c for c in df.columns if c.startswith("POW.")]
                if use_motion:
                    feature_cols += [c for c in df.columns if c.startswith("MOT.")]
                if use_cq:
                    feature_cols += [c for c in df.columns if c.startswith("CQ.")]

                feature_cols = list(set(feature_cols))
                if not feature_cols:
                    continue
                df[feature_cols] = df[feature_cols].fillna(0)

                df["Label"] = label
                keep_cols = feature_cols + ["Label"]
                if "Participant" in df.columns:
                    keep_cols.append("Participant")

                sub_df = df[keep_cols].copy()
                all_dfs.append(sub_df)

    if not all_dfs:
        print("No suitable data found for RF.")
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)
    numeric_cols = list(full_df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [c for c in numeric_cols if c != "Label"]

    if per_participant_norm and "Participant" in full_df.columns:
        def zscore_group(g):
            x = g[numeric_cols]
            return (x - x.mean())/(x.std(ddof=1)+1e-10)
        full_df[numeric_cols] = full_df.groupby("Participant")[numeric_cols].apply(zscore_group)

    X = full_df[numeric_cols].values
    y = full_df["Label"].values
    print(f"Loaded RF data: X={X.shape}, y={y.shape}")

    # global scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    X, y = load_features_for_rf(
        root_folder="processed",
        use_power_bands=True,
        use_motion=True,
        use_cq=True,
        per_participant_norm=True
    )
    if X is None:
        return

    # Best params from your search:
    # {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'max_features': 'log2'}
    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        max_features='log2',
        random_state=SEED,
        n_jobs=-1
    )

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=SEED
    )

    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RF Best Params] Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
