import os
import numpy as np
import pandas as pd
import random
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


def load_features_for_rf(
    root_folder="processed", 
    label_list=['d', 'u', 'r', 'l'],
    use_power_bands=True,
    use_motion=True,
    use_cq=True,
    per_participant_norm=True
):
    """
    Loads a combination of features: 
      • POW.*  (power bands)
      • MOT.*  (motion)
      • CQ.*   (contact quality)
    from CSV files ending with "_processed.csv" 
    and merges them into a single DataFrame for classification.

    Steps:
      1) Identify relevant feature columns in each CSV (based on user flags).
      2) Assign the "Label" column BEFORE slicing columns with keep_cols.
      3) Optionally perform per-participant z-scoring on numeric columns.
      4) Return scaled feature matrix X and label array y.
    """
    all_dfs = []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith("_processed.csv"):
                # label is the first character of the filename
                filepath = os.path.join(subdir, file)
                label = file[0]
                if label not in label_list:
                    continue

                df = pd.read_csv(filepath)

                # Identify which feature columns to keep
                feature_cols = []
                if use_power_bands:
                    feature_cols += [c for c in df.columns if c.startswith("POW.")]
                if use_motion:
                    feature_cols += [c for c in df.columns if c.startswith("MOT.")]
                if use_cq:
                    feature_cols += [c for c in df.columns if c.startswith("CQ.")]

                feature_cols = list(set(feature_cols))  # remove duplicates
                if not feature_cols:
                    # No relevant columns in this file
                    continue

                # Fill NaNs in the feature columns
                df[feature_cols] = df[feature_cols].fillna(0)

                # 1) Assign the "Label" column to df BEFORE slicing
                df["Label"] = label

                # Build a list of columns to keep: features + label (+ Participant)
                keep_cols = feature_cols + ["Label"]
                if "Participant" in df.columns:
                    keep_cols.append("Participant")

                # 2) Now slice the DataFrame with keep_cols
                sub_df = df[keep_cols].copy()

                # We append sub_df to all_dfs so we can concatenate later
                all_dfs.append(sub_df)

    if not all_dfs:
        print("No suitable data found for RF.")
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Collect numeric columns for normalization (exclude the label)
    numeric_cols = list(full_df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [c for c in numeric_cols if c != "Label"]  # do not normalize label

    # Optional: per-participant normalization
    if per_participant_norm and "Participant" in full_df.columns:
        def zscore_group(g):
            x = g[numeric_cols]
            return (x - x.mean()) / (x.std(ddof=1) + 1e-10)
        full_df[numeric_cols] = full_df.groupby("Participant")[numeric_cols].apply(zscore_group)

    # Prepare final X, y
    X = full_df[numeric_cols].values
    y = full_df["Label"].values
    print(f"Loaded RF data: X={X.shape}, y={y.shape}")

    # Optionally apply a global StandardScaler as well (often helps in many ML algorithms).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def rf_parameter_search(X, y, param_grid, seed=42):
    """
    Performs a simple grid search over param_grid to find best RandomForest params.
    param_grid: dictionary with lists of values, e.g.:
      {
        'n_estimators': [100, 200],
        'max_depth': [20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
      }
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=seed
    )

    best_acc = 0.0
    best_params = None

    combos = list(itertools.product(*param_grid.values()))
    print(f"Testing {len(combos)} RF configurations...")

    for combo in combos:
        params = dict(zip(param_grid.keys(), combo))
        rf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            max_features=params['max_features'],
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
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

    # Load the data with user-chosen combinations of features
    X, y = load_features_for_rf(
        root_folder="processed",
        use_power_bands=True,
        use_motion=True,
        use_cq=True,
        per_participant_norm=True
    )
    if X is None:
        return

    # Define our parameter grid
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }

    # Perform the parameter search
    best_params, best_acc = rf_parameter_search(X, y, param_grid, seed=SEED)
    print(f"\nBest RandomForest Params: {best_params}, Best Accuracy: {best_acc:.4f}")

    # Optionally re-train on the entire dataset and examine feature importances
    final_rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        max_features=best_params['max_features'],
        random_state=SEED,
        n_jobs=-1
    )
    from sklearn.preprocessing import LabelEncoder
    final_rf.fit(X, LabelEncoder().fit_transform(y))

    feature_importances = final_rf.feature_importances_
    print("Feature importances shape:", feature_importances.shape)
    # You could print or sort them—here's a quick preview:
    top_inds = np.argsort(feature_importances)[::-1][:10]
    print("Top 10 feature indices:", top_inds)
    print("Top 10 importances:", feature_importances[top_inds])


if __name__ == "__main__":
    main()