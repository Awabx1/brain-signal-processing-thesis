import os
import numpy as np
import pandas as pd
import random
import itertools

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_eeg_data_for_lstm(
    root_folder="processed", 
    label_list=['d', 'u', 'r', 'l'], 
    time_points=128,
    per_participant_norm=True
):
    """
    Similar to CNN approach, but storing data for LSTM.
    Appends (signal, label, participant). Then optionally does per-participant norm.
    """
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
    data_list = []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith("_processed.csv"):
                filepath = os.path.join(subdir, file)
                label = file[0]
                if label not in label_list:
                    continue

                df = pd.read_csv(filepath)
                df["Label"] = label  # assign label first

                participant = (
                    df["Participant"].iloc[0] 
                    if "Participant" in df.columns else "P0"
                )
                if not all(col in df.columns for col in channels_of_interest):
                    continue

                signal = df[channels_of_interest].values
                if signal.shape[0] < time_points:
                    continue

                signal = signal[:time_points, :]
                data_list.append((signal, label, participant))

    if not data_list:
        print("No data found for LSTM.")
        return np.array([]), np.array([])

    if per_participant_norm:
        from collections import defaultdict
        part_map = defaultdict(list)
        for (sig, lab, part) in data_list:
            part_map[part].append(sig)

        # Compute per-participant mean/std
        for part, signals in part_map.items():
            concatenated = np.concatenate(signals, axis=0)
            mean_ = concatenated.mean(axis=0)
            std_ = concatenated.std(axis=0, ddof=1) + 1e-10
            part_map[part] = (mean_, std_)

        X_list, y_list = [], []
        for (sig, lab, part) in data_list:
            mean_, std_ = part_map[part]
            sig_norm = (sig - mean_) / std_
            X_list.append(sig_norm)
            y_list.append(lab)

        X = np.array(X_list)
        y = np.array(y_list)
    else:
        X = np.array([item[0] for item in data_list])
        y = np.array([item[1] for item in data_list])

    print(f"Loaded LSTM data: {X.shape}, labels: {y.shape}")
    return X, y


def build_small_lstm_model(
    input_shape,
    num_classes,
    lstm_units=32,
    dense_units=16,
    dropout=0.3,
    learning_rate=1e-3
):
    """
    A smaller LSTM architecture to reduce overfitting.
    """
    model = models.Sequential()
    model.add(layers.LSTM(lstm_units, input_shape=input_shape, return_sequences=False))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


def lstm_parameter_search(X, y, param_grid, epochs=30, batch_size=8, seed=42):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=seed
    )

    best_acc = 0.0
    best_params = None

    combos = list(itertools.product(*param_grid.values()))
    print(f"Testing {len(combos)} LSTM configurations...")

    for combo in combos:
        params = dict(zip(param_grid.keys(), combo))
        print(f"Trying params: {params}")

        model = build_small_lstm_model(
            input_shape=(X.shape[1], X.shape[2]),
            num_classes=num_classes,
            lstm_units=params['lstm_units'],
            dense_units=params['dense_units'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate']
        )

        es = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[es],
            verbose=0
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f" --> Test Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = params

    return best_params, best_acc


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X, y = load_eeg_data_for_lstm(
        root_folder="processed", 
        time_points=128,
        per_participant_norm=True
    )
    if len(X) == 0:
        print("No data for LSTM. Exiting.")
        return

    param_grid = {
        'lstm_units': [16, 32],
        'dense_units': [16, 32],
        'dropout': [0.2, 0.3],
        'learning_rate': [1e-3, 1e-4]
    }

    best_params, best_acc = lstm_parameter_search(
        X, y, param_grid, epochs=25, batch_size=8, seed=SEED
    )
    print("\nBest LSTM Params:", best_params)
    print(f"Best Test Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
