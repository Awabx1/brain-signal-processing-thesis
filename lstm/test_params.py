
import os
import numpy as np
import pandas as pd
import random
import itertools

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_eeg_data_for_lstm(root_folder="processed", label_list=['d', 'u', 'r', 'l'], 
                           time_points=128):
    """
    Loads raw EEG channels [AF3, T7, Pz, T8, AF4] at fixed time_points for LSTM.
    Returns shape (samples, time_points, channels) for X, plus label array y.
    """
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
    all_X, all_y = [], []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith("_processed.csv"):
                filepath = os.path.join(subdir, file)
                label = file[0]
                if label not in label_list:
                    continue

                df = pd.read_csv(filepath)
                if not all(col in df.columns for col in channels_of_interest):
                    continue
                signal = df[channels_of_interest].values
                if signal.shape[0] < time_points:
                    continue

                signal = signal[:time_points, :]
                all_X.append(signal)
                all_y.append(label)

    X = np.array(all_X)
    y = np.array(all_y)
    print(f"Loaded LSTM data: {X.shape}, labels: {y.shape}")
    return X, y


def build_lstm_model(input_shape, num_classes, lstm_units=64, dropout=0.5, 
                     dense_units=32, learning_rate=1e-3):
    """
    Build a simple LSTM model with parameterized hyperparameters.
    """
    model = models.Sequential()
    model.add(layers.LSTM(lstm_units, return_sequences=False, input_shape=input_shape))
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


def lstm_parameter_search(X, y, param_grid, epochs=20, batch_size=8, seed=42):
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

        input_shape = (X.shape[1], X.shape[2])
        model = build_lstm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=params['lstm_units'],
            dropout=params['dropout'],
            dense_units=params['dense_units'],
            learning_rate=params['learning_rate']
        )

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
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

    X, y = load_eeg_data_for_lstm("processed", time_points=128)
    if len(X) == 0:
        print("No data for LSTM. Exiting.")
        return

    param_grid = {
        'lstm_units': [32, 64],
        'dropout': [0.3, 0.5],
        'dense_units': [32, 64],
        'learning_rate': [1e-3, 1e-4],
    }

    best_params, best_acc = lstm_parameter_search(
        X, y,
        param_grid=param_grid,
        epochs=15,   # or fewer if time is a concern
        batch_size=8,
        seed=SEED
    )

    print("\nBest LSTM Params:", best_params)
    print(f"Best Test Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
