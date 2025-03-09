import os
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_eeg_data_for_lstm(
    root_folder="processed", 
    label_list=['d', 'u', 'r', 'l'], 
    time_points=128,
    per_participant_norm=True
):
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
                df["Label"] = label
                participant = df["Participant"].iloc[0] if "Participant" in df.columns else "P0"
                if not all(col in df.columns for col in channels_of_interest):
                    continue

                signal = df[channels_of_interest].values
                if signal.shape[0] < time_points:
                    continue
                signal = signal[:time_points, :]
                data_list.append((signal, label, participant))

    if not data_list:
        return np.array([]), np.array([])

    if per_participant_norm:
        from collections import defaultdict
        part_map = defaultdict(list)
        for (sig, lab, part) in data_list:
            part_map[part].append(sig)
        for part, signals in part_map.items():
            cat = np.concatenate(signals, axis=0)
            mean_ = cat.mean(axis=0)
            std_ = cat.std(axis=0, ddof=1) + 1e-10
            part_map[part] = (mean_, std_)

        X_list, y_list = [], []
        for (sig, lab, part) in data_list:
            mean_, std_ = part_map[part]
            X_list.append((sig - mean_) / std_)
            y_list.append(lab)
        X = np.array(X_list)
        y = np.array(y_list)
    else:
        X = np.array([item[0] for item in data_list])
        y = np.array([item[1] for item in data_list])

    return X, y

def build_lstm_best(input_shape, num_classes):
    """
    Best LSTM params found:
      lstm_units=16, dense_units=16, dropout=0.3, learning_rate=0.0001
    """
    lstm_units = 16
    dense_units = 16
    dropout = 0.3
    learning_rate = 1e-4

    model = models.Sequential()
    model.add(layers.LSTM(lstm_units, input_shape=input_shape))
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

def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X, y = load_eeg_data_for_lstm("processed", time_points=128, per_participant_norm=True)
    if len(X) == 0:
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=SEED
    )
    
    model = build_lstm_best((X.shape[1], X.shape[2]), num_classes)
    model.fit(X_train, y_train, epochs=30, batch_size=8, validation_split=0.2, verbose=1)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[LSTM Best Params] Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()