import os
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

################################################################################
# LSTM also uses the same raw EEG columns: EEG.AF3, EEG.T7, EEG.Pz, EEG.T8, EEG.AF4
################################################################################

def load_eeg_data_for_lstm(root_folder="processed", label_list=['d', 'u', 'r', 'l'], 
                           time_points=128):
    """
    Similar to CNN, but for LSTM we keep the shape (samples, time, channels).
    """
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
    all_X = []
    all_y = []

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

                signal = signal[:time_points, :]  # shape = (128, 5)
                all_X.append(signal)
                all_y.append(label)

    all_X = np.array(all_X)  # shape = (n_samples, 128, 5)
    all_y = np.array(all_y)
    print(f"Loaded LSTM data: {all_X.shape}, labels: {all_y.shape}")
    return all_X, all_y


def build_lstm_model(input_shape, num_classes):
    """
    Basic LSTM with a single recurrent layer for classification.
    input_shape = (time_points, channels).
    """
    model = models.Sequential()
    model.add(layers.LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # 1) Load data
    X, y = load_eeg_data_for_lstm("processed", time_points=128)

    # 2) Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=SEED
    )

    # 4) Build LSTM model
    input_shape = (X.shape[1], X.shape[2])  # (128, 5)
    model = build_lstm_model(input_shape, num_classes)

    # 5) Compile
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    # 6) Train
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )

    # 7) Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[LSTM] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()