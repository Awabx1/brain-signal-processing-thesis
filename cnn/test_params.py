import os
import numpy as np
import pandas as pd
import random
import itertools

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

###############################################################################
# 1) Data Loading (same approach as before)
###############################################################################
def load_eeg_data_for_cnn(root_folder="processed", label_list=['d', 'u', 'r', 'l'], 
                          time_points=128):
    """
    Loads CSV files containing raw EEG data from columns:
        [EEG.AF3, EEG.T7, EEG.Pz, EEG.T8, EEG.AF4].
    Stacks them into a shape (samples, time_points, channels).
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
                # Ensure columns exist
                if not all(col in df.columns for col in channels_of_interest):
                    continue

                signal = df[channels_of_interest].values  # shape = (num_samples, 5)
                if signal.shape[0] < time_points:
                    continue

                # Slice or trim to fixed time_points
                signal = signal[:time_points, :]  # shape = (128, 5)
                all_X.append(signal)
                all_y.append(label)

    all_X = np.array(all_X)  # shape = (n_samples, 128, 5)
    all_y = np.array(all_y)
    print(f"Loaded CNN data: {all_X.shape}, labels: {all_y.shape}")
    return all_X, all_y


###############################################################################
# 2) CNN Model Builder
###############################################################################
def build_cnn_model(input_shape, 
                    num_classes, 
                    conv1_filters=16, 
                    conv2_filters=32, 
                    dropout1=0.25, 
                    dropout2=0.25, 
                    dense_units=64,
                    learning_rate=1e-3):
    """
    Build a simple CNN with given hyperparameters.
      - input_shape = (time_points, channels)
      - conv1_filters, conv2_filters: # of filters in Conv2D layers
      - dropout1, dropout2: dropout rates after each pooling block
      - dense_units: # of neurons in the Dense layer
      - learning_rate: for Adam optimizer
    """
    model = models.Sequential()
    # Reshape to (time, channels, 1) for 2D convolution
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))

    # Convolution / Pooling block 1
    model.add(layers.Conv2D(conv1_filters, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))
    model.add(layers.Dropout(dropout1))

    # Convolution / Pooling block 2
    model.add(layers.Conv2D(conv2_filters, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))
    model.add(layers.Dropout(dropout2))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


###############################################################################
# 3) Parameter Search
###############################################################################
def parameter_search_cnn(X, y, param_grid, epochs=20, batch_size=8, seed=42):
    """
    - param_grid: dictionary or lists of parameter values to try
    - epochs, batch_size can be adjusted for speed or thoroughness
    - returns best_params, best_acc
    """
    best_acc = 0.0
    best_params = None

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=seed
    )

    param_combinations = list(itertools.product(*param_grid.values()))
    print(f"Searching over {len(param_combinations)} configurations...")

    for combo in param_combinations:
        params = dict(zip(param_grid.keys(), combo))
        print("Testing params:", params)

        # Build model
        model = build_cnn_model(
            input_shape=(X.shape[1], X.shape[2]),
            num_classes=num_classes,
            conv1_filters=params['conv1_filters'],
            conv2_filters=params['conv2_filters'],
            dropout1=params['dropout1'],
            dropout2=params['dropout2'],
            dense_units=params['dense_units'],
            learning_rate=params['learning_rate'],
        )

        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0  # set to 1 or 2 to see training logs
        )

        # Evaluate on test
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        print(f" --> Test Accuracy: {test_acc:.4f}")
        # Track best
        if test_acc > best_acc:
            best_acc = test_acc
            best_params = params

    return best_params, best_acc


def main():
    # 1) Set seeds
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # 2) Load data
    X, y = load_eeg_data_for_cnn("processed", time_points=128)
    if len(X) == 0:
        print("No data loaded. Exiting.")
        return

    # 3) Define parameter grid
    param_grid = {
        'conv1_filters': [8, 16],
        'conv2_filters': [16, 32],
        'dropout1': [0.25, 0.4],
        'dropout2': [0.25, 0.4],
        'dense_units': [32, 64],
        'learning_rate': [1e-3, 1e-4]
    }

    # 4) Run parameter search
    best_params, best_acc = parameter_search_cnn(
        X, y,
        param_grid=param_grid,
        epochs=15,       # reduce epochs for faster search
        batch_size=8,
        seed=SEED
    )

    # 5) Print result
    print("\nBest Parameter Combination:")
    print(best_params)
    print(f"Best Test Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()