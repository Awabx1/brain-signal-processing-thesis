import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from itertools import product

# TensorFlow's internal Keras
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Activation,
    AveragePooling2D,
    SeparableConv2D,
    DepthwiseConv2D,
    Dropout,
    Flatten
)
# BatchNormalization from tensorflow.python.layers.normalization
from tensorflow.python.layers.normalization import BatchNormalization
# Adam from tensorflow.python.keras.optimizer_v2.adam
from tensorflow.python.keras.optimizer_v2.adam import Adam


# 1) Define EEGNet
def EEGNet(nb_classes,
           Chans=5,
           Samples=128,
           dropoutRate=0.3,
           kernLength=64,
           F1=8,
           D=2,
           F2=16):
    input_shape = (Chans, Samples, 1)
    inputs = Input(shape=input_shape)

    # Block1: Temporal Convolution
    x = Conv2D(
        filters=F1,
        kernel_size=(1, kernLength),
        padding='same',
        data_format='channels_last',
        use_bias=False
    )(inputs)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D(
        kernel_size=(Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        data_format='channels_last',
        padding='valid'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4), data_format='channels_last')(x)
    x = Dropout(dropoutRate)(x)

    # Block2: Separable Convolution
    x = SeparableConv2D(
        filters=F2,
        kernel_size=(1, 16),
        use_bias=False,
        padding='same',
        data_format='channels_last'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 8), data_format='channels_last')(x)
    x = Dropout(dropoutRate)(x)

    # Classification
    x = Flatten()(x)
    outputs = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


# 2) Load Data
def load_all_data(root_folder, label_list=['d', 'u', 'r', 'l'], time_points=128):
    all_X = []
    all_y = []
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_processed.csv'):
                filepath = os.path.join(subdir, file)
                label = file[0]     # e.g. 'd'
                instance = file[1]  # e.g. '1'
                if label not in label_list or not instance.isdigit():
                    continue

                df = pd.read_csv(filepath)
                signal = df[channels_of_interest].values.T
                if signal.shape[1] < time_points:
                    continue
                signal = signal[:, :time_points]
                signal = np.expand_dims(signal, axis=-1)

                all_X.append(signal)
                all_y.append(label)

    all_X = np.array(all_X)
    all_y = np.array(all_y)
    print(f"Total files loaded: {len(all_X)}")
    return all_X, all_y


# 3) Training & Parameter Search
def main():
    # 1) Load Data
    root_folder = 'processed'
    X, y = load_all_data(root_folder)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )

    nb_classes = len(np.unique(y_encoded))
    chans = X.shape[1]
    samples = X.shape[2]

    # Define parameter candidates
    random_seeds = [1, 42, 2023]
    learning_rates = [1e-3, 5e-4]
    batch_sizes = [4, 8]
    dropouts = [0.3, 0.5]
    kern_lengths = [32, 64]
    f1_list = [8, 16]
    d_list = [1, 2]
    f2_list = [16, 32]
    num_epochs = 50

    # If you want to vary epochs too:
    # epochs_list = [50, 100]

    # For each combination, train & track best
    best_val_acc = -1.0
    best_params = None

    # Build a "Cartesian product" of parameter sets
    for seed, lr, bs, do, kl, F1, D, F2 in product(
        random_seeds, learning_rates, batch_sizes,
        dropouts, kern_lengths, f1_list, d_list, f2_list
    ):

        # Fix random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Build EEGNet
        model = EEGNet(
            nb_classes=nb_classes,
            Chans=chans,
            Samples=samples,
            dropoutRate=do,
            kernLength=kl,
            F1=F1,
            D=D,
            F2=F2
        )

        # Compile with specific learning rate
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=lr),
            metrics=['accuracy']
        )

        # Train
        model.fit(
            X_train, y_train,
            batch_size=bs,
            epochs=num_epochs,
            validation_data=(X_val, y_val),
            verbose=0  # hide the training logs for brevity
        )

        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

        # Check if this is the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'seed': seed,
                'learning_rate': lr,
                'batch_size': bs,
                'dropout': do,
                'kernLength': kl,
                'F1': F1,
                'D': D,
                'F2': F2,
                'epochs': num_epochs
            }

            print(f"New best val_acc={val_acc:.4f} with params {best_params}")

    print("\n======================================")
    print(f"Best val_acc so far = {best_val_acc:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()