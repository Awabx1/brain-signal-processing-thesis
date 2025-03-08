import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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


# ---------------------------------------------------------
# 1) EEGNet Model (channels-last)
# ---------------------------------------------------------
def EEGNet(nb_classes,
           Chans=5,
           Samples=128,
           dropoutRate=0.3,
           kernLength=64,
           F1=8,
           D=2,
           F2=16):
    """
    EEGNet in 'channels_last' format. 
    Input shape: (Chans, Samples, 1) => default (5,128,1).
    """
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


# ---------------------------------------------------------
# 2) Load CSV Data
# ---------------------------------------------------------
def load_all_data(root_folder, label_list=['d', 'u', 'r', 'l'], time_points=128):
    """
    Loads CSV data from files ending with '_processed.csv'.
    - label = first char of filename (d/u/r/l)
    - instance = second char of filename (digit)
    - channels_of_interest = [AF3, T7, Pz, T8, AF4]
    """
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
                signal = df[channels_of_interest].values.T  # shape => (5, total_samples)

                # If not enough points, skip or pad here
                if signal.shape[1] < time_points:
                    print(f"File {file} only has {signal.shape[1]} samples, needs {time_points}")
                    continue

                # Trim to exact length
                signal = signal[:, :time_points]  # => (5, 128)
                # Add the final dimension => (5, 128, 1)
                signal = np.expand_dims(signal, axis=-1)

                all_X.append(signal)
                all_y.append(label)

    all_X = np.array(all_X)  # => (N, 5, 128, 1)
    all_y = np.array(all_y)

    print(f"Total files loaded: {len(all_X)}")
    return all_X, all_y


# ---------------------------------------------------------
# 3) Train EEGNet
# ---------------------------------------------------------
def main():
    root_folder = 'processed'
    X, y = load_all_data(root_folder)

    # Encode labels -> integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )

    nb_classes = len(np.unique(y_encoded))
    chans = X.shape[1]    # 5
    samples = X.shape[2]  # 128

    # Build EEGNet
    model = EEGNet(
        nb_classes=nb_classes,
        Chans=chans,
        Samples=samples,
        dropoutRate=0.5,
        kernLength=32,
        F1=8,
        D=1,
        F2=16
    )

    # Compile with the Adam from tensorflow.python.keras.optimizer_v2.adam
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        batch_size=4,
        epochs=100,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")


if __name__ == '__main__':
    main()