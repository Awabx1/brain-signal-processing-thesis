import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, 
    SeparableConv2D, DepthwiseConv2D, Dropout, Flatten
)
from tensorflow.keras.optimizers import Adam

def EEGNet(nb_classes, Chans = 5, Samples = 128,
           dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16):
    """
    A simplified EEGNet implementation for demonstration.
    nb_classes: number of classes (e.g., 4 for {d, u, r, l})
    Chans: number of electrodes/channels
    Samples: number of time points in each trial
    dropoutRate: dropout rate
    kernLength: length of temporal convolution kernel
    F1, D, F2: filter dimensions in the original EEGNet design
    """
    # Input shape: (batch, 1, Chans, Samples) – you can also use (batch, Chans, Samples, 1)
    input_shape = (1, Chans, Samples)

    inputs = Input(shape=input_shape)
    
    # Block1: Temporal Convolution
    # --------------------------------
    x = Conv2D(F1, (1, kernLength), padding='same', 
               input_shape=input_shape, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((Chans, 1), use_bias=False, 
                        depth_multiplier=D, depthwise_constraint=None,
                        padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)  
    x = Dropout(dropoutRate)(x)

    # Block2: Separable Convolution
    # --------------------------------
    x = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)

    # Classification
    # --------------------------------
    x = Flatten()(x)
    outputs = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)

# -----------------------------------------
# 2) Load and Preprocess Your CSV Data
# -----------------------------------------
def load_all_data(root_folder, label_list=['d', 'u', 'r', 'l'], 
                  time_points=128):
    """
    Example function to scan your participant folders, load your CSVs, 
    and build arrays of shape:
        X -> (num_trials, 1, num_channels, num_timepoints)
        y -> (num_trials,)
    NOTE: You must adapt to how your data is actually stored!
    """
    all_X = []
    all_y = []

    # For simplicity, assume you have a known set of files or a known structure:
    # We look for each subject folder, gather CSV for each trial, and extract 
    # the signal from columns [EEG.AF3, EEG.T7, EEG.Pz, EEG.T8, EEG.AF4],
    # or any subset of columns you want.
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']

    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_processed.csv'):
                filepath = os.path.join(subdir, file)
                df = pd.read_csv(filepath)

                # Get the label from the filename or from the 'Label' column
                # For instance, if the file is named "d1_..._processed.csv", 
                # the label might be 'd'.
                # Alternatively, you can use the 'Label' column in the CSV.
                label = df['Label'].iloc[0]  # if consistent

                if label not in label_list:
                    continue

                # Extract your channel signals for a certain time window or 
                # entire trial. 
                # E.g., for one "trial," suppose we take the first "time_points" rows:
                # You must ensure each trial truly has enough samples.
                signal = df[channels_of_interest].values[:time_points, :].T  # shape (5, time_points)
                
                # If shape is smaller than time_points, skip or pad
                if signal.shape[1] < time_points:
                    continue  # or pad it if you prefer

                # Expand to (1, 5, time_points)
                signal = np.expand_dims(signal, axis=0)

                all_X.append(signal)
                all_y.append(label)
    
    all_X = np.array(all_X)  # shape -> (NumTrials, 1, 5, time_points)
    all_y = np.array(all_y)

    return all_X, all_y

# --------------------------------------
# 3) Train EEGNet
# --------------------------------------
def main():
    root_folder = '/path/to/your/cleaned_csvs'
    X, y = load_all_data(root_folder)

    # Encode labels from 'd,u,r,l' -> 0,1,2,3
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )

    # Instantiate EEGNet. Adjust hyperparameters to your data dimension
    nb_classes = len(np.unique(y_encoded))
    chans = X.shape[2]      # 5
    samples = X.shape[3]    # e.g. 128

    model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples, 
                   dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=30,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Evaluate
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("Validation Loss: %.4f" % scores[0])
    print("Validation Accuracy: %.4f" % scores[1])

if __name__ == '__main__':
    main()