#pip uninstall tensorflow
#pip install tensorflow_macos==2.12.0
#pip uninstall keras
#pip install keras==2.12.0

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras import backend as Ks
from tensorflow.python.keras.models import Model

from tensorflow.python.layers.normalization import BatchNormalization

from tensorflow.python.keras.layers import (
    Input, Dense, Conv2D, Activation, AveragePooling2D, 
    SeparableConv2D, DepthwiseConv2D, Dropout, Flatten
)
# from tensorflow.python.keras.optimizers import Adam
from keras.optimizers import Adam

def EEGNet(nb_classes, Chans = 5, Samples = 128,
           dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16):
    """
    A simplified EEGNet implementation for demonstration.
    nb_classes: number of classes (e.g., 4 for {d, u, r, l})
    Chans: number of electrodes/channels`
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

    # Block2: Separable Convolution``
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
def load_all_data(root_folder, label_list=['d', 'u', 'r', 'l'], time_points=128):
    """
    Function to load and process EEG data from CSV files.
    """
    all_X = []
    all_y = []
    
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
    
    # Walk through all directories and files in the root folder
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_processed.csv'):
                # Full path to the CSV file
                filepath = os.path.join(subdir, file)
                
                # Load CSV file into DataFrame
                df = pd.read_csv(filepath)
                
                # Extract label from the beginning of filename
                label = file[0]  # first character is the label
                
                if label not in label_list:
                    continue
                
                # Extract the channel signals
                # Ensure there are enough samples, otherwise, continue or pad
                signal = df[channels_of_interest].values[:time_points, :].T
                
                if signal.shape[1] < time_points:
                    continue  # or pad here
                
                # Expand dimensions to 1 x 5 x time_points
                signal = np.expand_dims(signal, axis=0)
                
                # Append to list
                all_X.append(signal)
                all_y.append(label)
    
    # Convert lists to numpy arrays
    all_X = np.array(all_X)  # shape -> (NumTrials, 1, 5, time_points)
    all_y = np.array(all_y)
    
    print('Data loaded successfully.')
    
    return all_X, all_y

# --------------------------------------
# 3) Train EEGNet
# --------------------------------------
def main():
    root_folder = '/processed'
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