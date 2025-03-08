import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam  # Use Legacy Adam for M1/M2 Macs
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Activation, AveragePooling2D, 
    SeparableConv2D, DepthwiseConv2D, Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import AveragePooling2D

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

def EEGNet(nb_classes, Chans=5, Samples=128, dropoutRate=0.7):
    input_shape = (Chans, Samples, 1)
    inputs = Input(shape=input_shape)

    # Block 1: Temporal Convolution
    x = Conv2D(16, (64, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Depthwise Convolution
    x = DepthwiseConv2D((4, 1), use_bias=False, depth_multiplier=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    pool_size_1 = min(2, x.shape[1])
    x = AveragePooling2D((pool_size_1, 1))(x)
    x = Dropout(dropoutRate)(x)

    # Separable Convolution
    x = SeparableConv2D(32, (8, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    pool_size_2 = min(2, x.shape[1])
    x = AveragePooling2D((pool_size_2, 1))(x)  
    x = Dropout(dropoutRate)(x)

    # Fully Connected Layer
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropoutRate)(x)
    outputs = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)

def load_all_data(root_folder, label_list=['d', 'u', 'r', 'l'], time_points=128):
    all_X, all_y = [], []
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if not file.endswith('_processed.csv'):
                continue
            
            filepath = os.path.join(subdir, file)
            label = file[0]  
            
            if label not in label_list:
                continue
            
            try:
                df = pd.read_csv(filepath)
            except Exception:
                continue
            
            if not all(col in df.columns for col in channels_of_interest):
                continue

            signal = df[channels_of_interest].values[:time_points, :].T  
            if signal.shape[1] < time_points:
                continue

            scaler = StandardScaler()
            signal = scaler.fit_transform(signal)
            signal = np.expand_dims(signal, axis=-1)
            
            all_X.append(signal)
            all_y.append(label)

    all_X = np.array(all_X)
    all_y = np.array(all_y)
    
    return all_X, all_y

def main():
    root_folder = 'processed'
    X, y = load_all_data(root_folder)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )

    nb_classes = len(np.unique(y_encoded))
    chans, samples = X.shape[1], X.shape[2]

    model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),  
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=200,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping]
    )

    scores = model.evaluate(X_val, y_val, verbose=0)
    print("Validation Loss: %.4f" % scores[0])
    print("Validation Accuracy: %.4f" % scores[1])

if __name__ == '__main__':
    main()