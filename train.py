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

from tensorflow.keras.layers import AveragePooling2D

def EEGNet(nb_classes, Chans=5, Samples=128, dropoutRate=0.7):
    input_shape = (Chans, Samples, 1)
    inputs = Input(shape=input_shape)

    # Block 1: Temporal Convolution
    x = Conv2D(8, (32, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Depthwise Convolution (Fix: No 2D pool over Chans)
    x = DepthwiseConv2D((4, 1), use_bias=False, depth_multiplier=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #Pool only if dimension is large enough
    pool_size_1 = min(2, x.shape[1])  # Ensures we don’t pool below 1
    x = AveragePooling2D((pool_size_1, 1))(x)
    x = Dropout(dropoutRate)(x)

    # Separable Convolution
    x = SeparableConv2D(16, (8, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    pool_size_2 = min(2, x.shape[1])
    x = AveragePooling2D((pool_size_2, 1))(x)  
    x = Dropout(dropoutRate)(x)

    # Fully Connected
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropoutRate)(x)
    outputs = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)



def load_all_data(root_folder, label_list=['d', 'u', 'r', 'l'], time_points=128):
    """
    Load EEG Data & Apply Feature Scaling
    """
    all_X = []
    all_y = []
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']

    print(f"\n🔍 Searching for CSV files in: {root_folder}\n")
    file_count, skipped_files = 0, 0
    loaded_shapes = set()

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if not file.endswith('_processed.csv'):
                continue
            
            filepath = os.path.join(subdir, file)
            label = file[0]  
            
            if label not in label_list:
                skipped_files += 1
                continue
            
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                skipped_files += 1
                continue
            
            loaded_shapes.add(df.shape)
            if not all(col in df.columns for col in channels_of_interest):
                skipped_files += 1
                continue

            signal = df[channels_of_interest].values[:time_points, :].T  
            if signal.shape[1] < time_points:
                skipped_files += 1
                continue

            # Standardize EEG signals
            scaler = StandardScaler()
            signal = scaler.fit_transform(signal)

            signal = np.expand_dims(signal, axis=-1)  # Shape: (Chans, Samples, 1)
            all_X.append(signal)
            all_y.append(label)
            file_count += 1

    print(f"\n✅ Total valid CSV files loaded: {file_count}")
    print(f"🚫 Total files skipped: {skipped_files}")
    print(f"📊 Unique dataset shapes encountered: {loaded_shapes}")

    if len(all_X) == 0:
        raise RuntimeError("❌ No valid CSV files found.")

    all_X = np.array(all_X)  
    all_y = np.array(all_y)
    print(f"\n📊 Final dataset shape: X={all_X.shape}, y={all_y.shape}\n")
    
    return all_X, all_y

def main():
    root_folder = 'processed'
    X, y = load_all_data(root_folder)

    # Encode labels (e.g., 'd, u, r, l' -> 0,1,2,3)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )

    # Define Model Parameters
    nb_classes = len(np.unique(y_encoded))
    chans, samples = X.shape[1], X.shape[2]

    model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0005),  
        metrics=['accuracy']
    )

    # Train Model
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=100,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Evaluate Model
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("Validation Loss: %.4f" % scores[0])
    print("Validation Accuracy: %.4f" % scores[1])

if __name__ == '__main__':
    main()