import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam  # Use legacy only if needed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Activation, AveragePooling2D, 
    SeparableConv2D, DepthwiseConv2D, Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers.schedules import CosineDecay


early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=100, alpha=0.0001
)

def EEGNet(nb_classes, Chans=5, Samples=128, dropoutRate=0.3):
    input_shape = (Chans, Samples, 1)
    inputs = Input(shape=input_shape)

    # Block 1: Temporal Convolution
    x = Conv2D(24, (32, 1), padding='same', use_bias=False)(inputs)  
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Depthwise Convolution
    x = DepthwiseConv2D((4, 1), use_bias=False, depth_multiplier=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Pooling & Dropout
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

    # Fully Connected Layer with Higher Regularization
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.005))(x)  
    x = Dropout(dropoutRate)(x)

    outputs = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def add_noise(X, noise_level=0.1):
    """
    Add Gaussian noise to EEG data to improve generalization.
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def augment_data(X):
    X_shifted = np.roll(X, shift=5, axis=2)  # Time shift
    X_flipped = np.flip(X, axis=1)  # Flip channels
    return np.concatenate([X, X_shifted, X_flipped], axis=0)



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

            all_X.append(signal)
            all_y.append(label)

    all_X = np.array(all_X)
    all_y = np.array(all_y)

    return all_X, all_y

def preprocess_data(X_train, X_val):
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten to apply scaler
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

    return X_train_scaled, X_val_scaled


def main():
    root_folder = 'processed'
    X, y = load_all_data(root_folder)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\n**Class Distribution**")
    print(np.unique(y_encoded, return_counts=True))  # Debugging label encoding

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )

    # Normalize data (AFTER splitting)
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

    # Apply noise augmentation only to training data
    X_train_scaled = add_noise(X_train_scaled, noise_level=0.1)

    nb_classes = len(np.unique(y_encoded))
    chans, samples = X.shape[1], X.shape[2]

    model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples)

    # Define CosineDecay Learning Rate Scheduler
    lr_schedule = CosineDecay(
    initial_learning_rate=0.001, decay_steps=100, alpha=0.0001
)

    # Use inside Adam optimizer
    optimizer = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Train model
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    X_train = augment_data(X_train)
    y_train = np.tile(y_train, 3)

    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=200,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping],
        class_weight=class_weights_dict,
    )

    # Evaluate model
    scores = model.evaluate(X_val_scaled, y_val, verbose=0)
    print("Validation Loss: %.4f" % scores[0])
    print("Validation Accuracy: %.4f" % scores[1])

if __name__ == "__main__":
    main()