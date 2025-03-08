import os
import pandas as pd
import numpy as np

def load_all_data(root_folder, label_list=['d', 'u', 'r', 'l'], time_points=128):
    """
    Function to load and process EEG data from CSV files.
    """
    all_X = []
    all_y = []
    channels_of_interest = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
    
    # Walk through the directories and files in the root folder
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            # Check if file ends with '_processed.csv'
            if file.endswith('_processed.csv'):
                # Full path to the CSV file
                filepath = os.path.join(subdir, file)
                # Extract label and instance number from the beginning of the filename
                label = file[0]  # First character is the label
                instance = file[1]  # Second character is the instance digit
                
                # Ensure we're working with valid labels
                if label not in label_list or not instance.isdigit():
                    continue
                
                # print(f"Processing file: {filepath}")

                # Load CSV file into DataFrame
                df = pd.read_csv(filepath)

                # Extract the channel signals
                signal = df[channels_of_interest].values.T
                
                # Check if there are enough samples and pad if necessary
                if signal.shape[1] < time_points:
                    # Optionally, log this event
                    print(f"Not enough data points in {file}, expected {time_points}, found {signal.shape[1]}")
                    continue  # Skip or, optionally, use np.pad to fill

                signal = signal[:, :time_points]  # Ensure the same time_points length

                # Expand dimensions to (1, 5, time_points)
                signal = np.expand_dims(signal, axis=-1)

                # Append to list
                all_X.append(signal)
                all_y.append(label)

    # Convert lists to numpy arrays
    all_X = np.array(all_X)  # Shape -> (NumTrials, 5, time_points, 1)
    all_y = np.array(all_y)
    
    print(f'Total files loaded: {len(all_X)}')
    return all_X, all_y

# Ensure the correct path to the root folder
root_folder = 'processed'
X, y = load_all_data(root_folder)