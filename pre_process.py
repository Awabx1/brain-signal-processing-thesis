import os
import pandas as pd

# Define the time intervals you want to extract for each class label
CLASS_INTERVALS = {
    'i': (11, 21),  # idle
    'r': (15, 25),  # reading
    'u': (3, 13),   # up
    'd': (0, 10),   # down
    'l': (0, 10)    # left-right
}

def process_participant(participant_folder):
    # Folder that contains the original files
    csv_file_path = f'original/{participant_folder}'
    # Folder where processed files will be saved
    processed_folder = f'processed/{participant_folder}'
    os.makedirs(processed_folder, exist_ok=True)

    # Adjust this list to keep only columns you truly need!
    # Below is a “kitchen-sink” example that includes raw EEG,
    # motion, power bands, etc. Remove/comment out as needed.
    columns_to_keep = [
        # Basic timestamps
        'Timestamp', 'OriginalTimestamp',
        # We'll add 'RelativeTimestamp' manually after creation
        # (see below)

        # EEG channels
        'EEG.Counter', 'EEG.Interpolated',
        'EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4',

        # Overall raw signal quality and battery
        'EEG.RawCq', 'EEG.Battery', 'EEG.BatteryPercent',

        # Marker columns (if used for labeling or timing)
        'MarkerIndex', 'MarkerType', 'MarkerValueInt', 'EEG.MarkerHardware',

        # Channel quality metrics
        'CQ.AF3', 'CQ.T7', 'CQ.Pz', 'CQ.T8', 'CQ.AF4', 'CQ.Overall',

        # Motion (for artifact detection, optional)
        'MOT.CounterMems', 'MOT.InterpolatedMems',
        'MOT.Q0', 'MOT.Q1', 'MOT.Q2', 'MOT.Q3',
        'MOT.AccX', 'MOT.AccY', 'MOT.AccZ',
        'MOT.MagX', 'MOT.MagY', 'MOT.MagZ',

        # Power bands (optional, if you want to inspect or use them)
        'POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma',
        'POW.T7.Theta', 'POW.T7.Alpha', 'POW.T7.BetaL', 'POW.T7.BetaH', 'POW.T7.Gamma',
        'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma',
        'POW.T8.Theta', 'POW.T8.Alpha', 'POW.T8.BetaL', 'POW.T8.BetaH', 'POW.T8.Gamma',
        'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma',

        # EQ columns (if you want to track sampling/quality)
        'EQ.SampleRateQuality', 'EQ.OVERALL', 'EQ.AF3', 'EQ.T7', 'EQ.Pz', 'EQ.T8', 'EQ.AF4'
    ]

    for file in os.listdir(csv_file_path):
        if not file.endswith('.md.pm.bp.csv'):
            continue

        class_prefix = file.split('_')[0]  # e.g. "d1"
        class_code = class_prefix[0].lower()  # "d", "i", etc.
        if class_code not in CLASS_INTERVALS:
            continue

        (start_sec, end_sec) = CLASS_INTERVALS[class_code]
        csv_path = os.path.join(csv_file_path, file)

        # Adjust skiprows if needed depending on your CSV format
        df = pd.read_csv(csv_path, skiprows=[0], header=0)
        if 'Timestamp' not in df.columns:
            print(f"[WARNING] No 'Timestamp' in {file}, skipping.")
            continue

        # Create a relative timestamp (start of file = 0)
        df['RelativeTimestamp'] = df['Timestamp'] - df['Timestamp'].iloc[0]

        # Time filter using the RelativeTimestamp
        mask = (df['RelativeTimestamp'] >= start_sec) & (df['RelativeTimestamp'] <= end_sec)
        df_slice = df[mask].copy()

        # Keep only relevant columns (drop everything else).
        # Some columns_to_keep might not exist in this file, so filter:
        existing_cols = [c for c in columns_to_keep if c in df_slice.columns]
        # Add 'RelativeTimestamp' if not included above but you still want it
        if 'RelativeTimestamp' not in existing_cols:
            existing_cols.append('RelativeTimestamp')

        df_slice = df_slice[existing_cols]

        # Output path for processed CSV
        out_path = os.path.join(processed_folder, file.replace('.md.pm.bp.csv','_processed.csv'))

        # Save the slice with columns you kept
        df_slice.to_csv(out_path, index=False)
        print(f"[INFO] Processed file: {file}, rows in slice = {len(df_slice)} → {out_path}")


# Example usage for two participants:
process_participant('awan1')
process_participant('abdulhadi1')
