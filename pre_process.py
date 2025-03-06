import os
import pandas as pd

# Dictionary of time frames for each participant, label, and instance.
time_frames = {
    "abdulhadi1": {
        'r': [
            (27, 37), (25, 35), (34, 44), (25, 35), (0, 10),
            (0, 10), (0, 10), (0, 10), (5, 15), (0, 10), (15, 25)
        ],
        'u': [
            (30, 40), (18, 28), (0, 10), (0, 10), (0, 10),
            (0, 10), (15, 25), (0, 10), (0, 10), (12, 22), (8, 18)
        ],
        'd': [
            (20, 30), (0, 10), (8, 18), (9, 19), (0, 10),
            (0, 10), (65, 75), (20, 30), (0, 10), (2, 12), (8, 18)
        ],
        'l': [
            (50, 60), (0, 10), (0, 10), (0, 10), (0, 10),
            (0, 10), (7, 17), (0, 10), (0, 10), (3, 13), (10, 20)
        ]
    },
    "awan1": {
        'r': [
            (0, 10), (1, 11), (6, 16), (0, 10), (0, 10),
            (0, 10), (5, 15), (0, 10), (0, 10), (0, 10), (2, 12)
        ],
        'u': [
            (0, 10), (30, 40), (0, 10), (0, 10), (0, 10),
            (0, 10), (0, 10), (0, 10), (2, 12), (2, 12), (0, 10)
        ],
        'd': [
            (0, 10), (10, 20), (0, 10), (0, 10), (1, 11),
            (3, 13), (1, 11), (0, 10), (2, 12), (0, 10), (0, 10)
        ],
        'l': [
            (0, 10), (8, 18), (51, 61), (0, 10), (0, 10),
            (0, 10), (0, 10), (1, 11), (0, 10), (0, 10), (1, 11)
        ]
    },
    "rafay1": {
        'r': [
            (2, 12), (10, 20), (24, 34), (5, 15), (0, 10),
            (0, 10), (6, 16), (0, 10), (0, 10), (2, 12), (0, 10)
        ],
        'u': [
            (4, 14), (0, 10), (0, 10), (50, 60), (0, 10),
            (0, 10), (10, 20), (0, 10), (39, 49), (0, 10), (0, 10)
        ],
        'd': [
            (0, 10), (0, 10), (8, 18), (42, 52), (0, 10),
            (1, 11), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10)
        ],
        'l': [
            (0, 10), (3, 13), (31, 41), (3, 13), (0, 10),
            (5, 15), (0, 10), (4, 14), (1, 11), (0, 10), (3, 13)
        ]
    }
}

def process_participant(participant_folder):
    """
    Processes all .md.pm.bp.csv files in original/<participant_folder> by:
      • Identifying the label and instance from the filename, e.g. r1_..., d2_...
      • Using the corresponding (start_sec, end_sec) from time_frames
      • Keeping only specific columns
      • Saving the sliced file to processed/<participant_folder>
    """
    csv_file_path = f'original/{participant_folder}'
    processed_folder = f'processed/{participant_folder}'
    os.makedirs(processed_folder, exist_ok=True)

    # Columns to keep (customize as needed):
    columns_to_keep = [
        # Basic timestamps
        'Timestamp', 'OriginalTimestamp',
        # We'll add 'RelativeTimestamp' manually after creation
        # EEG channels
        'EEG.Counter', 'EEG.Interpolated',
        'EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4',
        # Overall raw signal quality and battery
        'EEG.RawCq', 'EEG.Battery', 'EEG.BatteryPercent',
        # Channel quality metrics
        'CQ.AF3', 'CQ.T7', 'CQ.Pz', 'CQ.T8', 'CQ.AF4', 'CQ.Overall',
        # Motion
        'MOT.CounterMems', 'MOT.InterpolatedMems',
        'MOT.Q0', 'MOT.Q1', 'MOT.Q2', 'MOT.Q3',
        'MOT.AccX', 'MOT.AccY', 'MOT.AccZ',
        'MOT.MagX', 'MOT.MagY', 'MOT.MagZ',
        # Power bands (optional)
        'POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma',
        'POW.T7.Theta', 'POW.T7.Alpha', 'POW.T7.BetaL', 'POW.T7.BetaH', 'POW.T7.Gamma',
        'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma',
        'POW.T8.Theta', 'POW.T8.Alpha', 'POW.T8.BetaL', 'POW.T8.BetaH', 'POW.T8.Gamma',
        'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma',
        # EQ columns
        'EQ.SampleRateQuality', 'EQ.OVERALL', 'EQ.AF3', 'EQ.T7', 'EQ.Pz', 'EQ.T8', 'EQ.AF4'
    ]

    # Get that participant's label→intervals mapping from the time_frames dictionary.
    if participant_folder not in time_frames:
        print(f"[WARNING] No time frames for {participant_folder}. Skipping.")
        return
    participant_intervals = time_frames[participant_folder]

    for file in os.listdir(csv_file_path):
        # Only process .md.pm.bp.csv files
        if not file.endswith('.md.pm.bp.csv'):
            continue
        
        # Example filename pattern: "r1_something.md.pm.bp.csv"
        # We assume:
        #   label = file[0] (i.e. 'r', 'u', 'd', 'l')
        #   index = int(file[1]) - 1 to match the time_frames list index
        try:
            label = file[0].lower()  # 'r', 'u', 'd', or 'l'
            instance_num = int(file[1])  # e.g. "1", "2", ...
        except (ValueError, IndexError):
            print(f"[WARNING] Could not parse label/instance from {file}. Skipping.")
            continue
        
        if label not in participant_intervals:
            print(f"[WARNING] Label '{label}' not found in time_frames for {participant_folder}. Skipping {file}.")
            continue
        
        # time_frames are zero-indexed, but filenames have 1–11. So we do instance_num - 1.
        instance_idx = instance_num - 1
        if instance_idx < 0 or instance_idx >= len(participant_intervals[label]):
            print(f"[WARNING] Instance index out of range for {file} (instance={instance_num}). Skipping.")
            continue

        (start_sec, end_sec) = participant_intervals[label][instance_idx]
        csv_path = os.path.join(csv_file_path, file)

        # Read CSV; adjust skiprows if needed
        df = pd.read_csv(csv_path, skiprows=[0], header=0)
        if 'Timestamp' not in df.columns:
            print(f"[WARNING] No 'Timestamp' in {file}, skipping.")
            continue

        # Create a relative timestamp (start of file = 0)
        df['RelativeTimestamp'] = df['Timestamp'] - df['Timestamp'].iloc[0]

        # Filter rows by the (start_sec, end_sec) interval
        mask = (df['RelativeTimestamp'] >= start_sec) & (df['RelativeTimestamp'] <= end_sec)
        df_slice = df[mask].copy()

        # Keep only the relevant columns that actually exist in this file
        existing_cols = [c for c in columns_to_keep if c in df_slice.columns]
        if 'RelativeTimestamp' not in existing_cols:
            existing_cols.append('RelativeTimestamp')
        df_slice = df_slice[existing_cols]

        # Output path for processed CSV
        out_name = file.replace('.md.pm.bp.csv', '_processed.csv')
        out_path = os.path.join(processed_folder, out_name)
        df_slice.to_csv(out_path, index=False)
        print(f"[INFO] Processed file: {file}, rows in slice = {len(df_slice)} → {out_path}")


# Example usage for the three participants:
process_participant('awan1')
process_participant('abdulhadi1')
process_participant('rafay1')