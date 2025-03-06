import os
import pandas as pd

CLASS_INTERVALS = {
   'i': (11, 21),   # idle
   'r': (15, 25),   # reading
   'u': (3, 13),    # up
   'd': (0, 10),    # down
   'l': (0, 10)     # left-right
}

def process_participant(participant_folder):
    
    csv_file_path=f'original/{participant_folder}'
    processed_folder = f"processed/{participant_folder}"
    os.makedirs(processed_folder, exist_ok=True)

    for file in os.listdir(csv_file_path):
        if not file.endswith('.md.pm.bp.csv'):
            continue

        class_prefix = file.split('_')[0]   # e.g. "d1"
        class_code = class_prefix[0].lower() # "d"
        if class_code not in CLASS_INTERVALS:
            continue

        (start_sec, end_sec) = CLASS_INTERVALS[class_code]

        csv_path = os.path.join(csv_file_path, file)
        
        # Try skiprows=[0] or skiprows=[0,1], depending on your file
        df = pd.read_csv(csv_path, skiprows=[0], header=0)
        
        if 'Timestamp' not in df.columns:
            print(f"[WARNING] No 'Timestamp' in {file}, skipping.")
            continue
        
        # Convert “absolute” timestamps to relative (start of file = 0)
        df['RelativeTimestamp'] = df['Timestamp'] - df['Timestamp'].iloc[0]
        
        # Time filter using the RelativeTimestamp
        mask = (df['RelativeTimestamp'] >= start_sec) & \
               (df['RelativeTimestamp'] <= end_sec)
        df_slice = df[mask]
        
        # If the slice is empty, you’ll just get headers and no rows
        out_path = os.path.join(processed_folder, file.replace('.md.pm.bp.csv','_processed.csv'))
        df_slice.to_csv(out_path, index=False)
        print(f"[INFO] Processed file: {file}, rows in slice = {len(df_slice)} → {out_path}")

# Example usage:
process_participant('awan1')
process_participant('abdulhadi1')