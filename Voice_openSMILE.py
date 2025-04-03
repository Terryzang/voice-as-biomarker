import opensmile
import pandas as pd
import os

# Initialize OpenSMILE using eGeMAPSv02 feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals  # Select the feature level to extract
)

# Define a batch processing function
def batch_process_wav(input_folder, output_csv):
    """
    Iterate through all WAV files in the input_folder, extract OpenSMILE voice features,
    and save the results to a CSV file. The first column is the participant ID.

    Parameters:
    - input_folder: str, path to the folder containing WAV files
    - output_csv: str, path to the output CSV file
    """
    data = []  # Store all extracted features
    n = 1

    # Traverse all WAV files in the folder
    for filename in sorted(os.listdir(input_folder)):
        print(n)
        if filename.endswith(".wav"):
            # Extract participant ID (first three characters)
            participant_id = filename[:3]

            # Get full file path
            wav_path = os.path.join(input_folder, filename)

            # Process the audio file and extract features
            features = smile.process_file(wav_path)

            # Insert ID column
            features.insert(0, "ID", participant_id)

            # Add to list
            data.append(features)
            n += 1

    # Concatenate all data
    df = pd.concat(data, ignore_index=True)

    # Save as CSV
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. Saved to {output_csv}")

input_folder = f"E:\\"  # Folder containing WAV files
output_csv = f"E:\\.csv"  # Output CSV file path
batch_process_wav(input_folder, output_csv)
