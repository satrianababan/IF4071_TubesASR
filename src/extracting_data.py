import os
import re
import numpy as np
from datasets import Dataset
from pydub import AudioSegment
from scipy.signal import resample

from slicing_audio import slice_wav
from parse_text import parse_txt_file

# Directory paths
wav_dir = "./dataset_testing/WAV"           # Directory containing original WAV files
txt_dir = "./dataset_testing/TXT"           # Directory containing TXT files
output_dir = "./sliced_audio_segments"      # Directory to save segmented WAV files
sampling_rate = 16000                       # Target sampling rate

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Prepare dataset
all_data = []

txt_files = [file for file in os.listdir(txt_dir) if file.endswith(".txt")]
total_files = len(txt_files)
processed_files = 0

print(f"Total files to process: {total_files}")

for file in txt_files:
    txt_path = os.path.join(txt_dir, file)
    wav_path = os.path.join(wav_dir, file.replace(".txt", ".wav"))

    if not os.path.exists(wav_path):
        continue

    # Parse the TXT file and segment the WAV
    segments = parse_txt_file(txt_path)
    segmented_wavs = slice_wav(wav_path, segments, output_dir, sampling_rate=sampling_rate)

    for segment_data in segmented_wavs:
        all_data.append({"audio": segment_data["audio"], "text": segment_data["text"], "filename": segment_data["filename"]})

    processed_files += 1
    progress = (processed_files / total_files) * 100
    print(f"Progress: {progress:.2f}%")

# Convert to WAV2VEC2 Dataset
hf_dataset = Dataset.from_dict({
    "audio": [x["audio"] for x in all_data],
    "text": [x["text"] for x in all_data],
    "filename": [x["filename"] for x in all_data]
})

# Save to disk (optional)
hf_dataset.save_to_disk("./results/WAV2VEC2_DATASET")
