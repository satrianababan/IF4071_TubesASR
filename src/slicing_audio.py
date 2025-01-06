import os
import numpy as np
from pydub import AudioSegment
from scipy.signal import resample

# Function to segment WAV and save each segment to a separate file
def slice_wav(wav_path, segments, output_dir, sampling_rate=16000):
    audio = AudioSegment.from_wav(wav_path)
    segmented_data = []
    total_segments = len(segments)

    for i, segment in enumerate(segments):
        start_ms = segment["start"] * 1000      # Convert seconds to milliseconds
        end_ms = segment["end"] * 1000          # Convert seconds to milliseconds
        text = segment["text"]                  # Get transcription

        # Extract the segment
        segment_audio = audio[start_ms:end_ms]

        # Save the segmented audio to the output directory
        filename = os.path.basename(wav_path).replace(".wav", f"_segment_{i+1}.wav")
        output_path = os.path.join(output_dir, filename)
        segment_audio.export(output_path, format="wav")

        # Convert to numpy array at the target sampling rate
        samples = np.array(segment_audio.get_array_of_samples())
        resampled = resample(samples, int(len(samples) * sampling_rate / segment_audio.frame_rate))

        segmented_data.append({"audio": resampled, "text": text, "filename": filename})

        # Print progress
        progress = ((i + 1) / total_segments) * 100
        print(f"Segment {i + 1}/{total_segments} processed. Progress: {progress:.2f}%")

    return segmented_data