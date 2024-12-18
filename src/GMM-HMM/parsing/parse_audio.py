import os
import torchaudio

# Function to slice the audio file based on the segments
def slice_audio(audio_path, segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    waveform, sample_rate = torchaudio.load(audio_path)

    audio_segments = []
    for i, segment in enumerate(segments):
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)
        audio_slice = waveform[:, start_sample:end_sample]

        # Save the sliced audio segment
        segment_path = os.path.join(output_dir, f"segment_{i + 1}.wav")
        torchaudio.save(segment_path, audio_slice, sample_rate)
        
        audio_segments.append({
            'path': segment_path,
            'transcription': segment['transcription']
        })
    return audio_segments