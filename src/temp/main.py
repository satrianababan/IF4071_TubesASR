import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from parsing.parse_txt import parse_transcription_file
from parsing.parse_audio import slice_audio
from collate import collate_fn
from decoding import greedy_decode, beam_search_decode

import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

from audio_dataset import AudioDataset
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking, AmplitudeToDB
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Preprocessing Function
transform = None

def collect_audio_paths(directory):
    # Recursively find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac']  # Add other extensions as needed
    return [
        str(file) for file in Path(directory).rglob('*')
        if file.suffix.lower() in audio_extensions
    ]

# Initialize Dataset, DataLoader
path = "sliced_audio_segments"
audio_paths = collect_audio_paths(path)
print("Number of Audio Files:", len(audio_paths))
dataset = AudioDataset(audio_paths, transform=transform)
print("Dataset Length:", len(dataset))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, collate_fn=collate_fn)

# Loop through DataLoader
for i, batch in enumerate(dataloader):
    padded_audio, labels = batch
    print("Batch:", i)
    print("Padded Audio Shape:", padded_audio.shape)
    print("Labels:", labels)

#     # Visualize Soundwave and Mel-Spectrogram for the first waveform
    for waveform in padded_audio:
        # Visualize raw waveform
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(waveform.t().numpy())
        plt.title('Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        # Visualize Mel-Spectrogram
        plt.subplot(2, 1, 2)
        plt.imshow(MelSpectrogram()(waveform).log2().squeeze().numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.title("Mel-spectrogram")
        
        plt.tight_layout()
        plt.show()
    break  # Exit after processing one batch

# Perform inference
with torch.no_grad():
    logits = model(padded_audio).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    # Print transcription
    print("Transcription:", transcription)