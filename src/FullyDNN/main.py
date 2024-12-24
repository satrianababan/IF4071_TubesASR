from parsing.parse_txt import parse_transcription_file
from parsing.parse_audio import slice_audio
from collate import collate_fn
from decoding import greedy_decode, beam_search_decode

import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from audio_dataset import AudioDataset
from asr_model import ASRModel
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking, AmplitudeToDB
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
import Levenshtein

# Preprocessing Function
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024),
    AmplitudeToDB(stype='power', top_db=80),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35),
)

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

#     # Visualize Soundwave and Mel-Spectrogram for the first waveform
    for waveform in padded_audio:
        # Visualize raw waveform
        plt.figure(figsize=(10, 3))
        plt.plot(waveform.squeeze().numpy())
        plt.title("Soundwave")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

        # Visualize Mel-Spectrogram
        plt.figure(figsize=(10, 3))
        plt.imshow(waveform.squeeze().numpy(), aspect='auto', origin='lower')        
        plt.colorbar()
        plt.title("Mel-spectrogram")
        plt.show()
    break  # Exit after processing one batch

# Model Initialization
input_dim = 128  # Mel-spectrogram features
hidden_dim = 256 # Hidden dimension of RNN
output_dim = 28  # 26 alphabets + space + blank token for CTC
model = ASRModel(input_dim, hidden_dim, output_dim)
print(model)

# Training Loop
criterion = CTCLoss()
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

for epoch in range(10):
    model.train()
    for batch in dataloader:
        padded_audio, labels = batch
        optimizer.zero_grad()

        # Forward Propagation
        outputs = model(padded_audio)
        # print("outputs: ", outputs)

        # Compute input lengths (actual sequence lengths in waveforms)
        input_lengths = torch.tensor([waveform.size(-1) for waveform in padded_audio])
        # print("input_lengths: ", input_lengths)

        # Compute target lengths
        if labels.dim() == 2:
            target_lengths = torch.tensor([t.size(0) for t in labels])  # Shape: [batch_size]
        elif labels.dim() == 1:
            # If it's 1D (a single transcription), handle it differently
            target_lengths = torch.tensor([labels.size(0)])
        else:
            raise ValueError("Unexpected dimensions for transcriptions tensor.")
        # print("target_lengths: ", target_lengths)
        
        
# Compute CTC Loss
def calculate_cer(decoded_texts, ground_truths):
    cer_values = []
    for decoded, ground_truth in zip(decoded_texts, ground_truths):
        # Compute CER using edit distance
        distance = Levenshtein.distance(decoded, ground_truth)
        cer = distance / len(ground_truth) if ground_truth else 1.0  # Avoid division by zero
        cer_values.append(cer)
    return sum(cer_values) / len(cer_values)

# Path to the .txt file
file_path = "./././dataset_testing/TXT/A0102_S0001_0_G1357.txt"

# Extract ground truths use parse_transcription_file function
transcription_segments = parse_transcription_file(file_path)
print("Transcription Segments:", transcription_segments)

# Extract ground truths
ground_truths = [segment['transcription'] for segment in transcription_segments]
print("Ground Truths:", ground_truths)
    
# Evaluation and Decoding
model.eval()
decoded_texts = []

with torch.no_grad():
    for batch in dataloader:
        waveforms, _ = batch  # Ignore transcriptions during decoding
        outputs = model(waveforms)  # Shape: [batch_size, seq_len, output_dim]
        # print("outputs: ", outputs)
        
        # Greedy Decoding
        greedy_decoded = greedy_decode(outputs)
        decoded_texts.extend(greedy_decoded)

        # Beam Search Decoding
        for output in outputs:
            beam_decoded = beam_search_decode(output, beam_width=10)
            # print(f"Beam Decoded Text: {beam_decoded}")
            
average_cer = calculate_cer(decoded_texts, ground_truths)
print(f"Average CER in percentage: {average_cer * 10:.2f}%")

# Save decoded results to a file
with open("decoded_texts.json", "w") as file:
    json.dump(decoded_texts, file, indent=4)

print("Decoding complete. Results saved to decoded_texts.json.")