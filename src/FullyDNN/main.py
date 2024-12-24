from parsing.parse_txt import parse_transcription_file
from parsing.parse_audio import slice_audio
from collate import collate_fn
from decoding import greedy_decode, beam_search_decode

import json
import torch
import torch.nn as nn
import editdistance
import matplotlib.pyplot as plt

from audio_dataset import AudioDataset
from asr_model import ASRModel
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking, AmplitudeToDB
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

# Load text transcriptions and audio paths
transcription_file = "././dataset_testing/TXT/A0102_S0001_0_G1357.txt"
segments = parse_transcription_file(transcription_file)

# Save the segments to a JSON file
with open("././jsonAudio/segments.json", "w") as file:
    json.dump(segments, file, indent=4)
    
# Load the segments from the JSON file
transcription_file = "./././dataset_testing/TXT/A0102_S0001_0_G1357.txt"
audio_file = "./././dataset_testing/WAV/A0102_S0001_0_G1357.wav"
if audio_file:
    output_dir = "sliced_audio_segments"
    segments = parse_transcription_file(transcription_file)
    audio_segments = slice_audio(audio_file, segments, output_dir)
else:
    print("Audio file not found")

# Preprocessing Function
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024),
    AmplitudeToDB(stype='power', top_db=80),  # Convert amplitude to decibels
    FrequencyMasking(freq_mask_param=15),  # Apply frequency masking
    TimeMasking(time_mask_param=35),  # Apply time masking
)

# Initialize Dataset, DataLoader
audio_paths = [segment['path'] for segment in audio_segments]
dataset = AudioDataset(audio_segments, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Visualization and Decoding
# index_to_char = {
#     0: '',    # Padding
#     1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
#     10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 
#     18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y',
#     26: 'Z'  # Letter Z, or you can reassign 26 to a space ' ' if desired
# }

# Convert indices to text
# def tensor_to_text(tensor, index_to_char):
#     texts = []
#     for row in tensor:
        # Print raw tensor row for debugging
        # print("Raw Tensor Row: ", row.tolist())

        # Filter out padding (0) and decode indices
        # decoded_chars = [index_to_char.get(idx.item(), '') for idx in row if idx != 0]
        # text = ''.join(decoded_chars)
        
        # Debug individual character decoding
        # print("Decoded Characters: ", decoded_chars)

        # Remove redundant spaces and strip any leading/trailing spaces
        # text = text.replace('  ', ' ').strip()
        # texts.append(text)
        # print("Decoded Text: ", text)
    # return texts

# Loop through DataLoader
for i, batch in enumerate(dataloader):
    waveforms, transcriptions = batch  # transcriptions: tensor with indices

    # Decode transcriptions
    # decoded_texts = tensor_to_text(transcriptions, index_to_char)

    # Print decoded text
    # for j, text in enumerate(decoded_texts):
    #     print(f"Decoded Text {j + 1}: {text}")

    # Visualize Mel-Spectrogram for the first waveform
    for waveform in waveforms:
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
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for batch in dataloader:
        waveforms, transcriptions = batch
        optimizer.zero_grad()

        # Forward Propagation
        outputs = model(waveforms)  # Shape: [batch_size, seq_len, output_dim]
        # print("outputs: ", outputs)

        # Compute input lengths (actual sequence lengths in waveforms)
        input_lengths = torch.tensor([waveform.size(-1) for waveform in waveforms])  # Shape: [batch_size]
        # print("input_lengths: ", input_lengths)

        # Compute target lengths
        target_lengths = torch.tensor([len(t) for t in transcriptions])  # Shape: [batch_size]
        # print("target_lengths: ", target_lengths)

        # Compute CTC Loss
        loss = criterion(
            outputs.log_softmax(2).transpose(0, 1),  # Shape: [seq_len, batch_size, num_classes]
            transcriptions,                          # Shape: [batch_size, max_target_len]
            input_lengths,                           # Shape: [batch_size]
            target_lengths                           # Shape: [batch_size]
        )

        # Backward Propagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    
# TODO: Implement the decoding function to decode the model predictions into text
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
            beam_decoded = beam_search_decode(output, beam_width=5)
            # print(f"Beam Decoded Text: {beam_decoded}")

# Save decoded results to a file
with open("decoded_texts.json", "w") as file:
    json.dump(decoded_texts, file, indent=4)

print("Decoding complete. Results saved to decoded_texts.json.")