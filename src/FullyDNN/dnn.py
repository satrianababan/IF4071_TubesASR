from audioDataset import AudioDataset
from asrModel import ASRModel
import torch
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn

# Preprocessing Function
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=128),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35),
)

# Initialize Dataset, DataLoader
audio_paths = ["audio1.wav", "audio2.wav"]  # Replace with real paths
transcriptions = ["hello", "world"]  # Replace with actual transcriptions
dataset = AudioDataset(audio_paths, transcriptions, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model Initialization
input_dim = 128  # Mel-spectrogram features
hidden_dim = 256
output_dim = 28  # 26 alphabets + space + blank token for CTC
model = ASRModel(input_dim, hidden_dim, output_dim)

# Training Loop
criterion = CTCLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(10):  # Adjust as needed
    model.train()
    for batch in dataloader:
        waveforms, transcriptions = batch
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(waveforms)
        
        # Compute CTC Loss
        input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in transcriptions])
        loss = criterion(outputs.log_softmax(2), transcriptions, input_lengths, target_lengths)

        # Backward Pass
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")