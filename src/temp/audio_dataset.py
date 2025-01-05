import torchaudio
from torch.utils.data import Dataset
from torchaudio import load
import torch

# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, audio_paths, transform=None):
        self.audio_paths = audio_paths
        self.transform = transform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load the audio file
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = load(audio_path)

        # Resample if needed (assuming your dataset uses 16000 Hz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Downmix to mono if waveform has multiple channels
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            waveform = self.transform(waveform)
        
        # Placeholder label (e.g., 0); replace this with actual labels if available
        label = 0

        return waveform, label