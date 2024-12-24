import torchaudio
from torch.utils.data import Dataset

# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, audio_segments, transform=None):
        self.audio_segments = audio_segments
        self.transform = transform

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        segment = self.audio_segments[idx]
        waveform, sample_rate = torchaudio.load(segment['path'])
        
        # Resample to 16kHz for consistency
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Apply transformation (feature extraction)
        if self.transform:
            waveform = self.transform(waveform)
        transcription = segment['transcription']
        return waveform, transcription