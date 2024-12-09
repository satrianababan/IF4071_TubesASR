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
        if self.transform:
            waveform = self.transform(waveform)
        transcription = segment['transcription']
        return waveform, transcription