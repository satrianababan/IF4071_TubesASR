import torchaudio
from torch.utils.data import Dataset

# Dataset Class
class AudioDataset(Dataset):
    # Constructor
    def __init__(self, audio_paths, transcriptions, transform=None):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.transform = transform

    # Length Method
    def __len__(self):
        return len(self.audio_paths)

    # Get Item Method
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        if self.transform:
            waveform = self.transform(waveform)
        transcription = self.transcriptions[idx]
        return waveform, transcription