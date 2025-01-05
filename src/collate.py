from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    # Extract audio data (Mel-Spectrograms) and labels
    audio_data = [item[0].squeeze(0) for item in batch]  # Remove channel dimension
    labels = [item[1] for item in batch]  # Labels

    # Determine the maximum time dimension in the batch
    max_time = max(audio.shape[-1] for audio in audio_data)

    # Pad each tensor to the maximum time dimension
    padded_audio = torch.stack([
        torch.nn.functional.pad(audio, (0, max_time - audio.shape[-1]))
        for audio in audio_data
    ])

    # Convert labels to a tensor
    labels = torch.tensor(labels)

    return padded_audio, labels