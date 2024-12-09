from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    waveforms = [item[0] for item in batch]  # Extract waveforms
    transcriptions = [item[1] for item in batch]  # Extract transcriptions

    # Apply squeeze to remove extra dimensions
    waveforms = [waveform.squeeze(0) for waveform in waveforms]  # Remove [1, n_mels, seq_len] -> [n_mels, seq_len]

    # Pad waveforms to the length of the longest waveform in the batch
    max_len = max(waveform.size(-1) for waveform in waveforms)  # Longest sequence length
    padded_waveforms = [
        torch.nn.functional.pad(waveform, (0, max_len - waveform.size(-1))) for waveform in waveforms
    ]
    waveforms = torch.stack(padded_waveforms)  # Stack into a batch tensor [batch_size, n_mels, seq_len]

    # Convert transcriptions to numerical indices (e.g., ASCII or custom vocabulary)
    vocab = {ch: idx for idx, ch in enumerate("abcdefghijklmnopqrstuvwxyz _")}  # Example vocab
    target_transcriptions = [
        torch.tensor([vocab[ch] for ch in transcription if ch in vocab]) 
        for transcription in transcriptions
    ]
    
    # Pad transcriptions to the length of the longest transcription
    max_trans_len = max(len(t) for t in target_transcriptions)
    target_transcriptions = [
        torch.nn.functional.pad(t, (0, max_trans_len - len(t)), value=0) for t in target_transcriptions
    ]
    target_transcriptions = torch.stack(target_transcriptions)
    
    return waveforms, target_transcriptions