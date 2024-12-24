import torch
import torch.nn.functional as F

def greedy_decode(output, blank_label=27, collapse_repeated=True):
    # Get argmax indices (most likely characters) along the class dimension
    arg_maxes = torch.argmax(F.softmax(output, dim=2), dim=2)  # Shape: [batch_size, seq_len]
    print(arg_maxes)
    decoded_batch = []

    for batch_indices in arg_maxes:  # Iterate through each sequence in the batch
        decoded = []
        prev_idx = None  # For collapsing repeated characters

        for i, idx in enumerate(batch_indices):
            idx_value = idx.item()

            # Skip blank token
            if idx_value == blank_label:
                continue

            # Skip repeated characters if collapse_repeated is enabled
            if collapse_repeated and i != 0 and idx_value == prev_idx:
                continue

            # Map index to character ('a' starts at index 0)
            decoded.append(chr(idx_value + 97))  # Convert to character
            prev_idx = idx_value  # Update previous index

        decoded_batch.append("".join(decoded))  # Join characters into a single string

    return decoded_batch

# Decoding using Beam Search Decoder
def beam_search_decode(output, beam_width=3):
    seq_len, vocab_size = output.size()  # Shape: [seq_len, vocab_size]
    beams = [([], 0)]  # List of tuples (sequence, log_prob)

    for t in range(seq_len):
        new_beams = []
        for seq, log_prob in beams:
            # Get probabilities for the current timestep
            probs = output[t].log_softmax(dim=0)  # Convert logits to log probabilities
            for v in range(vocab_size):
                new_seq = seq + [v]
                new_log_prob = log_prob + probs[v].item()
                new_beams.append((new_seq, new_log_prob))

        # Sort by log probability and keep top `beam_width` beams
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams

    # Select the sequence with the highest probability
    best_seq = beams[0][0]

    # Convert indices to characters
    decoded = []
    prev_idx = None
    for idx in best_seq:
        if idx != prev_idx and idx != vocab_size - 1:  # Skip repeated and blank indices
            decoded.append(chr(idx + 97))  # Assuming 'a' starts at index 0
        prev_idx = idx

    return "".join(decoded)