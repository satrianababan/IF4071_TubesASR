import torch
import torch.nn.functional as F
import heapq

def greedy_decode(output, blank_label=27, collapse_repeated=True, vocab=None):
    if vocab is None:
        # Default to mapping 'a'-'z' and space, blank token as 27
        vocab = {i: chr(97 + i) for i in range(26)}
        vocab[26] = ' '  # Space
        vocab[blank_label] = ''  # Blank token maps to empty string

    # Get the most likely index at each timestep
    arg_maxes = torch.argmax(F.log_softmax(output, dim=2), dim=2)  # [batch_size, seq_len]

    decoded_batch = []
    for batch_indices in arg_maxes:
        decoded = []
        prev_idx = None
        for idx in batch_indices:
            idx_value = idx.item()

            # Skip blank tokens
            if idx_value == blank_label:
                continue

            # Collapse repeated characters if enabled
            if collapse_repeated and idx_value == prev_idx:
                continue

            decoded.append(vocab.get(idx_value, ''))  # Map index to character
            prev_idx = idx_value

        decoded_batch.append("".join(decoded))  # Join characters into a string

    return decoded_batch


def beam_search_decode(output, beam_width=3, blank_label=27, vocab=None):
    if vocab is None:
        # Default to mapping 'a'-'z' and space, blank token as 27
        vocab = {i: chr(97 + i) for i in range(26)}
        vocab[26] = ' '  # Space
        vocab[blank_label] = ''  # Blank token maps to empty string

    seq_len, vocab_size = output.size()
    log_probs = F.log_softmax(output, dim=1)  # Convert logits to log probabilities

    # Priority queue for beam search
    beams = [(0, [])]  # List of tuples (cumulative log probability, sequence)

    for t in range(seq_len):
        new_beams = []
        for log_prob, seq in beams:
            for i in range(vocab_size):
                new_seq = seq + [i]
                new_log_prob = log_prob + log_probs[t, i].item()
                new_beams.append((new_log_prob, new_seq))

        # Keep top `beam_width` beams
        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0])

    # Choose the beam with the highest probability
    best_seq = beams[0][1]

    # Convert indices to characters
    decoded = []
    prev_idx = None
    for idx in best_seq:
        if idx == blank_label:
            continue  # Skip blank tokens
        if idx != prev_idx:  # Avoid repeated characters
            decoded.append(vocab.get(idx, ''))
        prev_idx = idx

    return "".join(decoded)