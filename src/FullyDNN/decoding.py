# Decoding using Greedy Decoder
def greedy_decode(output):
    # Apply argmax to get the most likely character index at each timestep
    timestep_indices = output.argmax(dim=2)  # Shape: [batch_size, seq_len]

    decoded_batch = []
    for indices in timestep_indices:
        # Convert indices to characters, excluding CTC blank token (assume blank token index = output_dim - 1)
        decoded = []
        prev_idx = None
        for idx in indices:
            if idx != prev_idx and idx != output.size(2) - 1:  # Skip repeated and blank indices
                decoded.append(chr(idx + 97))  # Assuming 'a' starts at index 0
            prev_idx = idx
        decoded_batch.append("".join(decoded))
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