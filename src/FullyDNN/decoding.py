# Decoding using Greedy Decoder
def greedy_decode(output):
    decoded = []
    for timestep in output.argmax(dim=2):  # Get max indices
        decoded.append("".join([chr(c + 97) for c in timestep]))  # Convert to char
    return decoded

# Decoding using Beam Search
def beam_search_decode(output, beam_width=3):
    T, V = output.shape  # T: timesteps, V: vocabulary size
    beams = [([], 0)]  # List of tuples (sequence, log_prob)
    
    for t in range(T):
        new_beams = []
        for seq, log_prob in beams:
            # Get probabilities for the current timestep
            for v in range(V):
                new_seq = seq + [v]
                new_log_prob = log_prob + output[t, v].item()
                new_beams.append((new_seq, new_log_prob))
        
        # Sort by log probability and keep top `beam_width` beams
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
    
    # Select the sequence with the highest probability
    best_seq = beams[0][0]
    return best_seq