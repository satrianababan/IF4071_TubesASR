import os
import numpy as np
import torchaudio
from sklearn.mixture import GaussianMixture
import Levenshtein

# --- Feature Extraction ---
def extract_mel_spectrogram(wav_path, n_mels=40):
    """Extract normalized Mel spectrogram features from an audio file."""
    waveform, sample_rate = torchaudio.load(wav_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels
    )
    mel_features = mel_spectrogram(waveform).squeeze(0).T  # Shape: [time, n_mels]
    # Normalize features
    mel_features = (mel_features - mel_features.mean(axis=0)) / mel_features.std(axis=0)
    return mel_features.numpy()

# --- HMM-GMM Model ---
class HMMGMM:
    def __init__(self, n_states, n_mixtures):
        self.n_states = n_states
        self.n_mixtures = n_mixtures
        self.gmms = [GaussianMixture(n_components=n_mixtures, covariance_type='diag', reg_covar=1e-4) for _ in range(n_states)]
        self.trans_probs = np.full((n_states, n_states), 1 / n_states)

    def train(self, X, labels):
        for state in range(self.n_states):
            state_data = X[labels == state]
            if len(state_data) < self.n_mixtures:  # Skip states with insufficient data
                print(f"Skipping state {state}: insufficient data ({len(state_data)} samples).")
                continue
            self.gmms[state].fit(state_data)
        self._update_trans_probs(labels)

    def _update_trans_probs(self, labels):
        counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(labels) - 1):
            counts[labels[i], labels[i + 1]] += 1
        # Apply Laplace smoothing to avoid zeros
        self.trans_probs = (counts + 1) / (counts.sum(axis=1, keepdims=True) + self.n_states)

    def decode(self, X):
        T, N = len(X), self.n_states
        log_likelihood = np.zeros((T, N))
        for t, x in enumerate(X):
            for state in range(N):
                log_likelihood[t, state] = self.gmms[state].score([x])
        return self._viterbi(log_likelihood)

    def _viterbi(self, log_likelihood):
        T, N = log_likelihood.shape
        dp = np.full((T, N), -np.inf)
        pointers = np.zeros((T, N), dtype=int)

        dp[0] = log_likelihood[0]
        for t in range(1, T):
            for j in range(N):
                scores = dp[t - 1] + np.log(self.trans_probs[:, j])
                dp[t, j] = np.max(scores) + log_likelihood[t, j]
                pointers[t, j] = np.argmax(scores)

        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(dp[-1])
        for t in range(T - 2, -1, -1):
            best_path[t] = pointers[t + 1, best_path[t + 1]]
        return best_path

# --- CER Calculation ---
def calculate_cer(decoded_texts, ground_truths):
    cer_values = []
    for decoded, ground_truth in zip(decoded_texts, ground_truths):
        distance = Levenshtein.distance(decoded, ground_truth)
        cer = distance / len(ground_truth) if ground_truth else 1.0  # Avoid division by zero
        cer_values.append(cer)
    return sum(cer_values) / len(cer_values)

# --- Character Mapping ---
def decode_char(state, state_to_char_map):
    """Map HMM state to a character."""
    return state_to_char_map.get(state, "?")

def parse_transcription_file(transcription_file):
    segments = []
    with open(transcription_file, "r") as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                start, end = map(float, parts[0].strip('[]').split(','))
                speaker_id = parts[1]
                gender = parts[2]
                transcription = parts[3]
                segments.append({
                    'start': start,
                    'end': end,
                    'speaker_id': speaker_id,
                    'gender': gender,
                    'transcription': transcription
                })
    return segments

# --- Main Code ---
# --- Main Code ---
if __name__ == "__main__":
    import os
    import numpy as np

    wav_dir = "data_dummy/WAV"
    text_dir = "data_dummy/TXT"

    n_mixtures = 3  # Number of Gaussian mixtures

    # Initialize variables to store the best configuration
    best_cer = float('inf')
    best_n_states = None
    best_vocab_size = None

    # Load feature sequences and ground truths
    feature_sequences = []
    for file in os.listdir(wav_dir):
        if file.endswith(".wav"):
            wav_path = os.path.join(wav_dir, file)
            features = extract_mel_spectrogram(wav_path)
            feature_sequences.append(features)

    ground_truths_compiled = []
    for file in os.listdir(text_dir):
        if file.endswith(".txt"):
            transcription_segments = parse_transcription_file(os.path.join(text_dir, file))
            ground_truths = [segment['transcription'] for segment in transcription_segments]
            ground_truths_compiled.append(' '.join(ground_truths))  # Flatten ground truths

    # Grid search for the best combination of n_states and vocab_size
    for n_states in range(10, 41):  # n_states range: 20 to 40
        for vocab_size in range(26, 31):  # vocab_size range: 26 to 30
            try:
                # Prepare mock labels
                concatenated_features = np.vstack(feature_sequences)
                mock_labels = np.random.randint(0, n_states, size=len(concatenated_features))

                # Train HMM-GMM
                hmm_gmm = HMMGMM(n_states=n_states, n_mixtures=n_mixtures)
                hmm_gmm.train(concatenated_features, mock_labels)

                # Map states to characters
                state_to_char_map = {i: chr(97 + i % vocab_size) for i in range(n_states)}

                # Calculate CER for current configuration
                total_cer = 0
                for features, ground_truth in zip(feature_sequences, ground_truths_compiled):
                    decoded_states = hmm_gmm.decode(features)
                    predicted_text = ''.join(decode_char(state, state_to_char_map) for state in decoded_states)
                    cer = calculate_cer([predicted_text], [ground_truth])
                    total_cer += cer

                # Average CER across all sequences
                avg_cer = total_cer / len(feature_sequences)

                # Check if the current configuration is the best
                if avg_cer < best_cer:
                    best_cer = avg_cer
                    best_n_states = n_states
                    best_vocab_size = vocab_size

                print(f"n_states: {n_states}, vocab_size: {vocab_size}, CER: {avg_cer:.2%}")
            except Exception as e:
                print(f"Error: {e}")

    # Display the best configuration
    print("\nBest Configuration:")
    print(f"n_states: {best_n_states}, vocab_size: {best_vocab_size}, CER: {best_cer:.2%}")