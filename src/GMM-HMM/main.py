from parsing.parse_txt import parse_transcription_file
from parsing.parse_audio import slice_audio
from collate import collate_fn

import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from audio_dataset import AudioDataset
from hmmlearn import hmm
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking, AmplitudeToDB
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

import random
import numpy as np

# Set the random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load text transcriptions and audio paths
transcription_file = "././dataset_testing/TXT/A0102_S0001_0_G1357.txt"
segments = parse_transcription_file(transcription_file)

# Save the segments to a JSON file
with open("././jsonAudio/segments.json", "w") as file:
    json.dump(segments, file, indent=4)
    
# Load the segments from the JSON file
transcription_file = "././dataset_testing/TXT/A0102_S0001_0_G1357.txt"
audio_file = "././dataset_testing/WAV/A0102_S0001_0_G1357.wav"
if audio_file:
    output_dir = "sliced_audio_segments"
    segments = parse_transcription_file(transcription_file)
    audio_segments = slice_audio(audio_file, segments, output_dir)
else:
    print("Audio file not found")

# Preprocessing Function
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024),
    AmplitudeToDB(stype='power', top_db=80),  # Convert amplitude to decibels
    FrequencyMasking(freq_mask_param=15),  # Apply frequency masking
    TimeMasking(time_mask_param=35),  # Apply time masking
)

# Initialize Dataset, DataLoader
audio_paths = [segment['path'] for segment in audio_segments]
dataset = AudioDataset(audio_segments, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, collate_fn=collate_fn)

# Loop through DataLoader
for i, batch in enumerate(dataloader):
    waveforms, transcriptions = batch  # transcriptions: tensor with indices

#     # Visualize Soundwave and Mel-Spectrogram for the first waveform
    for waveform in waveforms:
        # Visualize raw waveform
        plt.figure(figsize=(10, 3))
        plt.plot(waveform.squeeze().numpy())
        plt.title("Soundwave")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

        # Visualize Mel-Spectrogram
        plt.figure(figsize=(10, 3))
        plt.imshow(waveform.squeeze().numpy(), aspect='auto', origin='lower')        
        plt.colorbar()
        plt.title("Mel-spectrogram")
        plt.show()
    break  # Exit after processing one batch

# Step 4: Extract features and print them
features_list = []
labels_list = []

print("Aligning features and labels...")
for features, labels in dataloader:
    for i in range(features.shape[0]):  # Iterate over the batch
        features_flat = features[i].T  # Shape: [time_frames, frequency_bins]
        num_frames = features_flat.shape[0]  # Number of time frames (116)

        # Repeat and trim the label sequence to match the number of time frames
        repeated_labels = labels[i].repeat((num_frames // len(labels[i])) + 1)
        repeated_labels = repeated_labels[:num_frames]  # Trim to match time frames

        # Append flattened features and corresponding labels
        features_list.append(features_flat.numpy())  # Convert to numpy
        labels_list.extend(repeated_labels.numpy())  # Convert to numpy

# Convert lists to numpy arrays
features_all = np.vstack(features_list)  # Shape: [total_time_frames, frequency_bins]
labels_all = np.array(labels_list)       # Shape: [total_time_frames]

print("Final Features shape:", features_all.shape)  # Should match total frames
print("Final Labels shape:", labels_all.shape)      # Should match total frames

# ========================
# Step 6: Train GMMs for Acoustic Modeling
# ========================
print("Training GMM for each state...")
num_states = len(set(labels_all))  # Assume each label is a state

# Define the minimum samples required for a given GMM component
min_samples = 10
reg_covar = 1e-4  # Small regularization value to stabilize covariance estimation
gmms = []  # List to store GMMs for each state

for state in range(num_states):
    print(f"Training GMM for State {state}...")
    
    # Select features corresponding to the current state
    state_features = features_all[labels_all == state]
    num_samples = state_features.shape[0]
    
    # Check if enough samples are available
    if num_samples < min_samples:
        print(f"  Skipping State {state} due to insufficient samples ({num_samples} < {min_samples}).")
        gmms.append(None)
        continue
    
    # Adjust n_components dynamically based on the number of samples
    n_components = min(4, num_samples)  # Reduce components if samples are fewer
    print(f"  Number of samples: {num_samples}, using {n_components} components.")
    
    # Initialize and fit GMM with regularization
    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type="diag",
                        random_state=seed,
                        reg_covar=1e-3  # Try larger values like 1e-2, 1e-1 if needed
                    )

    gmm.fit(state_features)
    gmms.append(gmm)

print("GMM training complete!")

# ========================
# Step 7: HMM Initialization
# ========================
print("Initializing HMM...")
# Filter out None values from gmms
valid_gmms = [gmm for gmm in gmms if gmm is not None]
valid_states = [state for state, gmm in enumerate(gmms) if gmm is not None]

if len(valid_gmms) == 0:
    raise ValueError("No valid GMMs were trained. Check your data or minimum sample threshold.")

# Initialize HMM with valid states
n_components = len(valid_gmms)
hmm_model = hmm.GMMHMM(n_components=n_components, n_mix=4, covariance_type="diag", random_state=seed)

# Create mapping from valid GMM indices to original state indices
# valid_to_original = {i: idx for i, idx in enumerate(range(len(gmms))) if gmms[idx] is not None}

# Prepare HMM initialization parameters
n_components = len(valid_gmms)  # Number of valid states
n_features = features_all.shape[1]  # Number of features (e.g., 128)
n_mix = 4  # Number of mixture components

# Initialize HMM
hmm_model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix, covariance_type="diag", random_state=seed)

# Initialize start probabilities and transition matrix
hmm_model.startprob_ = np.full(n_components, 1 / n_components)  # Uniform start probabilities
hmm_model.transmat_ = np.full((n_components, n_components), 1 / n_components)  # Uniform transition probabilities

# Initialize means_, covars_, and weights_
hmm_model.means_ = np.array([gmm.means_ for gmm in valid_gmms]).reshape(n_components, n_mix, n_features)
hmm_model.covars_ = np.array([gmm.covariances_ for gmm in valid_gmms]).reshape(n_components, n_mix, n_features)
hmm_model.weights_ = np.array([gmm.weights_ for gmm in valid_gmms])

# Map the original states to the valid states for decoding
original_to_valid = {state: idx for idx, state in enumerate(valid_states)}
valid_to_original = {idx: state for idx, state in enumerate(valid_states)}

# ========================
# Step 8: Decoding with HMM
# ========================
print("Decoding...")
test_features = features_all[:100]  # Example test features
log_prob, predicted_states = hmm_model.decode(test_features, algorithm="viterbi")

# Map valid states back to the original states
predicted_states = [valid_to_original[state] for state in predicted_states]

print("Log Probability of Sequence:", log_prob)
print("Predicted States:", predicted_states)

# ========================
# Step 9: Map States to Words
# ========================
# Define a simple mapping of states to words/phonemes
# state_to_word = {
#     0: "cat", 1: "sits", 2: "on", 3: "a", 4: "mat"
# }

# # Convert predicted states to words
# decoded_sequence = [state_to_word[state] for state in predicted_states]
# print("Decoded Sequence:", " ".join(decoded_sequence))

# ========================
# Step 10: Visualization
# ========================
plt.figure(figsize=(10, 5))
plt.plot(predicted_states, marker="o")
plt.title("Predicted States Sequence")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()