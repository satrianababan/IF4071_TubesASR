from parsing.parse_txt import parse_transcription_file
from parsing.parse_audio import slice_audio
from collate import collate_fn

import torch.nn as nn
import matplotlib.pyplot as plt

from audio_dataset import AudioDataset
from hmmlearn import hmm
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking, AmplitudeToDB
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from pathlib import Path

import numpy as np
import torch
import random

# Preprocessing Function
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024),
    AmplitudeToDB(stype='power', top_db=80),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35),
)

def collect_audio_paths(directory):
    # Recursively find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac']  # Add other extensions as needed
    return [
        str(file) for file in Path(directory).rglob('*')
        if file.suffix.lower() in audio_extensions
    ]

# Initialize Dataset, DataLoader
path = "sliced_audio_segments"
audio_paths = collect_audio_paths(path)
print("Number of Audio Files:", len(audio_paths))
dataset = AudioDataset(audio_paths, transform=transform)
print("Dataset Length:", len(dataset))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, collate_fn=collate_fn)

# Loop through DataLoader
for i, batch in enumerate(dataloader):
    padded_audio, labels = batch

#     # Visualize Soundwave and Mel-Spectrogram for the first waveform
    for waveform in padded_audio:
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
        num_frames = features_flat.shape[0]  # Number of time frames (e.g., 116)

        # Ensure labels[i] is a sequence
        label_seq = labels[i] if labels[i].ndim > 0 else labels[i].unsqueeze(0)

        # Repeat and trim the label sequence to match the number of time frames
        repeated_labels = label_seq.repeat((num_frames // len(label_seq)) + 1)
        repeated_labels = repeated_labels[:num_frames]  # Trim to match time frames

        # Append flattened features and corresponding labels
        features_list.append(features_flat.numpy())  # Convert to numpy
        labels_list.extend(repeated_labels.numpy())  # Convert to numpy

# Convert lists to numpy arrays
features_all = np.vstack(features_list)  # Shape: [total_time_frames, frequency_bins]
labels_all = np.array(labels_list)       # Shape: [total_time_frames]

print("Features_all shape:", features_all.shape)
print("Labels_all shape:", labels_all.shape)
print("Unique labels:", np.unique(labels_all))

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
    print(f"State {state}: {state_features.shape[0]} samples")
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
                        reg_covar=1e-3  # Try larger values like 1e-2, 1e-1 if needed
                    )

    gmm.fit(state_features)
    gmms.append(gmm)

print("GMM training complete!")

# ========================
# Step 7: HMM Initialization
# ========================
print("Initializing HMM...")
for state, gmm in enumerate(gmms):
    if gmm is not None:
        print(f"GMM for state {state}:")
        # print("Means:", gmm.means_)
        print("Weights:", gmm.weights_)

# Filter out None values from gmms
valid_gmms = [gmm for gmm in gmms if gmm is not None]
valid_states = [state for state, gmm in enumerate(gmms) if gmm is not None]

if len(valid_gmms) == 0:
    raise ValueError("No valid GMMs were trained. Check your data or minimum sample threshold.")

# Initialize HMM with valid states
n_components = len(valid_gmms)
hmm_model = hmm.GMMHMM(n_components=n_components, n_mix=4, covariance_type="diag")

# Create mapping from valid GMM indices to original state indices
# valid_to_original = {i: idx for i, idx in enumerate(range(len(gmms))) if gmms[idx] is not None}

# Prepare HMM initialization parameters
n_components = len(valid_gmms)  # Number of valid states
n_features = features_all.shape[1]  # Number of features (e.g., 128)
n_mix = 4  # Number of mixture components

# Initialize HMM
hmm_model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix, covariance_type="diag")

# Initialize start probabilities and transition matrix
hmm_model.startprob_ = np.full(n_components, 1 / n_components)  # Uniform start probabilities
hmm_model.transmat_ = np.full((n_components, n_components), 1 / n_components)  # Uniform transition probabilities

# Initialize means_, covars_, and weights_
hmm_model.means_ = np.array([gmm.means_ for gmm in valid_gmms]).reshape(n_components, n_mix, n_features)
hmm_model.covars_ = np.array([gmm.covariances_ for gmm in valid_gmms]).reshape(n_components, n_mix, n_features)
hmm_model.weights_ = np.array([gmm.weights_ for gmm in valid_gmms])

# Print HMM parameters
print("Start probabilities:", hmm_model.startprob_)
print("Transition matrix:", hmm_model.transmat_)
print("Means shape:", hmm_model.means_.shape)
print("Covariances shape:", hmm_model.covars_.shape)
print("Weights shape:", hmm_model.weights_.shape)

# Visualize the transition matrix
plt.figure(figsize=(8, 6))
plt.imshow(hmm_model.transmat_, cmap="viridis")
plt.colorbar()
plt.title("HMM Transition Matrix")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# Map the original states to the valid states for decoding
original_to_valid = {state: idx for idx, state in enumerate(valid_states)}
valid_to_original = {idx: state for idx, state in enumerate(valid_states)}

# ========================
# Step 8: Decoding with HMM
# ========================
print("Decoding...")
test_features = features_all[:100]  # Example test features
print("Test features shape:", test_features.shape)
# print("First few feature vectors:", test_features[:5])
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