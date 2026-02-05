"""
VoxProof - Training Script for AI Voice Classifier
===================================================

This script trains a neural network to distinguish between AI-generated
and human speech. We use a hybrid approach combining traditional acoustic
features with deep embeddings from Wav2Vec2.

The idea is simple:
- AI voices tend to have unnatural pitch patterns, too-clean signals
- Human voices have natural variations, breath sounds, room acoustics
- By combining hand-crafted features with learned embeddings, we get
  the best of both worlds

Usage:
    python train_classifier.py

Dataset structure:
    dataset/
        human/  -> Real human speech files (.mp3)
        ai/     -> AI-generated speech files (.mp3)

Output:
    model/classifier.pth  -> Trained model weights

Author: VoxProof Team
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydub import AudioSegment
from tqdm import tqdm

# Suppress noisy transformer warnings - they clutter the output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import transformers
transformers.logging.set_verbosity_error()


# =============================================================================
# CONFIGURATION
# These are the knobs you can turn to tweak training behavior
# =============================================================================

DATASET_DIR = Path("dataset")
HUMAN_DIR = DATASET_DIR / "human"   # Label: 0
AI_DIR = DATASET_DIR / "ai"         # Label: 1

MODEL_SAVE_PATH = Path("model/classifier.pth")

# Audio settings - 16kHz is standard for speech models
SAMPLE_RATE = 16000

# Training hyperparameters
BATCH_SIZE = 16       # How many samples per gradient update
LEARNING_RATE = 0.001 # Adam optimizer learning rate
EPOCHS = 15           # Number of passes through the dataset
TRAIN_RATIO = 0.8     # 80% train, 20% validation
SEED = 42             # For reproducibility

# Wav2Vec2 pretrained model - this gives us rich audio embeddings
WAV2VEC_MODEL = "facebook/wav2vec2-base-960h"
WAV2VEC_DIM = 768     # Wav2Vec2 embedding size
ACOUSTIC_DIM = 18     # Our hand-crafted acoustic features
INPUT_DIM = WAV2VEC_DIM + ACOUSTIC_DIM  # Total: 786 dimensions


# =============================================================================
# REPRODUCIBILITY
# Set all random seeds so we get consistent results across runs
# =============================================================================

def set_seed(seed: int):
    """Fix random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# A simple but effective feedforward network with dropout for regularization
# =============================================================================

class VoiceClassifier(nn.Module):
    """
    Binary classifier for AI vs Human voice detection.
    
    Takes 786-dimensional input (18 acoustic + 768 wav2vec features)
    and outputs a single logit. Apply sigmoid for probability.
    
    Architecture: 786 -> 512 -> 256 -> 128 -> 64 -> 1
    With BatchNorm and Dropout between each layer for stable training.
    """
    
    def __init__(self, input_dim: int = 786, dropout_rate: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: Input projection
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2: Hidden
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3: Hidden
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 4: Hidden
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 5: Output (single logit)
            nn.Linear(64, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# AUDIO LOADING
# Handle various audio formats and normalize to a standard representation
# =============================================================================

def load_audio(filepath: Path) -> Optional[np.ndarray]:
    """
    Load an audio file and preprocess it for feature extraction.
    
    Steps:
    1. Load MP3/WAV using pydub (handles format conversion)
    2. Convert to mono (we don't need stereo for voice analysis)
    3. Resample to 16kHz (required by Wav2Vec2)
    4. Normalize to [-1, 1] range (consistent amplitude)
    
    Returns None if loading fails - we'll skip these files.
    """
    try:
        # Load audio file
        if filepath.suffix.lower() == ".mp3":
            audio_segment = AudioSegment.from_mp3(str(filepath))
        else:
            audio_segment = AudioSegment.from_file(str(filepath))
        
        # Convert stereo to mono
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Get raw samples as numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Convert to float32 based on bit depth
        if audio_segment.sample_width == 2:  # 16-bit audio (most common)
            samples = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit audio
            samples = samples.astype(np.float32) / 2147483648.0
        else:
            samples = samples.astype(np.float32) / (np.max(np.abs(samples)) + 1e-8)
        
        # Resample to target sample rate if needed
        original_sr = audio_segment.frame_rate
        if original_sr != SAMPLE_RATE:
            samples = librosa.resample(samples, orig_sr=original_sr, target_sr=SAMPLE_RATE)
        
        # Peak normalize to [-1, 1]
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
        
        return samples
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


# =============================================================================
# FEATURE EXTRACTION
# The core of our approach: extract meaningful features from audio
# =============================================================================

def extract_acoustic_features(waveform: np.ndarray) -> np.ndarray:
    """
    Extract hand-crafted acoustic features that help distinguish AI from human voices.
    
    Features extracted (18 total):
    - MFCCs (13): Mel-frequency cepstral coefficients capture the "timbre" of voice
    - Pitch mean: Average fundamental frequency
    - Pitch std: Pitch variation (AI voices are often too stable!)
    - Spectral rolloff: Where most audio energy is concentrated
    - Zero crossing rate: Signal noisiness (humans have breath, room noise)
    - Duration: Length of the clip
    
    These features were designed by audio researchers over decades and work
    surprisingly well for detecting synthetic voices.
    """
    # MFCCs - the bread and butter of audio analysis
    mfccs = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    # Pitch extraction using pYIN algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz (low male voice)
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (high female voice)
        sr=SAMPLE_RATE
    )
    
    # Handle unvoiced segments (NaN values)
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) > 0:
        pitch_mean = float(np.mean(f0_valid))
        pitch_std = float(np.std(f0_valid))
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
    
    # Spectral rolloff - frequency below which 85% of energy lies
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=waveform, sr=SAMPLE_RATE, roll_percent=0.85
    )
    spectral_rolloff_mean = float(np.mean(spectral_rolloff))
    
    # Zero crossing rate - how "noisy" the signal is
    zcr = librosa.feature.zero_crossing_rate(waveform)
    zcr_mean = float(np.mean(zcr))
    
    # Duration in seconds
    duration = len(waveform) / SAMPLE_RATE
    
    # Combine all features into a single vector
    features = np.concatenate([
        mfcc_mean,  # 13 features
        np.array([pitch_mean, pitch_std, spectral_rolloff_mean, zcr_mean, duration])  # 5 features
    ])
    
    return features  # Total: 18 features


class Wav2VecEmbedder:
    """
    Extract deep audio embeddings using Facebook's pretrained Wav2Vec2 model.
    
    This model was trained on thousands of hours of speech and learned to
    represent audio in a 768-dimensional space. It captures complex patterns
    that are hard to describe with hand-crafted features.
    
    We freeze the model (no training) and just use it as a feature extractor.
    """
    
    def __init__(self):
        self.device = torch.device("cpu")  # CPU only for simplicity
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load(self):
        """Load the pretrained model (lazy loading to save memory at startup)."""
        if self._loaded:
            return
        
        print(f"Loading Wav2Vec2 model: {WAV2VEC_MODEL}")
        self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
        self.model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters - we're only extracting embeddings
        for param in self.model.parameters():
            param.requires_grad = False
        
        self._loaded = True
        print("Wav2Vec2 loaded on CPU")
    
    @torch.no_grad()
    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Extract 768-dimensional embedding from audio waveform."""
        if not self._loaded:
            self.load()
        
        # Preprocess audio for the model
        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        
        # Forward pass through the model
        outputs = self.model(input_values)
        
        # Mean pool over time dimension to get a single vector
        embedding = outputs.last_hidden_state.mean(dim=1)
        
        return embedding.cpu().numpy().squeeze()


# =============================================================================
# DATASET PREPARATION
# Load all audio files and extract features
# =============================================================================

def collect_audio_paths() -> Tuple[List[Path], List[int]]:
    """
    Scan the dataset directories and collect all audio file paths with labels.
    
    Returns:
        paths: List of file paths
        labels: 0 for human, 1 for AI
    """
    paths = []
    labels = []
    
    # Collect human voice samples (label = 0)
    human_files = list(HUMAN_DIR.glob("*.mp3")) + list(HUMAN_DIR.glob("*.wav"))
    for f in human_files:
        paths.append(f)
        labels.append(0)
    
    # Collect AI-generated samples (label = 1)
    ai_files = list(AI_DIR.glob("*.mp3")) + list(AI_DIR.glob("*.wav"))
    for f in ai_files:
        paths.append(f)
        labels.append(1)
    
    print(f"Found {len(human_files)} human samples, {len(ai_files)} AI samples")
    
    return paths, labels


def extract_all_features(
    paths: List[Path], 
    labels: List[int],
    embedder: Wav2VecEmbedder
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process all audio files and extract features.
    
    For each file:
    1. Load and preprocess the audio
    2. Extract 18 acoustic features
    3. Extract 768-dim Wav2Vec2 embedding
    4. Concatenate into a 786-dim feature vector
    
    Skips files that fail to load or are too short.
    """
    all_features = []
    all_labels = []
    
    print(f"\nExtracting features from {len(paths)} audio files...")
    
    for path, label in tqdm(zip(paths, labels), total=len(paths), desc="Processing"):
        # Load audio
        waveform = load_audio(path)
        if waveform is None:
            continue
        
        # Skip very short clips (less than 100ms)
        if len(waveform) < SAMPLE_RATE * 0.1:
            continue
        
        # Extract both types of features
        acoustic = extract_acoustic_features(waveform)    # 18 dims
        embedding = embedder.extract(waveform)            # 768 dims
        
        # Combine: [acoustic, wav2vec] = 786 dims
        combined = np.concatenate([acoustic, embedding])
        
        all_features.append(combined)
        all_labels.append(label)
    
    return np.array(all_features), np.array(all_labels)


def train_val_split(
    features: np.ndarray, 
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle the data and split into training and validation sets.
    
    Default split: 80% training, 20% validation
    """
    n_samples = len(features)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * TRAIN_RATIO)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_val = features[val_indices]
    y_val = labels[val_indices]
    
    return X_train, y_train, X_val, y_val


# =============================================================================
# TRAINING LOOP
# Standard PyTorch training with mini-batches
# =============================================================================

def create_batches(
    features: np.ndarray, 
    labels: np.ndarray, 
    batch_size: int, 
    shuffle: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split features and labels into mini-batches for training.
    
    Shuffling is important during training to prevent the model from
    learning the order of samples instead of their features.
    """
    n_samples = len(features)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        
        X_batch = torch.tensor(features[batch_indices], dtype=torch.float32)
        y_batch = torch.tensor(labels[batch_indices], dtype=torch.float32).unsqueeze(1)
        
        batches.append((X_batch, y_batch))
    
    return batches


def train_epoch(
    model: VoiceClassifier,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> float:
    """
    Train the model for one epoch.
    
    Standard training loop:
    1. Forward pass
    2. Compute loss
    3. Backward pass (compute gradients)
    4. Update weights
    
    Returns the average loss over all batches.
    """
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in batches:
        optimizer.zero_grad()              # Clear old gradients
        outputs = model(X_batch)           # Forward pass
        loss = criterion(outputs, y_batch) # Compute loss
        loss.backward()                    # Backward pass
        optimizer.step()                   # Update weights
        total_loss += loss.item()
    
    return total_loss / len(batches)


def validate(
    model: VoiceClassifier,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module
) -> float:
    """
    Evaluate model on validation set.
    
    No gradient computation needed here - we're just measuring performance.
    Returns the average loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in batches:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(batches)


# =============================================================================
# MAIN TRAINING FUNCTION
# Orchestrates the entire training process
# =============================================================================

def main():
    """
    Main training entry point.
    
    This function:
    1. Loads the dataset
    2. Extracts features from all audio files
    3. Splits into train/validation sets
    4. Trains the classifier for N epochs
    5. Saves the best model based on validation loss
    """
    print("=" * 60)
    print("VoxProof Voice Classifier Training")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    set_seed(SEED)
    
    # Step 1: Collect audio paths
    paths, labels = collect_audio_paths()
    
    if len(paths) == 0:
        print("No audio files found in dataset/")
        return
    
    # Step 2: Initialize Wav2Vec2 embedder
    embedder = Wav2VecEmbedder()
    
    # Step 3: Extract features from all files
    features, labels = extract_all_features(paths, labels, embedder)
    
    print(f"\nTotal samples: {len(features)}")
    print(f"Feature dimension: {features.shape[1]}")
    
    # Step 4: Train/validation split
    X_train, y_train, X_val, y_val = train_val_split(features, labels)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Step 5: Initialize model, loss function, and optimizer
    model = VoiceClassifier(input_dim=INPUT_DIM)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # Step 6: Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("-" * 60)
    
    for epoch in range(1, EPOCHS + 1):
        # Create batches (shuffle training data each epoch)
        train_batches = create_batches(X_train, y_train, BATCH_SIZE, shuffle=True)
        val_batches = create_batches(X_val, y_val, BATCH_SIZE, shuffle=False)
        
        # Train and validate
        train_loss = train_epoch(model, train_batches, criterion, optimizer)
        val_loss = validate(model, val_batches, criterion)
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"           -> Saved best model (val_loss: {val_loss:.4f})")
    
    print("-" * 60)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
