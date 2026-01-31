"""
Training Script for VoxProof Voice Classifier
==============================================

This script trains the neural network classifier to distinguish between 
AI-generated and human voice samples. It uses a combination of:
- Acoustic features (MFCCs, pitch, spectral characteristics)  
- Deep embeddings from wav2vec2

The trained model is saved to model/classifier.pth

Author: VoxProof Team
Created for: AI Security Hackathon
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from model.model import VoiceClassifier

# Set up logging for training progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration - tweak these values based on your dataset
# ============================================================================

class TrainingConfig:
    """All the hyperparameters in one place for easy tuning."""
    
    # Model architecture (must match VoiceClassifier)
    INPUT_DIM = 786  # 18 acoustic features + 768 wav2vec2 embedding
    
    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4  # L2 regularization to prevent overfitting
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for this many epochs
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Synthetic data generation (for demo - replace with real data!)
    NUM_SYNTHETIC_SAMPLES = 5000
    
    # Random seed for reproducibility
    SEED = 42
    
    # Output paths
    MODEL_SAVE_PATH = "model/classifier.pth"
    BEST_MODEL_PATH = "model/classifier_best.pth"


def set_seed(seed: int):
    """
    Fix all random seeds for reproducible training.
    
    This is crucial for debugging and comparing experiments.
    Without this, you'd get different results every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # These settings can slow down training but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Synthetic Data Generation
# ============================================================================
# 
# IMPORTANT: In production, replace this with REAL audio data!
# 
# This synthetic data is designed to mimic the statistical properties we 
# expect to see in real AI-generated vs human voice samples. It's useful
# for testing the pipeline, but won't give you a production-quality model.
#
# For real training, you need:
# 1. Human voice samples from various speakers, languages, recording conditions
# 2. AI-generated samples from various TTS systems (ElevenLabs, Bark, etc.)
# ============================================================================

def generate_human_voice_features() -> np.ndarray:
    """
    Generate synthetic features that mimic HUMAN voice characteristics.
    
    Human voices typically show:
    - Higher pitch variance (natural intonation)
    - More spectral variation (breathing, emotion)
    - Natural noise floor from recording environment
    - Irregular rhythm and timing
    
    Returns:
        Feature vector of shape (786,)
    """
    features = []
    
    # --- MFCC Features (13 values) ---
    # Human MFCCs tend to be more variable due to articulation changes
    mfcc_base = np.random.randn(13) * 15 + np.array([
        -250, 80, -20, 30, -15, 25, -10, 15, -8, 12, -5, 8, -3
    ])
    # Add some natural variation
    mfcc = mfcc_base + np.random.randn(13) * 5
    features.extend(mfcc)
    
    # --- Pitch (F0) statistics ---
    # Humans have varied pitch - women typically 165-255 Hz, men 85-180 Hz
    pitch_mean = np.random.uniform(100, 250)
    # High pitch variance is a hallmark of natural speech
    pitch_std = np.random.uniform(30, 80)  # Significant variation!
    features.extend([pitch_mean, pitch_std])
    
    # --- Spectral Rolloff ---
    # Human speech has rich harmonics, rolloff typically 2000-5000 Hz
    spectral_rolloff = np.random.uniform(2500, 5000)
    features.append(spectral_rolloff)
    
    # --- Zero Crossing Rate ---
    # Natural recordings have some ambient noise, so moderate ZCR
    zcr = np.random.uniform(0.04, 0.12)
    features.append(zcr)
    
    # --- Duration ---
    # Typical utterance length
    duration = np.random.uniform(1.0, 10.0)
    features.append(duration)
    
    # --- wav2vec2 Embedding (768 values) ---
    # Human voice embeddings tend to cluster in certain regions
    # We simulate this with a mixture of Gaussians
    embedding = np.random.randn(768) * 0.3
    # Add some structure - real embeddings aren't uniform random
    embedding[:256] += 0.2  # First third slightly positive
    embedding[512:] -= 0.1  # Last third slightly negative
    # Add speaker-specific variation
    embedding += np.random.randn(768) * 0.1
    features.extend(embedding)
    
    return np.array(features, dtype=np.float32)


def generate_ai_voice_features() -> np.ndarray:
    """
    Generate synthetic features that mimic AI-GENERATED voice characteristics.
    
    AI/TTS voices typically show:
    - Lower pitch variance (robotic monotone)
    - Cleaner signal (no mic noise, perfect clarity)
    - More uniform spectral characteristics
    - Suspiciously consistent timing
    
    Returns:
        Feature vector of shape (786,)
    """
    features = []
    
    # --- MFCC Features (13 values) ---
    # AI-generated MFCCs are often more consistent/smooth
    mfcc_base = np.random.randn(13) * 10 + np.array([
        -260, 70, -25, 25, -12, 20, -8, 12, -6, 10, -4, 6, -2
    ])
    # Less variation than human
    mfcc = mfcc_base + np.random.randn(13) * 2
    features.extend(mfcc)
    
    # --- Pitch (F0) statistics ---
    # TTS systems often have unnaturally stable pitch
    pitch_mean = np.random.uniform(120, 220)
    # KEY DIFFERENCE: Low pitch variance = robotic!
    pitch_std = np.random.uniform(5, 25)  # Much less variation
    features.extend([pitch_mean, pitch_std])
    
    # --- Spectral Rolloff ---
    # AI voices can have unusual frequency distribution
    # Either too clean (low) or overprocessed (high)
    if np.random.random() > 0.5:
        spectral_rolloff = np.random.uniform(1500, 2500)  # Too clean
    else:
        spectral_rolloff = np.random.uniform(5500, 7000)  # Over-bright
    features.append(spectral_rolloff)
    
    # --- Zero Crossing Rate ---
    # Synthetic audio is often suspiciously clean
    zcr = np.random.uniform(0.01, 0.04)  # Very low noise
    features.append(zcr)
    
    # --- Duration ---
    duration = np.random.uniform(1.0, 10.0)
    features.append(duration)
    
    # --- wav2vec2 Embedding (768 values) ---
    # AI voice embeddings have different distributional properties
    embedding = np.random.randn(768) * 0.25
    # Different structure than human embeddings
    embedding[:256] -= 0.15
    embedding[256:512] += 0.25
    # Less natural variation
    embedding += np.random.randn(768) * 0.05
    features.extend(embedding)
    
    return np.array(features, dtype=np.float32)


class SyntheticVoiceDataset(Dataset):
    """
    PyTorch Dataset for synthetic voice detection data.
    
    In a real scenario, you'd load actual audio files here and 
    extract features on-the-fly or from a preprocessed cache.
    """
    
    def __init__(self, num_samples: int = 5000, seed: int = 42):
        """
        Generate a balanced dataset of human and AI samples.
        
        Args:
            num_samples: Total number of samples to generate
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        
        # Set seed for this dataset generation
        np.random.seed(seed)
        
        # Generate balanced dataset (50% human, 50% AI)
        self.features = []
        self.labels = []
        
        samples_per_class = num_samples // 2
        
        logger.info(f"Generating {samples_per_class} human voice samples...")
        for _ in range(samples_per_class):
            self.features.append(generate_human_voice_features())
            self.labels.append(0)  # 0 = HUMAN
        
        logger.info(f"Generating {samples_per_class} AI voice samples...")
        for _ in range(samples_per_class):
            self.features.append(generate_ai_voice_features())
            self.labels.append(1)  # 1 = AI_GENERATED
        
        # Convert to numpy arrays
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # Shuffle the dataset
        shuffle_idx = np.random.permutation(len(self.features))
        self.features = self.features[shuffle_idx]
        self.labels = self.labels[shuffle_idx]
        
        logger.info(f"Dataset created: {len(self)} samples total")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single sample as tensors."""
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label


def split_dataset(
    dataset: SyntheticVoiceDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/validation/test sets.
    
    We need all three:
    - Train: Model learns from this
    - Validation: Used to tune hyperparameters and early stopping
    - Test: Final evaluation, only used once!
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Use PyTorch's random_split for proper shuffling
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(TrainingConfig.SEED)
    )
    
    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    return train_set, val_set, test_set


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one complete epoch.
    
    Args:
        model: The neural network
        train_loader: DataLoader for training data
        optimizer: The optimizer (Adam, SGD, etc.)
        criterion: Loss function (BCEWithLogitsLoss for binary classification)
        device: CPU or CUDA
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()  # Enable dropout, batch norm in training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        # Move data to GPU if available
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)  # Add dimension for BCEWithLogitsLoss
        
        # Zero out gradients from previous batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass - compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * features.size(0)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    accuracy = correct / total
    
    return epoch_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on validation/test data.
    
    We use torch.no_grad() here because we're not training,
    so we don't need to compute gradients. This saves memory
    and speeds up evaluation.
    """
    model.eval()  # Disable dropout, use running stats for batch norm
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradient computation needed
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * features.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    val_loss = running_loss / total
    accuracy = correct / total
    
    return val_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device
) -> nn.Module:
    """
    Full training loop with early stopping and model checkpointing.
    
    Early stopping prevents overfitting by halting training when
    validation performance stops improving.
    """
    # Binary Cross Entropy with Logits - combines sigmoid + BCE for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer - good default choice, adapts learning rate per parameter
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY  # L2 regularization
    )
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # We want to minimize loss
        factor=0.5,      # Halve the LR
        patience=5       # Wait 5 epochs before reducing
    )
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info("=" * 60)
    
    # Training loop
    for epoch in range(config.EPOCHS):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Logging
        logger.info(
            f"Epoch [{epoch+1:3d}/{config.EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model checkpoint
            torch.save(best_model_state, config.BEST_MODEL_PATH)
            logger.info(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model weights")
    
    return model


def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
):
    """
    Final evaluation on the held-out test set.
    
    This should only be run ONCE after all training and tuning is complete.
    If you peek at test results and then tune, you're cheating!
    """
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    logger.info("=" * 60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2%}")
    logger.info("=" * 60)
    
    # Additional metrics would go here in production:
    # - Precision, Recall, F1 Score
    # - Confusion Matrix
    # - ROC-AUC
    # - Per-class accuracy
    
    return test_loss, test_acc


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main training script entry point.
    
    Run this to train the voice classifier from scratch.
    """
    print("\n" + "=" * 60)
    print("VoxProof Voice Classifier Training")
    print("=" * 60 + "\n")
    
    # Configuration
    config = TrainingConfig()
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Determine device - use GPU if available for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create synthetic dataset
    # NOTE: In production, replace this with real audio data!
    logger.info("\n[Step 1/5] Generating synthetic training data...")
    dataset = SyntheticVoiceDataset(
        num_samples=config.NUM_SYNTHETIC_SAMPLES,
        seed=config.SEED
    )
    
    # Split into train/val/test
    logger.info("\n[Step 2/5] Splitting dataset...")
    train_set, val_set, test_set = split_dataset(
        dataset, 
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,  # Important for training!
        num_workers=0,  # Set >0 on Linux for faster loading
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    # Initialize model
    logger.info("\n[Step 3/5] Initializing model...")
    model = VoiceClassifier(dropout_rate=0.3)
    model.to(device)
    
    # Count parameters (useful for debugging model size)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Ensure output directory exists
    Path(config.MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    logger.info("\n[Step 4/5] Training model...")
    model = train_model(model, train_loader, val_loader, config, device)
    
    # Final evaluation on test set
    logger.info("\n[Step 5/5] Evaluating on test set...")
    evaluate_on_test(model, test_loader, device)
    
    # Save final model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    logger.info(f"\n✓ Final model saved to: {config.MODEL_SAVE_PATH}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {config.MODEL_SAVE_PATH}")
    print("\nNext steps:")
    print("1. Start the API server: uvicorn app:app --reload")
    print("2. Test with: python test_client.py your_audio.mp3")
    print("\n⚠️  For production, retrain with REAL audio data!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
