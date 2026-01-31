"""Model module for VoxProof."""

from .model import (
    VoiceClassifier,
    Wav2VecEmbedder,
    VoiceDetectionModel,
    PredictionResult,
    get_model,
    create_dummy_weights
)

__all__ = [
    "VoiceClassifier",
    "Wav2VecEmbedder", 
    "VoiceDetectionModel",
    "PredictionResult",
    "get_model",
    "create_dummy_weights"
]
