"""
Audio Processing Module - The Ears of VoxProof
===============================================

This module handles everything related to audio input processing:

1. Base64 Decoding: API receives audio as Base64 strings (web-friendly)
2. MP3 Decoding: Convert compressed audio to raw samples  
3. Resampling: Standardize to 16kHz (required by wav2vec2)
4. Normalization: Scale audio to consistent level
5. Feature Extraction: Pull out acoustic characteristics

Why these steps matter:
- Raw MP3s come in all shapes and sizes (44.1kHz, stereo, etc.)
- Our models expect consistent 16kHz mono input
- Normalization prevents loud/quiet recordings from skewing results
- Features like MFCCs capture the "texture" of voice

Author: VoxProof Team
License: MIT
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Module logger for debugging audio issues
logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """
    Container for extracted acoustic features.
    
    These features are "hand-crafted" - designed by audio researchers over decades.
    They capture different aspects of how voice sounds:
    
    - MFCCs: The "timbre" or tonal quality (what makes voices sound different)
    - Pitch: How high/low the voice is, and how much it varies
    - Spectral Rolloff: Where most of the audio energy is concentrated
    - Zero Crossing Rate: How "noisy" or "clean" the signal is
    
    For AI detection, the KEY insight is that synthetic voices often have:
    - Unnaturally stable pitch (robotic monotone)
    - Suspiciously clean signals (no breath noise, room reverb)
    - Unusual spectral characteristics (over-processed)
    """
    mfcc_mean: np.ndarray          # 13 MFCC coefficients (averaged over time)
    pitch_mean: float               # Average fundamental frequency (Hz)
    pitch_std: float                # How much pitch varies (low = robotic)
    spectral_rolloff_mean: float    # Frequency below which 85% of energy lies
    zero_crossing_rate_mean: float  # Rate of sign changes (high = noisy)
    duration: float                 # Length in seconds
    
    def to_vector(self) -> np.ndarray:
        """
        Flatten all features into a single numpy array.
        
        This is what gets fed to the neural network classifier.
        Total dimensions: 13 (MFCCs) + 5 (others) = 18
        """
        return np.concatenate([
            self.mfcc_mean,
            np.array([
                self.pitch_mean,
                self.pitch_std,
                self.spectral_rolloff_mean,
                self.zero_crossing_rate_mean,
                self.duration
            ])
        ])


class AudioProcessor:
    """
    The audio preprocessing pipeline.
    
    Takes raw audio in various formats and produces:
    1. Clean waveform (16kHz, mono, normalized)
    2. Acoustic features (MFCCs, pitch, etc.)
    
    This standardization is crucial - without it, the same voice could
    produce wildly different features depending on recording settings.
    
    Pipeline:
        Base64 string -> MP3 bytes -> Raw samples -> 16kHz mono -> Normalized -> Features
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize the audio processor.
        
        Args:
            target_sample_rate: What sample rate to normalize to.
                               16000 Hz is standard for speech models.
                               (Higher = more detail but more computation)
        """
        self.target_sample_rate = target_sample_rate
        logger.debug(f"AudioProcessor initialized (target: {target_sample_rate}Hz)")
    
    def decode_base64(self, audio_base64: str) -> bytes:
        """
        Decode Base64-encoded audio to raw bytes.
        
        Web APIs often use Base64 because it's safe to include in JSON.
        This converts it back to binary data we can actually process.
        
        Handles both:
        - Plain Base64: "SGVsbG8gV29ybGQ="
        - Data URLs: "data:audio/mp3;base64,SGVsbG8gV29ybGQ="
        """
        try:
            # Strip data URL prefix if present (common from browser uploads)
            if "base64," in audio_base64:
                audio_base64 = audio_base64.split("base64,")[1]
            
            audio_bytes = base64.b64decode(audio_base64)
            return audio_bytes
        except Exception as e:
            logger.error(f"Base64 decoding failed: {e}")
            raise ValueError(f"Invalid Base64 encoding: {e}")
    
    def mp3_to_waveform(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """
        Convert MP3 bytes to a numpy waveform array.
        
        MP3 is compressed - we need raw samples to do signal processing.
        pydub handles the heavy lifting of MP3 decoding.
        
        Returns:
            Tuple of (waveform as float32 array, original sample rate)
        """
        try:
            # Use pydub to handle MP3 decoding
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Get raw audio data
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Normalize to float32 in range [-1, 1]
            if audio_segment.sample_width == 2:  # 16-bit audio
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:  # 32-bit audio
                samples = samples.astype(np.float32) / 2147483648.0
            else:
                samples = samples.astype(np.float32) / np.max(np.abs(samples))
            
            sample_rate = audio_segment.frame_rate
            return samples, sample_rate
            
        except Exception as e:
            logger.error(f"MP3 conversion failed: {e}")
            raise ValueError(f"Failed to decode MP3 audio: {e}")
    
    def resample(self, waveform: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Resample audio to our target sample rate (16kHz).
        
        Different recordings come at different sample rates:
        - CD quality: 44.1 kHz
        - Professional: 48 kHz  
        - Phone calls: 8 kHz
        
        We need to standardize to 16kHz because:
        1. wav2vec2 expects exactly 16kHz
        2. Consistent sample rate = consistent features
        
        librosa's resample uses high-quality polyphase filtering.
        """
        # Skip if already at target rate
        if original_sr == self.target_sample_rate:
            return waveform
        
        try:
            resampled = librosa.resample(
                waveform, 
                orig_sr=original_sr, 
                target_sr=self.target_sample_rate
            )
            return resampled
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            raise ValueError(f"Audio resampling failed: {e}")
    
    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range using peak normalization.
        
        This ensures that loud and quiet recordings produce similar
        features. Without this, a whisper might look very different
        from a shout even if they're the same voice.
        
        Peak normalization divides by the maximum absolute value,
        so the loudest sample becomes exactly 1.0 or -1.0.
        """
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def process_audio(self, audio_base64: str) -> np.ndarray:
        """
        Run the full audio processing pipeline.
        
        Takes a Base64 string and returns a clean, normalized waveform
        ready for feature extraction and model inference.
        
        Pipeline steps:
        1. Base64 decode -> raw MP3 bytes
        2. MP3 decode -> raw audio samples
        3. Resample -> 16kHz (if needed)
        4. Normalize -> peak at 1.0
        5. Pad -> ensure minimum length
        
        Args:
            audio_base64: Base64-encoded MP3 audio
            
        Returns:
            Normalized waveform as numpy array, shape (num_samples,)
        """
        # Step 1: Decode Base64 to get raw MP3 bytes
        audio_bytes = self.decode_base64(audio_base64)
        
        # Step 2: Decode MP3 to get raw audio samples
        waveform, sample_rate = self.mp3_to_waveform(audio_bytes)
        
        # Step 3: Resample to 16kHz if needed
        waveform = self.resample(waveform, sample_rate)
        
        # Step 4: Normalize to [-1, 1] range
        waveform = self.normalize(waveform)
        
        # Step 5: Handle very short audio clips
        # wav2vec2 needs a minimum length to work properly
        min_samples = int(0.1 * self.target_sample_rate)  # At least 100ms
        if len(waveform) < min_samples:
            # Pad with silence (zeros)
            padding = min_samples - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')
            logger.warning(f"Audio too short! Padded to {len(waveform)} samples")
        
        duration = len(waveform) / self.target_sample_rate
        logger.debug(f"Audio processed: {duration:.2f}s, {len(waveform):,} samples @ {self.target_sample_rate}Hz")
        return waveform
    
    def extract_features(self, waveform: np.ndarray) -> AudioFeatures:
        """
        Extract hand-crafted acoustic features from the waveform.
        
        These features capture different aspects of voice quality:
        
        1. MFCCs (Mel-Frequency Cepstral Coefficients)
           - The "fingerprint" of how a voice sounds
           - Based on how humans perceive different frequencies
           - 13 coefficients is standard for speech
        
        2. Pitch (F0 - Fundamental Frequency)
           - How high or low the voice is
           - mean: Average pitch (Hz)
           - std: How much it varies (KEY for AI detection!)
        
        3. Spectral Rolloff
           - Frequency below which 85% of the energy is contained
           - Low = mostly low frequencies (muffled)
           - High = lots of high frequency content (bright/harsh)
        
        4. Zero Crossing Rate
           - How often the signal crosses zero
           - Higher = noisier signal (breath, room noise)
           - Very low = suspiciously clean (synthetic?)
        
        Args:
            waveform: Preprocessed audio (16kHz, normalized)
            
        Returns:
            AudioFeatures dataclass with all extracted features
        """
        try:
            # MFCCs - Mel-frequency cepstral coefficients
            mfccs = librosa.feature.mfcc(
                y=waveform, 
                sr=self.target_sample_rate, 
                n_mfcc=13
            )
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Pitch (F0) extraction using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.target_sample_rate
            )
            
            # Filter out unvoiced frames and NaN values
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                pitch_mean = float(np.mean(f0_valid))
                pitch_std = float(np.std(f0_valid))
            else:
                pitch_mean = 0.0
                pitch_std = 0.0
            
            # Spectral rolloff - frequency below which 85% of energy is contained
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=waveform, 
                sr=self.target_sample_rate,
                roll_percent=0.85
            )
            spectral_rolloff_mean = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate - measure of noisiness
            zcr = librosa.feature.zero_crossing_rate(waveform)
            zcr_mean = float(np.mean(zcr))
            
            # Duration in seconds
            duration = len(waveform) / self.target_sample_rate
            
            features = AudioFeatures(
                mfcc_mean=mfcc_mean,
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                spectral_rolloff_mean=spectral_rolloff_mean,
                zero_crossing_rate_mean=zcr_mean,
                duration=duration
            )
            
            logger.debug(f"Features extracted - Pitch: {pitch_mean:.0f}Hz ±{pitch_std:.0f}, ZCR: {zcr_mean:.3f}, Duration: {duration:.1f}s")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise ValueError(f"Failed to extract audio features: {e}")


# ============================================================================
# Singleton Pattern for Processor
# ============================================================================
#
# We use a module-level singleton so we don't re-initialize the processor
# for every request. The processor is stateless anyway, so this is safe.
# ============================================================================

_processor: Optional[AudioProcessor] = None


def get_processor(sample_rate: int = 16000) -> AudioProcessor:
    """
    Get the global AudioProcessor instance.
    
    Creates a new processor on first call, returns the existing one after that.
    This is the recommended way to get a processor instance.
    
    Args:
        sample_rate: Target sample rate (16000 for wav2vec2 compatibility)
        
    Returns:
        The global AudioProcessor singleton
    """
    global _processor
    if _processor is None:
        _processor = AudioProcessor(target_sample_rate=sample_rate)
    return _processor
