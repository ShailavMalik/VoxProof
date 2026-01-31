"""
Explanation Module - Making AI Decisions Understandable
========================================================

One of the biggest problems with ML models is they're black boxes.
Users get a "AI_GENERATED" label but have no idea WHY.

This module provides human-readable explanations based on the acoustic
features we extract. It's not perfect (the real decision comes from the
neural network), but it gives users SOMETHING to understand.

Key insight: We explain in terms of things humans can relate to:
- "Unnaturally stable pitch" (robotic monotone)
- "Suspiciously clean audio" (no background noise)
- "Limited frequency range" (sounds muffled or thin)

These are rule-based heuristics, not the actual model reasoning.
Think of them as "likely contributing factors" rather than "the reason."

Author: VoxProof Team
License: MIT
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from audio.processing import AudioFeatures
from model.model import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class ExplanationRule:
    """
    Defines a single rule for generating explanations.
    
    Each rule checks a specific acoustic property and provides
    different explanations depending on whether we classified
    the audio as AI or Human.
    
    Attributes:
        name: Short identifier for the rule (for logging)
        description: What this rule is checking
        check_fn: Lambda that takes AudioFeatures, returns True if rule applies
        ai_explanation: What to say if rule triggers AND we said AI_GENERATED
        human_explanation: What to say if rule triggers AND we said HUMAN
    """
    name: str
    description: str
    check_fn: callable
    ai_explanation: str
    human_explanation: str


class ExplanationGenerator:
    """
    Generates human-readable explanations for voice detection results.
    
    The neural network gives us a probability, but users want to know WHY.
    This module bridges that gap with rule-based explanations based on
    acoustic features.
    
    IMPORTANT: These explanations are approximations, not the true model
    reasoning. The neural network might be picking up on patterns that
    don't map neatly to our rules. Use with appropriate caveats!
    
    The thresholds below were tuned based on typical voice characteristics.
    You may need to adjust them for your specific use case.
    """
    
    # -------------------------------------------------------------------------
    # Thresholds - tweak these based on your data!
    # -------------------------------------------------------------------------
    # These values define what counts as "abnormal" for each feature
    
    # Pitch variance thresholds (in Hz)
    # Human speech typically varies 30-60 Hz in pitch
    PITCH_STD_LOW_THRESHOLD = 20.0      # Below this = suspiciously stable
    PITCH_STD_HIGH_THRESHOLD = 80.0     # Above this = unusually variable
    
    # Zero crossing rate thresholds
    # This correlates with high-frequency content and noise
    ZCR_LOW_THRESHOLD = 0.03            # Below this = too clean
    ZCR_HIGH_THRESHOLD = 0.15           # Above this = very noisy
    
    # Spectral rolloff thresholds (in Hz)
    # Where most of the audio energy is concentrated
    ROLLOFF_LOW_THRESHOLD = 2000.0      # Below this = lacking high frequencies
    ROLLOFF_HIGH_THRESHOLD = 6000.0     # Above this = unusually bright
    
    def __init__(self):
        self.rules = self._build_rules()
        
    def _build_rules(self) -> List[ExplanationRule]:
        """Build the set of explanation rules."""
        return [
            # Pitch variance rules
            ExplanationRule(
                name="pitch_stability",
                description="Analyzes pitch variation patterns",
                check_fn=lambda f: f.pitch_std < self.PITCH_STD_LOW_THRESHOLD,
                ai_explanation="Unnaturally stable pitch pattern detected - synthetic voices often lack natural pitch fluctuations",
                human_explanation="Natural pitch variation observed - consistent with human speech patterns"
            ),
            ExplanationRule(
                name="pitch_variability",
                description="Checks for natural pitch modulation",
                check_fn=lambda f: f.pitch_std > self.PITCH_STD_HIGH_THRESHOLD,
                ai_explanation="Exaggerated pitch variation detected - may indicate synthetic voice artifacts",
                human_explanation="Rich prosodic variation detected - natural human intonation patterns"
            ),
            
            # Zero crossing rate rules (noisiness/clarity)
            ExplanationRule(
                name="synthetic_clarity",
                description="Detects unnaturally clean audio",
                check_fn=lambda f: f.zero_crossing_rate_mean < self.ZCR_LOW_THRESHOLD,
                ai_explanation="Unusually clean signal detected - synthetic voices often lack natural ambient noise",
                human_explanation="Clean recording quality with natural clarity"
            ),
            ExplanationRule(
                name="noise_presence",
                description="Analyzes background noise characteristics",
                check_fn=lambda f: f.zero_crossing_rate_mean > self.ZCR_HIGH_THRESHOLD,
                ai_explanation="Unusual high-frequency content detected - possible synthesis artifacts",
                human_explanation="Natural ambient noise and breath sounds detected"
            ),
            
            # Spectral rolloff rules
            ExplanationRule(
                name="frequency_distribution",
                description="Analyzes spectral energy distribution",
                check_fn=lambda f: f.spectral_rolloff_mean < self.ROLLOFF_LOW_THRESHOLD,
                ai_explanation="Limited frequency range detected - synthetic voices may lack natural harmonic richness",
                human_explanation="Appropriate frequency distribution for speech"
            ),
            ExplanationRule(
                name="high_frequency_content",
                description="Checks high frequency characteristics",
                check_fn=lambda f: f.spectral_rolloff_mean > self.ROLLOFF_HIGH_THRESHOLD,
                ai_explanation="Extended high-frequency content - may indicate synthesis or post-processing",
                human_explanation="Natural spectral characteristics with rich harmonics"
            ),
            
            # Duration-based rules
            ExplanationRule(
                name="short_sample",
                description="Handles very short audio clips",
                check_fn=lambda f: f.duration < 1.0,
                ai_explanation="Very short sample - limited acoustic information available for confident classification",
                human_explanation="Very short sample - limited acoustic information available for confident classification"
            ),
            
            # MFCC-based rules (using mean of first coefficient as proxy for energy)
            ExplanationRule(
                name="energy_consistency",
                description="Analyzes energy envelope patterns",
                check_fn=lambda f: abs(f.mfcc_mean[0]) < 100 if len(f.mfcc_mean) > 0 else False,
                ai_explanation="Uniform energy envelope detected - robotic prosody characteristics",
                human_explanation="Natural energy dynamics in speech pattern"
            ),
        ]
    
    def generate_explanation(
        self, 
        features: AudioFeatures, 
        prediction: PredictionResult
    ) -> str:
        """
        Generate a human-readable explanation for the prediction.
        
        Args:
            features: Extracted acoustic features
            prediction: Model prediction result
            
        Returns:
            Explanation string
        """
        is_ai = prediction.classification == "AI_GENERATED"
        triggered_explanations: List[Tuple[str, float]] = []
        
        # Check each rule
        for rule in self.rules:
            try:
                if rule.check_fn(features):
                    explanation = rule.ai_explanation if is_ai else rule.human_explanation
                    triggered_explanations.append((explanation, 1.0))
            except Exception as e:
                logger.debug(f"Rule {rule.name} check failed: {e}")
                continue
        
        # Build final explanation
        if not triggered_explanations:
            # Default explanations
            if is_ai:
                return "Audio characteristics suggest AI-generated content based on neural network analysis"
            else:
                return "Audio characteristics are consistent with natural human speech patterns"
        
        # Combine explanations (take top 2-3 most relevant)
        unique_explanations = list(set(exp for exp, _ in triggered_explanations))[:3]
        
        if len(unique_explanations) == 1:
            return unique_explanations[0]
        
        # Combine multiple explanations
        base = unique_explanations[0]
        additional = "; ".join(unique_explanations[1:]).lower()
        
        if additional:
            return f"{base}. Additionally: {additional}"
        
        return base
    
    def get_detailed_analysis(
        self, 
        features: AudioFeatures, 
        prediction: PredictionResult
    ) -> dict:
        """
        Get detailed analysis breakdown for debugging/advanced users.
        
        Returns dict with all feature values and their interpretations.
        """
        analysis = {
            "classification": prediction.classification,
            "confidence": prediction.confidence_score,
            "features": {
                "pitch": {
                    "mean_hz": round(features.pitch_mean, 2),
                    "std_hz": round(features.pitch_std, 2),
                    "interpretation": self._interpret_pitch(features)
                },
                "spectral_rolloff": {
                    "mean_hz": round(features.spectral_rolloff_mean, 2),
                    "interpretation": self._interpret_rolloff(features)
                },
                "zero_crossing_rate": {
                    "mean": round(features.zero_crossing_rate_mean, 4),
                    "interpretation": self._interpret_zcr(features)
                },
                "duration_seconds": round(features.duration, 2)
            },
            "explanation": self.generate_explanation(features, prediction)
        }
        
        return analysis
    
    def _interpret_pitch(self, features: AudioFeatures) -> str:
        """Interpret pitch characteristics."""
        if features.pitch_std < self.PITCH_STD_LOW_THRESHOLD:
            return "Very stable (potentially synthetic)"
        elif features.pitch_std > self.PITCH_STD_HIGH_THRESHOLD:
            return "Highly variable (expressive or artifact)"
        else:
            return "Normal variation (natural range)"
    
    def _interpret_rolloff(self, features: AudioFeatures) -> str:
        """Interpret spectral rolloff."""
        if features.spectral_rolloff_mean < self.ROLLOFF_LOW_THRESHOLD:
            return "Low frequency content (limited harmonics)"
        elif features.spectral_rolloff_mean > self.ROLLOFF_HIGH_THRESHOLD:
            return "High frequency content (rich or processed)"
        else:
            return "Normal spectral distribution"
    
    def _interpret_zcr(self, features: AudioFeatures) -> str:
        """Interpret zero crossing rate."""
        if features.zero_crossing_rate_mean < self.ZCR_LOW_THRESHOLD:
            return "Very clean signal (potentially synthetic)"
        elif features.zero_crossing_rate_mean > self.ZCR_HIGH_THRESHOLD:
            return "Noisy signal (natural or artifacts)"
        else:
            return "Normal noise characteristics"


# Module-level singleton
_explainer: ExplanationGenerator = None


def get_explainer() -> ExplanationGenerator:
    """Get or create the explanation generator singleton."""
    global _explainer
    if _explainer is None:
        _explainer = ExplanationGenerator()
    return _explainer
