"""
Aerodrome AI Intelligence Module

Advanced AI integration for pattern recognition, prediction, and analysis
using Google Gemini 2.0 models with sophisticated capabilities.
"""

from .gemini_client import GeminiClient
from .pattern_recognition import PatternRecognitionEngine
from .prediction_engine import PredictionEngine

__all__ = [
    "GeminiClient",
    "PatternRecognitionEngine", 
    "PredictionEngine"
]