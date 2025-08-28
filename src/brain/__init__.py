"""
Aerodrome AI Brain Module

This module implements the LangGraph-based brain system for the Aerodrome AI agent.
The brain orchestrates market observation, analysis, decision-making, and execution
using a state machine approach.
"""

from .core import AerodromeBrain
from .state import BrainState, BrainConfig
from .algorithms import OpportunityScorer, RiskAssessor
from .errors import BrainError, ErrorHandler

__all__ = [
    'AerodromeBrain',
    'BrainState',
    'BrainConfig',
    'OpportunityScorer',
    'RiskAssessor',
    'BrainError',
    'ErrorHandler'
]