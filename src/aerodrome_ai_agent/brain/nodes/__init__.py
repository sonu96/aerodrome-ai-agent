"""
Brain Nodes - Individual cognitive functions for the LangGraph state machine

Each node represents a specific cognitive function in the brain's decision-making process:
- Observation: Market data collection and portfolio state monitoring
- Memory: Pattern recall and historical analysis  
- Analysis: Opportunity identification and scoring
- Decision: Risk-adjusted decision making
- Execution: Transaction building and blockchain interaction
- Learning: Pattern extraction from results
"""

from .observation import ObservationNode
from .memory import MemoryNode
from .analysis import AnalysisNode
from .decision import DecisionNode
from .execution import ExecutionNode
from .learning import LearningNode
from .monitoring import MonitoringNode

__all__ = [
    "ObservationNode",
    "MemoryNode", 
    "AnalysisNode",
    "DecisionNode",
    "ExecutionNode",
    "LearningNode",
    "MonitoringNode",
]