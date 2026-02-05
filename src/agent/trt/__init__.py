"""
Test-time Recursive Thinking (TRT) module.

Provides strategy-conditioned reasoning with self-verification.
"""

from .strategy import (
    Strategy, StrategyEvaluation, RolloutResult, TRTDecision, TRTEngine
)

__all__ = [
    'Strategy', 'StrategyEvaluation', 'RolloutResult', 'TRTDecision', 'TRTEngine'
]
