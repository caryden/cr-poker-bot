"""
Evaluation module for poker bot analysis.

Provides trace capture, session logging, and performance analysis.
"""

from .tracer import (
    DecisionTrace, HandTrace, SessionTrace,
    Tracer, TraceStore
)

from .analyzer import (
    PerformanceMetrics, DecisionAnalysis,
    SessionAnalyzer, compare_sessions
)

from .runner import (
    EvaluationConfig, EvaluationRunner, EvaluationResult,
    quick_eval, full_eval
)

__all__ = [
    # Tracing
    'DecisionTrace', 'HandTrace', 'SessionTrace',
    'Tracer', 'TraceStore',
    # Analysis
    'PerformanceMetrics', 'DecisionAnalysis',
    'SessionAnalyzer', 'compare_sessions',
    # Runner
    'EvaluationConfig', 'EvaluationRunner', 'EvaluationResult',
    'quick_eval', 'full_eval',
]
