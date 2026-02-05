"""
Prompt templates module.

Provides structured prompts for the poker agent.
"""

from .templates import (
    SYSTEM_PROMPT,
    PREFLOP_TEMPLATE,
    POSTFLOP_TEMPLATE,
    DECISION_VERIFICATION_TEMPLATE,
    STRATEGY_CONDITIONED_TEMPLATE,
    PromptBuilder,
    format_villains,
    format_beliefs,
    format_legal_actions
)

__all__ = [
    'SYSTEM_PROMPT',
    'PREFLOP_TEMPLATE',
    'POSTFLOP_TEMPLATE',
    'DECISION_VERIFICATION_TEMPLATE',
    'STRATEGY_CONDITIONED_TEMPLATE',
    'PromptBuilder',
    'format_villains',
    'format_beliefs',
    'format_legal_actions',
]
