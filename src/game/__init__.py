"""
Game module for poker bot.

Provides game runners, opponents, and evaluation tools.
"""

from .runner import (
    HandResult, SessionStats,
    Deck, SimplePokerEngine,
    HandRunner, SessionRunner
)

from .opponents import (
    RandomPlayer, CallingStation, TightPassive,
    LooseAggressive, TagBot, create_opponent
)

__all__ = [
    # Runner
    'HandResult', 'SessionStats',
    'Deck', 'SimplePokerEngine',
    'HandRunner', 'SessionRunner',
    # Opponents
    'RandomPlayer', 'CallingStation', 'TightPassive',
    'LooseAggressive', 'TagBot', 'create_opponent',
]
