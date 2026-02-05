"""
Poker tools module.

Provides utilities for hand evaluation, equity calculation,
pot odds, and GTO strategy advice.
"""

from .hand_eval import (
    HandRank, HandEvaluation, HandEvaluator,
    evaluate_hand, compare_hands
)

from .equity import (
    EquityResult, Deck, HandRange, EquityCalculator,
    calculate_equity, calculate_equity_vs_range
)

from .pot_odds import (
    BetSize, PotOddsResult, EVResult, MDFResult,
    PotOddsCalculator,
    pot_odds, expected_value_call, minimum_defense_frequency,
    required_fold_equity
)

from .gto import (
    ActionRecommendation, GTORecommendation, PreflopRange, GTOAdvisor,
    get_open_range, should_open, should_3bet, should_4bet
)

__all__ = [
    # Hand evaluation
    'HandRank', 'HandEvaluation', 'HandEvaluator',
    'evaluate_hand', 'compare_hands',
    # Equity
    'EquityResult', 'Deck', 'HandRange', 'EquityCalculator',
    'calculate_equity', 'calculate_equity_vs_range',
    # Pot odds
    'BetSize', 'PotOddsResult', 'EVResult', 'MDFResult',
    'PotOddsCalculator',
    'pot_odds', 'expected_value_call', 'minimum_defense_frequency',
    'required_fold_equity',
    # GTO
    'ActionRecommendation', 'GTORecommendation', 'PreflopRange', 'GTOAdvisor',
    'get_open_range', 'should_open', 'should_3bet', 'should_4bet',
]
