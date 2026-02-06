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

from .board_texture import (
    BoardWetness, BoardPairedness, BoardHighness, BoardTexture, BoardAnalyzer,
    analyze_board, get_texture_description, is_favorable_cbet_board, get_draw_potential
)

from .bet_sizing import (
    HandStrengthCategory, BetPurpose, BetSizing, BetSizingAdvisor,
    get_preflop_sizing, get_postflop_sizing, get_value_sizing, get_bluff_sizing
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
    # Board texture
    'BoardWetness', 'BoardPairedness', 'BoardHighness', 'BoardTexture', 'BoardAnalyzer',
    'analyze_board', 'get_texture_description', 'is_favorable_cbet_board', 'get_draw_potential',
    # Bet sizing
    'HandStrengthCategory', 'BetPurpose', 'BetSizing', 'BetSizingAdvisor',
    'get_preflop_sizing', 'get_postflop_sizing', 'get_value_sizing', 'get_bluff_sizing',
]
