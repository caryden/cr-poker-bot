"""Tests for pot odds calculator."""

import pytest
import math
from src.tools.pot_odds import (
    PotOddsCalculator, PotOddsResult, EVResult, MDFResult,
    pot_odds, expected_value_call, minimum_defense_frequency,
    required_fold_equity
)


class TestPotOddsCalculator:
    """Tests for PotOddsCalculator."""

    @pytest.fixture
    def calc(self):
        return PotOddsCalculator()

    def test_pot_odds_basic(self, calc):
        # Pot is 100, need to call 50
        # Pot odds = 50 / (100 + 50) = 33%
        result = calc.pot_odds(100, 50)

        assert math.isclose(result.pot_odds, 0.333, rel_tol=0.01)
        assert result.to_call == 50
        assert result.pot_after_call == 150

    def test_pot_odds_2_to_1(self, calc):
        # 2:1 odds = need 33% equity
        result = calc.pot_odds(200, 100)

        assert math.isclose(result.pot_odds, 0.333, rel_tol=0.01)
        assert "2.0:1" in result.pot_odds_ratio

    def test_pot_odds_3_to_1(self, calc):
        # 3:1 odds = need 25% equity
        result = calc.pot_odds(300, 100)

        assert math.isclose(result.pot_odds, 0.25, rel_tol=0.01)
        assert "3.0:1" in result.pot_odds_ratio

    def test_pot_odds_free(self, calc):
        # No bet to call
        result = calc.pot_odds(100, 0)

        assert result.pot_odds == 0.0
        assert result.pot_odds_ratio == "free"

    def test_ev_call_positive(self, calc):
        # Pot 100, call 50, 50% equity
        # EV = 0.5 * 100 - 0.5 * 50 = 50 - 25 = 25
        result = calc.expected_value_call(100, 50, 0.5)

        assert result.ev > 0
        assert result.is_profitable

    def test_ev_call_negative(self, calc):
        # Pot 100, call 100, 25% equity
        # EV = 0.25 * 100 - 0.75 * 100 = 25 - 75 = -50
        result = calc.expected_value_call(100, 100, 0.25)

        assert result.ev < 0
        assert not result.is_profitable

    def test_ev_call_break_even(self, calc):
        # Pot 100, call 50, 33.3% equity
        # EV = 0.333 * 100 - 0.667 * 50 ≈ 0
        result = calc.expected_value_call(100, 50, 1/3)

        assert abs(result.ev) < 1  # Close to break-even

    def test_ev_bet(self, calc):
        # Pot 100, bet 50, 50% fold equity, 40% equity when called
        result = calc.expected_value_bet(100, 50, 0.5, 0.4)

        # EV = 0.5 * 100 + 0.5 * (0.4 * 150 - 0.6 * 50)
        # = 50 + 0.5 * (60 - 30) = 50 + 15 = 65
        assert result.ev > 0

    def test_mdf_pot_size_bet(self, calc):
        # Pot 100, opponent bets 100 (pot-size)
        # MDF = 100 / (100 + 100) = 50%
        result = calc.minimum_defense_frequency(100, 100)

        assert math.isclose(result.mdf, 0.5, rel_tol=0.01)
        assert math.isclose(result.fold_frequency, 0.5, rel_tol=0.01)

    def test_mdf_half_pot_bet(self, calc):
        # Pot 100, opponent bets 50 (half pot)
        # MDF = 100 / (100 + 50) = 67%
        result = calc.minimum_defense_frequency(50, 100)

        assert math.isclose(result.mdf, 0.667, rel_tol=0.01)

    def test_mdf_overbet(self, calc):
        # Pot 100, opponent bets 200 (2x pot)
        # MDF = 100 / (100 + 200) = 33%
        result = calc.minimum_defense_frequency(200, 100)

        assert math.isclose(result.mdf, 0.333, rel_tol=0.01)

    def test_required_fold_equity_pure_bluff(self, calc):
        # Pot 100, bet 100, 0% equity when called
        # Need 50% fold equity
        fe = calc.required_fold_equity(100, 100, equity_when_called=0)

        assert math.isclose(fe, 0.5, rel_tol=0.01)

    def test_required_fold_equity_with_equity(self, calc):
        # If we have equity when called, need less fold equity
        fe_no_eq = calc.required_fold_equity(100, 100, 0.0)
        fe_with_eq = calc.required_fold_equity(100, 100, 0.3)

        assert fe_with_eq < fe_no_eq

    def test_bet_size_for_pot_odds(self, calc):
        # Want to give 25% pot odds
        # If pot is 100, bet = 0.25 * 100 / (1 - 0.25) = 33.3
        bet = calc.bet_size_for_pot_odds(100, 0.25)

        assert math.isclose(bet, 33.33, rel_tol=0.01)

    def test_spr_calculation(self, calc):
        spr = calc.stack_to_pot_ratio(100, 50)
        assert spr == 2.0

        spr_deep = calc.stack_to_pot_ratio(500, 50)
        assert spr_deep == 10.0

    def test_commitment_threshold(self, calc):
        assert "committed" in calc.commitment_threshold(1.5).lower()
        assert "deep" in calc.commitment_threshold(15).lower()


class TestConvenienceFunctions:
    """Tests for module convenience functions."""

    def test_pot_odds_function(self):
        result = pot_odds(100, 50)
        assert isinstance(result, PotOddsResult)
        assert result.to_call == 50

    def test_ev_call_function(self):
        result = expected_value_call(100, 50, 0.5)
        assert isinstance(result, EVResult)

    def test_mdf_function(self):
        result = minimum_defense_frequency(100, 100)
        assert isinstance(result, MDFResult)

    def test_required_fe_function(self):
        fe = required_fold_equity(100, 100)
        assert 0 <= fe <= 1
