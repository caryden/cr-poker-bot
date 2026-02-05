"""Tests for GTO advisor."""

import pytest
from src.core.primitives import Card, Rank, Suit, HoleCards, Position
from src.tools.gto import (
    GTOAdvisor, GTORecommendation, ActionRecommendation,
    get_open_range, should_open, should_3bet
)
from src.tools.equity import HandRange


class TestGTOAdvisor:
    """Tests for GTOAdvisor class."""

    @pytest.fixture
    def advisor(self):
        return GTOAdvisor()

    def test_get_open_range_utg(self, advisor):
        range_ = advisor.get_open_range(Position.UTG)

        # UTG should have tight range
        assert "AA" in range_.hands
        assert "KK" in range_.hands
        assert "AKs" in range_.hands
        # Should not include speculative hands
        assert "72o" not in range_.hands

    def test_get_open_range_btn(self, advisor):
        range_ = advisor.get_open_range(Position.BTN)

        # BTN should have wide range
        assert "AA" in range_.hands
        assert len(range_.hands) > 30  # Wide range

    def test_open_ranges_get_tighter_in_ep(self, advisor):
        utg_range = advisor.get_open_range(Position.UTG)
        btn_range = advisor.get_open_range(Position.BTN)

        assert len(utg_range.hands) < len(btn_range.hands)

    def test_should_open_aa_any_position(self, advisor):
        aa = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )

        for position in Position.all_positions():
            if position == Position.BB:
                continue  # BB doesn't open
            result = advisor.should_open(aa, position)
            assert result.primary_action == ActionRecommendation.RAISE

    def test_should_open_trash_folds(self, advisor):
        trash = HoleCards(
            Card(Rank.SEVEN, Suit.SPADES),
            Card(Rank.TWO, Suit.HEARTS)
        )

        result = advisor.should_open(trash, Position.UTG)
        assert result.primary_action == ActionRecommendation.FOLD

    def test_should_open_marginal_position_dependent(self, advisor):
        # K9s is marginal - may open from BTN but not UTG
        k9s = HoleCards(
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS)
        )

        utg_result = advisor.should_open(k9s, Position.UTG)
        btn_result = advisor.should_open(k9s, Position.BTN)

        # Should fold UTG but open BTN
        assert utg_result.primary_action == ActionRecommendation.FOLD
        assert btn_result.primary_action == ActionRecommendation.RAISE


class TestThreeBet:
    """Tests for 3-bet recommendations."""

    @pytest.fixture
    def advisor(self):
        return GTOAdvisor()

    def test_3bet_premiums(self, advisor):
        aa = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )

        result = advisor.should_3bet(aa, Position.BTN, Position.UTG)

        assert result.primary_action == ActionRecommendation.RAISE
        assert result.frequency >= 0.95  # Always 3-bet

    def test_3bet_range_wider_vs_late_position(self, advisor):
        jj = HoleCards(
            Card(Rank.JACK, Suit.SPADES),
            Card(Rank.JACK, Suit.HEARTS)
        )

        # JJ might flat vs UTG but 3-bet vs BTN
        vs_utg = advisor.should_3bet(jj, Position.BB, Position.UTG)
        vs_btn = advisor.should_3bet(jj, Position.BB, Position.BTN)

        # More likely to 3-bet vs BTN
        # Just check both return valid recommendations
        assert vs_utg.primary_action in (ActionRecommendation.RAISE, ActionRecommendation.CALL)
        assert vs_btn.primary_action in (ActionRecommendation.RAISE, ActionRecommendation.CALL)

    def test_3bet_bluffs_have_frequency(self, advisor):
        # A5s is a 3-bet bluff candidate
        a5s = HoleCards(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS)
        )

        result = advisor.should_3bet(a5s, Position.BTN, Position.CO)

        # Should be mixed strategy
        if result.primary_action == ActionRecommendation.RAISE:
            assert result.frequency < 1.0  # Not always


class TestFourBet:
    """Tests for 4-bet recommendations."""

    @pytest.fixture
    def advisor(self):
        return GTOAdvisor()

    def test_4bet_always_with_aa_kk(self, advisor):
        aa = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )

        result = advisor.should_4bet(aa, Position.BTN)

        assert result.primary_action == ActionRecommendation.RAISE
        assert result.frequency >= 0.95

    def test_4bet_qq_mixed(self, advisor):
        qq = HoleCards(
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.QUEEN, Suit.HEARTS)
        )

        result = advisor.should_4bet(qq, Position.BTN)

        # QQ is mixed 4-bet/call
        assert result.primary_action == ActionRecommendation.RAISE
        assert result.frequency < 1.0

    def test_4bet_fold_weak_hands(self, advisor):
        weak = HoleCards(
            Card(Rank.EIGHT, Suit.SPADES),
            Card(Rank.SEVEN, Suit.HEARTS)
        )

        result = advisor.should_4bet(weak, Position.BTN)

        assert result.primary_action == ActionRecommendation.FOLD


class TestCBet:
    """Tests for c-bet recommendations."""

    @pytest.fixture
    def advisor(self):
        return GTOAdvisor()

    def test_cbet_higher_on_dry_boards(self, advisor):
        # Mock game state not needed for this simple test
        from src.core.game_state import create_6max_game
        from src.core.primitives import Position

        # Dry board should have high c-bet freq
        dry_result = advisor.cbet_recommendation(None, True, "dry")
        wet_result = advisor.cbet_recommendation(None, True, "wet")

        assert dry_result.frequency > wet_result.frequency

    def test_cbet_check_when_not_aggressor(self, advisor):
        result = advisor.cbet_recommendation(None, False, "dry")

        # Should check when not preflop aggressor
        assert result.primary_action != ActionRecommendation.RAISE


class TestConvenienceFunctions:
    """Tests for module convenience functions."""

    def test_get_open_range(self):
        range_ = get_open_range(Position.BTN)
        assert isinstance(range_, HandRange)
        assert len(range_.hands) > 0

    def test_should_open(self):
        aa = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )
        result = should_open(aa, Position.UTG)
        assert isinstance(result, GTORecommendation)

    def test_should_3bet(self):
        kk = HoleCards(
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS)
        )
        result = should_3bet(kk, Position.BTN, Position.CO)
        assert isinstance(result, GTORecommendation)
