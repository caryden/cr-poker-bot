"""Tests for equity calculator."""

import pytest
from src.core.primitives import Card, Rank, Suit, HoleCards, Board
from src.tools.equity import (
    Deck, HandRange, EquityCalculator, EquityResult,
    calculate_equity
)


class TestDeck:
    """Tests for Deck class."""

    def test_deck_creation(self):
        deck = Deck()
        assert len(deck) == 52

    def test_deck_remove(self):
        deck = Deck()
        card = Card(Rank.ACE, Suit.SPADES)
        deck.remove([card])
        assert len(deck) == 51
        assert card not in deck.cards

    def test_deck_draw(self):
        deck = Deck()
        drawn = deck.draw(5)
        assert len(drawn) == 5
        assert len(deck) == 47
        for card in drawn:
            assert card not in deck.cards

    def test_deck_reset(self):
        deck = Deck()
        deck.draw(10)
        deck.reset()
        assert len(deck) == 52


class TestHandRange:
    """Tests for HandRange class."""

    def test_range_from_string_simple(self):
        range_ = HandRange.from_string("AA,KK,QQ")
        assert "AA" in range_
        assert "KK" in range_
        assert "QQ" in range_
        assert "JJ" not in range_

    def test_range_plus_notation_pairs(self):
        range_ = HandRange.from_string("JJ+")
        assert "AA" in range_
        assert "KK" in range_
        assert "QQ" in range_
        assert "JJ" in range_
        assert "TT" not in range_

    def test_range_get_combos(self):
        range_ = HandRange.from_string("AA")
        combos = range_.get_combos()
        # AA has 6 combos (4 choose 2)
        assert len(combos) == 6

    def test_range_get_combos_suited(self):
        range_ = HandRange.from_string("AKs")
        combos = range_.get_combos()
        # Suited has 4 combos (one per suit)
        assert len(combos) == 4
        for combo in combos:
            assert combo.is_suited

    def test_range_get_combos_offsuit(self):
        range_ = HandRange.from_string("AKo")
        combos = range_.get_combos()
        # Offsuit has 12 combos
        assert len(combos) == 12
        for combo in combos:
            assert not combo.is_suited

    def test_range_with_dead_cards(self):
        range_ = HandRange.from_string("AA")
        dead = [Card(Rank.ACE, Suit.SPADES)]
        combos = range_.get_combos(dead)
        # With one ace dead, only 3 combos remain
        assert len(combos) == 3

    def test_range_from_percentage(self):
        range_ = HandRange.from_percentage(5)
        # Top 5% should include AA, KK, QQ, JJ, AKs, AKo
        assert "AA" in range_
        assert "KK" in range_
        assert len(range_) > 0


class TestEquityCalculator:
    """Tests for EquityCalculator."""

    @pytest.fixture
    def calculator(self):
        return EquityCalculator()

    def test_aa_vs_random_preflop(self, calculator):
        """AA should have ~85% equity vs random hand preflop."""
        aa = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )
        board = Board()

        result = calculator.calculate_vs_random(aa, board, simulations=2000)

        # AA vs random should be around 85%, allow variance
        assert 0.75 < result.equity < 0.92

    def test_equity_result_properties(self, calculator):
        aa = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )
        board = Board()

        result = calculator.calculate_vs_random(aa, board, simulations=500)

        # Check result has expected properties
        assert 0 <= result.equity <= 1
        assert 0 <= result.win_pct <= 1
        assert 0 <= result.tie_pct <= 1
        assert 0 <= result.lose_pct <= 1
        assert abs(result.win_pct + result.tie_pct + result.lose_pct - 1.0) < 0.01

    def test_calculate_matchup(self, calculator):
        """Test specific hand matchup."""
        aa = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )
        kk = HoleCards(
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS)
        )

        eq_aa, eq_kk = calculator.calculate_matchup(aa, kk, simulations=1000)

        # AA vs KK is roughly 80/20
        assert eq_aa.equity > 0.75
        assert eq_kk.equity < 0.25
        assert abs(eq_aa.equity + eq_kk.equity - 1.0) < 0.01

    def test_calculate_vs_range(self, calculator):
        """Test equity vs specific range."""
        aa = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )
        board = Board()
        villain_range = HandRange.from_string("KK,QQ,JJ,AKs,AKo")

        result = calculator.calculate_vs_range(
            aa, board, villain_range, simulations=1000
        )

        # AA should have good equity vs this range
        assert result.equity > 0.65

    def test_postflop_equity(self, calculator):
        """Test equity calculation on flop."""
        # Set vs flush draw
        set_hand = HoleCards(
            Card(Rank.EIGHT, Suit.SPADES),
            Card(Rank.EIGHT, Suit.HEARTS)
        )
        flush_draw = HoleCards(
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.DIAMONDS)
        )
        board = Board([
            Card(Rank.EIGHT, Suit.DIAMONDS),
            Card(Rank.TWO, Suit.DIAMONDS),
            Card(Rank.FIVE, Suit.CLUBS),
        ])

        eq_set, eq_draw = calculator.calculate_matchup(
            set_hand, flush_draw, board, simulations=1000
        )

        # Set should be ahead of flush draw
        assert eq_set.equity > eq_draw.equity


class TestConvenienceFunctions:
    """Tests for module convenience functions."""

    def test_calculate_equity(self):
        hole = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES)
        )
        board = Board()

        result = calculate_equity(hole, board, simulations=1000)

        assert isinstance(result, EquityResult)
        assert 0.45 < result.equity < 0.80  # AKs vs random ~67% (wide range for MC variance)
