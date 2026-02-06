"""Tests for hand evaluation."""

import pytest
from src.core.primitives import Card, Rank, Suit, HoleCards, Board
from src.tools.hand_eval import (
    HandRank, HandEvaluation, HandEvaluator,
    evaluate_hand, compare_hands
)


class TestHandRank:
    """Tests for HandRank enum."""

    def test_rank_ordering(self):
        assert HandRank.HIGH_CARD < HandRank.PAIR
        assert HandRank.PAIR < HandRank.TWO_PAIR
        assert HandRank.STRAIGHT < HandRank.FLUSH
        assert HandRank.FLUSH < HandRank.FULL_HOUSE
        assert HandRank.FULL_HOUSE < HandRank.FOUR_OF_A_KIND
        assert HandRank.STRAIGHT_FLUSH < HandRank.ROYAL_FLUSH

    def test_rank_str(self):
        assert str(HandRank.HIGH_CARD) == "High Card"
        assert str(HandRank.FULL_HOUSE) == "Full House"


class TestHandEvaluator:
    """Tests for HandEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        return HandEvaluator()

    def test_high_card(self, evaluator):
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
            Card(Rank.JACK, Suit.CLUBS),
            Card(Rank.NINE, Suit.SPADES),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.HIGH_CARD
        assert "Ace" in result.description

    def test_pair(self, evaluator):
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.QUEEN, Suit.CLUBS),
            Card(Rank.JACK, Suit.SPADES),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.PAIR
        assert "Ace" in result.description

    def test_two_pair(self, evaluator):
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.SPADES),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.TWO_PAIR

    def test_three_of_a_kind(self, evaluator):
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.SPADES),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.THREE_OF_A_KIND

    def test_straight(self, evaluator):
        cards = [
            Card(Rank.TEN, Suit.SPADES),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.DIAMONDS),
            Card(Rank.SEVEN, Suit.CLUBS),
            Card(Rank.SIX, Suit.SPADES),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.STRAIGHT
        assert "Ten" in result.description

    def test_wheel_straight(self, evaluator):
        """Test A-2-3-4-5 (wheel) straight."""
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.SPADES),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.STRAIGHT
        assert "Five" in result.description  # 5-high straight

    def test_flush(self, evaluator):
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.FLUSH

    def test_full_house(self, evaluator):
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.KING, Suit.SPADES),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.FULL_HOUSE
        assert "Ace" in result.description
        assert "King" in result.description

    def test_four_of_a_kind(self, evaluator):
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.KING, Suit.SPADES),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.FOUR_OF_A_KIND

    def test_straight_flush(self, evaluator):
        cards = [
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.SIX, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.STRAIGHT_FLUSH

    def test_royal_flush(self, evaluator):
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS),
        ]
        result = evaluator.evaluate_5(cards)
        assert result.rank == HandRank.ROYAL_FLUSH


class TestHandComparison:
    """Tests for comparing hands."""

    @pytest.fixture
    def evaluator(self):
        return HandEvaluator()

    def test_pair_beats_high_card(self, evaluator):
        pair = [
            Card(Rank.TWO, Suit.SPADES),
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.SPADES),
        ]
        high_card = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
            Card(Rank.JACK, Suit.CLUBS),
            Card(Rank.NINE, Suit.SPADES),
        ]

        pair_eval = evaluator.evaluate_5(pair)
        high_eval = evaluator.evaluate_5(high_card)

        assert pair_eval > high_eval

    def test_higher_pair_wins(self, evaluator):
        aces = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.SPADES),
        ]
        kings = [
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.SPADES),
        ]

        aces_eval = evaluator.evaluate_5(aces)
        kings_eval = evaluator.evaluate_5(kings)

        assert aces_eval > kings_eval

    def test_kicker_decides(self, evaluator):
        ace_kicker = [
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.SPADES),
        ]
        queen_kicker = [
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS),
        ]

        ace_eval = evaluator.evaluate_5(ace_kicker)
        queen_eval = evaluator.evaluate_5(queen_kicker)

        assert ace_eval > queen_eval

    def test_same_hand_ties(self, evaluator):
        hand1 = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.JACK, Suit.SPADES),
            Card(Rank.NINE, Suit.SPADES),
        ]
        hand2 = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS),
        ]

        eval1 = evaluator.evaluate_5(hand1)
        eval2 = evaluator.evaluate_5(hand2)

        assert eval1 == eval2


class TestHoldemEvaluation:
    """Tests for 7-card Hold'em evaluation."""

    @pytest.fixture
    def evaluator(self):
        return HandEvaluator()

    def test_evaluate_holdem_basic(self, evaluator):
        hole = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES)
        )
        board = Board([
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
            Card(Rank.JACK, Suit.CLUBS),
            Card(Rank.TWO, Suit.SPADES),
        ])

        result = evaluator.evaluate_holdem(hole, board)
        assert result.rank == HandRank.TWO_PAIR

    def test_evaluate_holdem_flush(self, evaluator):
        hole = HoleCards(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.TWO, Suit.HEARTS)
        )
        board = Board([
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TEN, Suit.CLUBS),
            Card(Rank.NINE, Suit.DIAMONDS),
        ])

        result = evaluator.evaluate_holdem(hole, board)
        assert result.rank == HandRank.FLUSH

    def test_board_plays(self, evaluator):
        """Test when board makes best hand."""
        hole = HoleCards(
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS)
        )
        board = Board([
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.KING, Suit.SPADES),
        ])

        result = evaluator.evaluate_holdem(hole, board)
        assert result.rank == HandRank.FOUR_OF_A_KIND


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_evaluate_hand(self):
        hole = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )
        board = Board([
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.SPADES),
        ])

        result = evaluate_hand(hole, board)
        assert result.rank == HandRank.THREE_OF_A_KIND

    def test_compare_hands(self):
        hole1 = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )
        hole2 = HoleCards(
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS)
        )
        # Board that doesn't connect with either hand
        board = Board([
            Card(Rank.SEVEN, Suit.DIAMONDS),
            Card(Rank.FIVE, Suit.CLUBS),
            Card(Rank.THREE, Suit.SPADES),
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.NINE, Suit.DIAMONDS),
        ])

        eval1 = evaluate_hand(hole1, board)
        eval2 = evaluate_hand(hole2, board)

        assert compare_hands(eval1, eval2) == 1  # AA beats KK
        assert compare_hands(eval2, eval1) == -1
        assert compare_hands(eval1, eval1) == 0
