"""Tests for core primitive types."""

import pytest
from src.core.primitives import (
    Suit, Rank, Card, HoleCards, Street, Position,
    ActionType, Action, Board, Pot
)


class TestCard:
    """Tests for Card class."""

    def test_card_creation(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

    def test_card_str(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert str(card) == "As"

        card2 = Card(Rank.TEN, Suit.HEARTS)
        assert str(card2) == "Th"

    def test_card_from_str(self):
        card = Card.from_str("As")
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

        card2 = Card.from_str("Th")
        assert card2.rank == Rank.TEN
        assert card2.suit == Suit.HEARTS

        card3 = Card.from_str("2c")
        assert card3.rank == Rank.TWO
        assert card3.suit == Suit.CLUBS

    def test_card_from_str_invalid(self):
        with pytest.raises(ValueError):
            Card.from_str("X")

        with pytest.raises(ValueError):
            Card.from_str("Ax")  # Invalid suit

    def test_card_hashable(self):
        card1 = Card(Rank.ACE, Suit.SPADES)
        card2 = Card(Rank.ACE, Suit.SPADES)
        assert card1 == card2
        assert hash(card1) == hash(card2)

        # Can be used in sets
        card_set = {card1, card2}
        assert len(card_set) == 1


class TestRank:
    """Tests for Rank enum."""

    def test_rank_str(self):
        assert str(Rank.ACE) == "A"
        assert str(Rank.KING) == "K"
        assert str(Rank.TEN) == "T"
        assert str(Rank.TWO) == "2"

    def test_rank_from_char(self):
        assert Rank.from_char("A") == Rank.ACE
        assert Rank.from_char("a") == Rank.ACE
        assert Rank.from_char("T") == Rank.TEN
        assert Rank.from_char("2") == Rank.TWO

    def test_rank_values(self):
        assert Rank.TWO.value == 2
        assert Rank.ACE.value == 14
        assert Rank.ACE.value > Rank.KING.value


class TestHoleCards:
    """Tests for HoleCards class."""

    def test_hole_cards_creation(self):
        card1 = Card(Rank.ACE, Suit.SPADES)
        card2 = Card(Rank.KING, Suit.HEARTS)
        hole = HoleCards(card1, card2)
        assert str(hole) == "AsKh"

    def test_hole_cards_canonical_order(self):
        # Lower card first should be reordered
        card1 = Card(Rank.KING, Suit.HEARTS)
        card2 = Card(Rank.ACE, Suit.SPADES)
        hole = HoleCards(card1, card2)
        # Ace should be first after canonical ordering
        assert hole.card1.rank == Rank.ACE

    def test_hole_cards_is_pair(self):
        aa = HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS))
        assert aa.is_pair

        ak = HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
        assert not ak.is_pair

    def test_hole_cards_is_suited(self):
        aks = HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES))
        assert aks.is_suited

        ako = HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
        assert not ako.is_suited

    def test_hole_cards_notation(self):
        aa = HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS))
        assert aa.notation == "AA"

        aks = HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES))
        assert aks.notation == "AKs"

        ako = HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
        assert ako.notation == "AKo"

    def test_hole_cards_gap(self):
        # Connectors
        jt = HoleCards(Card(Rank.JACK, Suit.SPADES), Card(Rank.TEN, Suit.HEARTS))
        assert jt.gap == 0

        # One-gapper
        j9 = HoleCards(Card(Rank.JACK, Suit.SPADES), Card(Rank.NINE, Suit.HEARTS))
        assert j9.gap == 1

    def test_hole_cards_from_str(self):
        hole = HoleCards.from_str("AsKh")
        assert hole.card1.rank == Rank.ACE
        assert hole.card2.rank == Rank.KING


class TestPosition:
    """Tests for Position enum."""

    def test_position_values(self):
        assert str(Position.UTG) == "UTG"
        assert str(Position.BTN) == "BTN"
        assert str(Position.BB) == "BB"

    def test_preflop_order(self):
        assert Position.UTG.preflop_order == 0
        assert Position.BB.preflop_order == 5

    def test_postflop_order(self):
        assert Position.SB.postflop_order == 0
        assert Position.BTN.postflop_order == 5

    def test_all_positions(self):
        positions = Position.all_positions()
        assert len(positions) == 6
        assert positions[0] == Position.UTG
        assert positions[-1] == Position.BB


class TestAction:
    """Tests for Action class."""

    def test_action_fold(self):
        action = Action.fold()
        assert action.action_type == ActionType.FOLD
        assert str(action) == "fold"

    def test_action_check(self):
        action = Action.check()
        assert action.action_type == ActionType.CHECK
        assert str(action) == "check"

    def test_action_call(self):
        action = Action.call(50)
        assert action.action_type == ActionType.CALL
        assert action.amount == 50
        assert str(action) == "call 50"

    def test_action_bet(self):
        action = Action.bet(100)
        assert action.action_type == ActionType.BET
        assert action.amount == 100
        assert str(action) == "bet 100"

    def test_action_raise(self):
        action = Action.raise_to(200)
        assert action.action_type == ActionType.RAISE
        assert action.amount == 200

    def test_action_all_in(self):
        action = Action.all_in(500)
        assert action.action_type == ActionType.ALL_IN
        assert action.is_all_in
        assert action.amount == 500

    def test_action_type_properties(self):
        assert ActionType.BET.is_aggressive
        assert ActionType.RAISE.is_aggressive
        assert not ActionType.CALL.is_aggressive

        assert ActionType.CHECK.is_passive
        assert ActionType.CALL.is_passive
        assert not ActionType.RAISE.is_passive


class TestBoard:
    """Tests for Board class."""

    def test_empty_board(self):
        board = Board()
        assert len(board) == 0
        assert board.street == Street.PREFLOP
        assert str(board) == "(empty)"

    def test_flop_board(self):
        board = Board()
        board.deal_flop(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS)
        )
        assert len(board) == 3
        assert board.street == Street.FLOP
        assert board.flop is not None

    def test_turn_board(self):
        board = Board()
        board.deal_flop(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS)
        )
        board.deal_turn(Card(Rank.JACK, Suit.CLUBS))
        assert len(board) == 4
        assert board.street == Street.TURN
        assert board.turn is not None

    def test_river_board(self):
        board = Board()
        board.deal_flop(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS)
        )
        board.deal_turn(Card(Rank.JACK, Suit.CLUBS))
        board.deal_river(Card(Rank.TEN, Suit.SPADES))
        assert len(board) == 5
        assert board.street == Street.RIVER
        assert board.river is not None

    def test_board_from_str(self):
        board = Board.from_str("As Kh Qd")
        assert len(board) == 3
        assert board.cards[0].rank == Rank.ACE

    def test_deal_turn_before_flop_raises(self):
        board = Board()
        with pytest.raises(ValueError):
            board.deal_turn(Card(Rank.ACE, Suit.SPADES))


class TestPot:
    """Tests for Pot class."""

    def test_pot_creation(self):
        pot = Pot()
        assert pot.main == 0
        assert pot.total == 0

    def test_pot_add(self):
        pot = Pot()
        pot.add(100)
        assert pot.main == 100
        assert pot.total == 100

        pot.add(50)
        assert pot.main == 150
        assert pot.total == 150
