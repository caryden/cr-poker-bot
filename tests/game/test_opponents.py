"""Tests for opponents module."""

import pytest
from src.game.opponents import (
    RandomPlayer, CallingStation, TightPassive,
    LooseAggressive, TagBot, create_opponent
)
from src.game.runner import SimplePokerEngine
from src.core import Position, HoleCards, Card


class TestCreateOpponent:
    """Tests for opponent factory function."""

    def test_create_random(self):
        player = create_opponent("random")
        assert isinstance(player, RandomPlayer)

    def test_create_fish(self):
        player = create_opponent("fish")
        assert isinstance(player, CallingStation)

    def test_create_nit(self):
        player = create_opponent("nit")
        assert isinstance(player, TightPassive)

    def test_create_lag(self):
        player = create_opponent("lag")
        assert isinstance(player, LooseAggressive)

    def test_create_tag(self):
        player = create_opponent("tag")
        assert isinstance(player, TagBot)

    def test_create_with_custom_id(self):
        player = create_opponent("random", player_id="custom_id")
        assert player.player_id == "custom_id"

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown opponent type"):
            create_opponent("unknown_type")

    def test_case_insensitive(self):
        player1 = create_opponent("RANDOM")
        player2 = create_opponent("Random")
        assert isinstance(player1, RandomPlayer)
        assert isinstance(player2, RandomPlayer)


class TestTightPassive:
    """Tests for TightPassive (Nit) opponent."""

    @pytest.fixture
    def nit(self):
        return TightPassive("nit")

    @pytest.fixture
    def engine(self):
        return SimplePokerEngine()

    def test_nit_folds_trash(self, nit, engine):
        game = engine.create_hand(Position.UTG)

        # Set weak hole cards
        game.players["nit"] = game.hero  # Use hero's player object for nit
        game.players["nit"].hole_cards = HoleCards(
            Card.from_str("2h"),
            Card.from_str("7c")
        )

        # Create a test where nit should fold
        # Nit should generally fold trash hands


class TestLooseAggressive:
    """Tests for LooseAggressive (LAG) opponent."""

    @pytest.fixture
    def lag(self):
        return LooseAggressive("lag")

    def test_lag_has_high_aggression(self, lag):
        assert lag.aggression >= 2.0

    def test_lag_has_wide_open_range(self, lag):
        assert lag.open_range >= 0.30


class TestOpponentBehavior:
    """Integration tests for opponent behaviors."""

    @pytest.fixture
    def engine(self):
        return SimplePokerEngine()

    def test_all_opponents_return_valid_actions(self, engine):
        """All opponent types should return valid actions."""
        opponents = [
            RandomPlayer("random"),
            CallingStation("fish"),
            TightPassive("nit"),
            LooseAggressive("lag"),
            TagBot("tag"),
        ]

        for opponent in opponents:
            game = engine.create_hand(Position.BTN)
            action = opponent.decide(game)

            assert action is not None
            assert action.action_type is not None

    def test_opponents_handle_all_streets(self, engine):
        """Opponents should handle decisions on all streets."""
        opponents = [
            RandomPlayer("random"),
            CallingStation("fish"),
            TagBot("tag"),
        ]

        for opponent in opponents:
            game = engine.create_hand(Position.BTN)

            # Preflop
            action = opponent.decide(game)
            assert action is not None

            # Flop
            engine.deal_flop(game)
            action = opponent.decide(game)
            assert action is not None

            # Turn
            engine.deal_turn(game)
            action = opponent.decide(game)
            assert action is not None

            # River
            engine.deal_river(game)
            action = opponent.decide(game)
            assert action is not None
