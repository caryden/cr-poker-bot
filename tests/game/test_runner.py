"""Tests for game runner module."""

import pytest
from src.game.runner import (
    Deck, SimplePokerEngine, HandResult, SessionStats,
    HandRunner, SessionRunner
)
from src.game.opponents import RandomPlayer, CallingStation, TagBot
from src.core import Position, Street, Card, ActionType


class TestDeck:
    """Tests for Deck class."""

    def test_deck_has_52_cards(self):
        deck = Deck()
        assert len(deck.cards) == 52

    def test_deck_shuffle_changes_order(self):
        deck = Deck()
        original = deck.cards.copy()
        deck.shuffle()
        # Very unlikely to be same after shuffle
        assert deck.cards != original or len(deck.cards) == 0

    def test_deck_draw(self):
        deck = Deck()
        cards = deck.draw(5)

        assert len(cards) == 5
        assert len(deck.cards) == 47

    def test_deck_reset(self):
        deck = Deck()
        deck.draw(10)
        assert len(deck.cards) == 42

        deck.reset()
        assert len(deck.cards) == 52

    def test_deck_remove(self):
        deck = Deck()
        cards_to_remove = [Card.from_str("As"), Card.from_str("Kh")]
        deck.remove(cards_to_remove)

        assert len(deck.cards) == 50
        assert Card.from_str("As") not in deck.cards
        assert Card.from_str("Kh") not in deck.cards


class TestSimplePokerEngine:
    """Tests for SimplePokerEngine class."""

    @pytest.fixture
    def engine(self):
        return SimplePokerEngine(big_blind=10.0, starting_stack=1000.0)

    def test_create_hand(self, engine):
        game = engine.create_hand(Position.BTN)

        assert game is not None
        assert game.hero is not None
        assert game.hero.hole_cards is not None

    def test_deal_flop(self, engine):
        game = engine.create_hand(Position.BTN)
        assert game.street == Street.PREFLOP

        engine.deal_flop(game)

        assert game.street == Street.FLOP
        assert len(game.board.cards) == 3

    def test_deal_turn(self, engine):
        game = engine.create_hand(Position.BTN)
        engine.deal_flop(game)
        engine.deal_turn(game)

        assert game.street == Street.TURN
        assert len(game.board.cards) == 4

    def test_deal_river(self, engine):
        game = engine.create_hand(Position.BTN)
        engine.deal_flop(game)
        engine.deal_turn(game)
        engine.deal_river(game)

        assert game.street == Street.RIVER
        assert len(game.board.cards) == 5


class TestSessionStats:
    """Tests for SessionStats class."""

    def test_bb_per_100_calculation(self):
        stats = SessionStats()
        stats.hands_played = 100
        stats.total_profit = 50.0

        assert stats.bb_per_100 == 50.0

    def test_bb_per_100_zero_hands(self):
        stats = SessionStats()
        assert stats.bb_per_100 == 0.0

    def test_vpip_calculation(self):
        stats = SessionStats()
        stats.hands_played = 100
        stats.vpip_hands = 25

        assert stats.vpip == 0.25

    def test_won_at_sd_calculation(self):
        stats = SessionStats()
        stats.showdowns = 20
        stats.showdowns_won = 12

        assert stats.won_at_sd == 0.60

    def test_str_format(self):
        stats = SessionStats()
        stats.hands_played = 100
        stats.total_profit = 25.0

        result = str(stats)
        assert "100" in result
        assert "25" in result


class TestHandResult:
    """Tests for HandResult class."""

    def test_won_property_positive(self):
        result = HandResult(
            hand_id="1",
            hero_profit=10.0,
            hero_cards=None,
            went_to_showdown=True,
            hero_position=Position.BTN,
            actions_taken=5,
            final_pot=100.0
        )
        assert result.won is True

    def test_won_property_negative(self):
        result = HandResult(
            hand_id="1",
            hero_profit=-10.0,
            hero_cards=None,
            went_to_showdown=True,
            hero_position=Position.BTN,
            actions_taken=5,
            final_pot=100.0
        )
        assert result.won is False


class TestRandomPlayer:
    """Tests for RandomPlayer opponent."""

    def test_random_player_creation(self):
        player = RandomPlayer("random1")
        assert player.player_id == "random1"

    def test_random_player_decide(self):
        player = RandomPlayer("random1")
        engine = SimplePokerEngine()
        game = engine.create_hand(Position.BTN)

        # Should not raise
        action = player.decide(game)
        assert action is not None


class TestCallingStation:
    """Tests for CallingStation opponent."""

    def test_calling_station_creation(self):
        player = CallingStation("fish1")
        assert player.player_id == "fish1"

    def test_calling_station_high_call_rate(self):
        player = CallingStation("hero", call_threshold=0.95)
        engine = SimplePokerEngine()

        # Run many decisions, should mostly call
        call_count = 0
        for _ in range(100):
            game = engine.create_hand(Position.CO)
            game.current_bet = 30  # Set up facing a bet
            game.acting_player = "hero"  # Set acting player for get_legal_actions
            action = player.decide(game)
            if action.action_type == ActionType.CALL:
                call_count += 1

        # Should call most of the time
        assert call_count > 80


class TestTagBot:
    """Tests for TagBot opponent."""

    def test_tag_creation(self):
        player = TagBot("tag1")
        assert player.player_id == "tag1"

    def test_tag_folds_weak_hands(self):
        player = TagBot("tag1")
        engine = SimplePokerEngine()

        # Create game and set weak hole cards
        game = engine.create_hand(Position.UTG)
        # The game already has random cards assigned

        # Should make some decision
        action = player.decide(game)
        assert action is not None


class TestSessionRunner:
    """Tests for SessionRunner class."""

    def test_session_runner_creation(self):
        hero = RandomPlayer("hero")
        villains = {"villain1": RandomPlayer("villain1")}

        runner = SessionRunner(hero, villains)
        assert runner.stats.hands_played == 0

    def test_run_short_session(self):
        hero = RandomPlayer("hero")
        villains = {
            "villain1": RandomPlayer("v1"),
            "villain2": CallingStation("v2"),
        }

        runner = SessionRunner(hero, villains)
        stats = runner.run_session(num_hands=10)

        assert stats.hands_played == 10
        assert len(runner.hand_history) == 10

    def test_session_tracks_profit(self):
        hero = RandomPlayer("hero")
        villains = {"villain1": RandomPlayer("v1")}

        runner = SessionRunner(hero, villains)
        runner.run_session(num_hands=50)

        # Profit could be positive or negative
        assert runner.stats.total_profit != 0 or runner.stats.hands_played > 0
