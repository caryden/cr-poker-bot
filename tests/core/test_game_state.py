"""Tests for game state management."""

import pytest
from src.core.game_state import (
    PlayerStatus, PlayerState, TableState, GameState,
    create_6max_game, ActionRecord
)
from src.core.primitives import (
    Position, Street, Action, ActionType, Card, Rank, Suit, HoleCards
)


class TestPlayerState:
    """Tests for PlayerState class."""

    def test_player_creation(self):
        player = PlayerState(
            player_id="hero",
            position=Position.BTN,
            stack=100.0
        )
        assert player.player_id == "hero"
        assert player.position == Position.BTN
        assert player.stack == 100.0
        assert player.is_active

    def test_player_status_active(self):
        player = PlayerState("p1", Position.BTN, 100.0)
        assert player.is_active
        assert player.is_in_hand

    def test_player_status_folded(self):
        player = PlayerState("p1", Position.BTN, 100.0, status=PlayerStatus.FOLDED)
        assert not player.is_active
        assert not player.is_in_hand

    def test_player_status_all_in(self):
        player = PlayerState("p1", Position.BTN, 0.0, status=PlayerStatus.ALL_IN)
        assert not player.is_active  # Can't act anymore
        assert player.is_in_hand  # Still in hand though

    def test_reset_street_bet(self):
        player = PlayerState("p1", Position.BTN, 100.0)
        player.bet_this_street = 50.0
        player.reset_street_bet()
        assert player.bet_this_street == 0.0


class TestTableState:
    """Tests for TableState class."""

    def test_table_creation(self):
        table = TableState(
            table_id="test_table",
            small_blind=0.5,
            big_blind=1.0
        )
        assert table.max_players == 6
        assert table.small_blind == 0.5
        assert table.big_blind == 1.0

    def test_rotate_button(self):
        table = TableState("test", btn_seat=0)
        table.rotate_button()
        assert table.btn_seat == 1

        table.btn_seat = 5
        table.rotate_button()
        assert table.btn_seat == 0  # Wraps around


class TestGameState:
    """Tests for GameState class."""

    @pytest.fixture
    def sample_game(self):
        """Create a sample 6-max game for testing."""
        stacks = {
            Position.UTG: 100.0,
            Position.HJ: 100.0,
            Position.CO: 100.0,
            Position.BTN: 100.0,
            Position.SB: 100.0,
            Position.BB: 100.0,
        }
        hero_cards = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS)
        )
        return create_6max_game(
            hand_id="test_hand_1",
            hero_id="hero",
            hero_position=Position.BTN,
            hero_cards=hero_cards,
            stacks=stacks,
            small_blind=0.5,
            big_blind=1.0
        )

    def test_game_creation(self, sample_game):
        assert sample_game.hand_id == "test_hand_1"
        assert sample_game.hero_id == "hero"
        assert sample_game.street == Street.PREFLOP
        assert sample_game.current_bet == 1.0  # BB

    def test_hero_property(self, sample_game):
        hero = sample_game.hero
        assert hero.player_id == "hero"
        assert hero.position == Position.BTN

    def test_villains_property(self, sample_game):
        villains = sample_game.villains
        assert len(villains) == 5
        assert all(v.player_id != "hero" for v in villains)

    def test_active_players(self, sample_game):
        active = sample_game.active_players
        assert len(active) == 6  # All players active initially

    def test_blinds_posted(self, sample_game):
        sb_player = sample_game.get_player_by_position(Position.SB)
        bb_player = sample_game.get_player_by_position(Position.BB)

        assert sb_player.bet_this_street == 0.5
        assert sb_player.stack == 99.5
        assert bb_player.bet_this_street == 1.0
        assert bb_player.stack == 99.0

    def test_pot_after_blinds(self, sample_game):
        assert sample_game.pot.total == 1.5  # SB + BB

    def test_to_call(self, sample_game):
        # Hero (BTN) needs to call BB
        assert sample_game.to_call == 1.0

    def test_pot_odds(self, sample_game):
        # Pot is 1.5, to call is 1.0
        # Pot odds = 1.0 / (1.5 + 1.0) = 0.4
        assert abs(sample_game.pot_odds - 0.4) < 0.01

    def test_effective_stack(self, sample_game):
        # All players have 100 (minus blinds)
        assert sample_game.effective_stack == 99.0  # SB has least

    def test_spr(self, sample_game):
        # Stack 99, pot 1.5 -> SPR = 66
        assert abs(sample_game.spr - 66.0) < 1.0

    def test_first_to_act_preflop(self, sample_game):
        # UTG acts first preflop
        acting = sample_game.acting_player
        player = sample_game.players[acting]
        assert player.position == Position.UTG

    def test_is_hero_turn(self, sample_game):
        # UTG acts first, not hero
        assert not sample_game.is_hero_turn()

    def test_get_legal_actions_facing_bet(self, sample_game):
        # UTG is acting, facing BB
        actions = sample_game.get_legal_actions()

        action_types = [a.action_type for a in actions]
        assert ActionType.FOLD in action_types
        assert ActionType.CALL in action_types
        assert ActionType.RAISE in action_types
        assert ActionType.CHECK not in action_types  # Can't check facing bet

    def test_to_dict_serialization(self, sample_game):
        data = sample_game.to_dict()

        assert data['hand_id'] == "test_hand_1"
        assert data['street'] == "preflop"
        assert data['pot'] == 1.5
        assert data['hero']['position'] == "BTN"
        assert len(data['villains']) == 5


class TestCreate6maxGame:
    """Tests for create_6max_game factory function."""

    def test_basic_game_creation(self):
        stacks = {pos: 100.0 for pos in Position.all_positions()}
        hero_cards = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )

        game = create_6max_game(
            hand_id="h1",
            hero_id="hero",
            hero_position=Position.CO,
            hero_cards=hero_cards,
            stacks=stacks
        )

        assert game.hero.position == Position.CO
        assert game.hero.hole_cards.is_pair
        assert game.hero.hole_cards.notation == "AA"

    def test_hero_cards_set_correctly(self):
        stacks = {pos: 100.0 for pos in Position.all_positions()}
        hero_cards = HoleCards(
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS)
        )

        game = create_6max_game(
            hand_id="h1",
            hero_id="hero",
            hero_position=Position.UTG,
            hero_cards=hero_cards,
            stacks=stacks
        )

        assert game.hero.hole_cards is not None
        assert game.hero.hole_cards.is_suited

    def test_villain_cards_hidden(self):
        stacks = {pos: 100.0 for pos in Position.all_positions()}
        hero_cards = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        )

        game = create_6max_game(
            hand_id="h1",
            hero_id="hero",
            hero_position=Position.BTN,
            hero_cards=hero_cards,
            stacks=stacks
        )

        for villain in game.villains:
            assert villain.hole_cards is None

    def test_partial_table(self):
        # Only 4 players
        stacks = {
            Position.CO: 100.0,
            Position.BTN: 100.0,
            Position.SB: 100.0,
            Position.BB: 100.0,
        }
        hero_cards = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS)
        )

        game = create_6max_game(
            hand_id="h1",
            hero_id="hero",
            hero_position=Position.BTN,
            hero_cards=hero_cards,
            stacks=stacks
        )

        assert len(game.players) == 4
        assert game.num_active == 4


class TestHeadsUp:
    """Tests for heads-up scenarios."""

    def test_is_heads_up(self):
        stacks = {
            Position.SB: 100.0,
            Position.BB: 100.0,
        }
        hero_cards = HoleCards(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS)
        )

        game = create_6max_game(
            hand_id="h1",
            hero_id="hero",
            hero_position=Position.SB,
            hero_cards=hero_cards,
            stacks=stacks
        )

        assert game.is_heads_up
        assert game.num_in_hand == 2
