"""
Simple rule-based opponents for testing and evaluation.

Provides various opponent types:
- RandomPlayer: Acts randomly (baseline)
- CallingStation: Calls everything (Fish)
- TightPassive: Only plays premium hands, rarely raises (Nit)
- LooseAggressive: Plays many hands aggressively (LAG)
- TagBot: Plays tight-aggressive with basic strategy (TAG)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random

from ..core import (
    Card, HoleCards, Position, Street, ActionType, Action,
    GameState, PlayerState
)
from ..tools.hand_eval import evaluate_hand
from ..tools.gto import should_open, should_3bet


@dataclass
class OpponentConfig:
    """Configuration for opponent behavior."""
    vpip: float = 0.30  # % of hands voluntarily played
    pfr: float = 0.20   # % of hands raised preflop
    aggression: float = 1.5  # Aggression factor
    fold_to_cbet: float = 0.50  # % fold to c-bet
    bluff_frequency: float = 0.20  # % of bets that are bluffs


class RandomPlayer:
    """
    Completely random player - baseline for testing.

    Makes uniformly random decisions from legal actions.
    """

    def __init__(self, player_id: str = "random"):
        self._player_id = player_id

    @property
    def player_id(self) -> str:
        return self._player_id

    def decide(self, game_state: GameState) -> Action:
        """Make a random legal decision."""
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            return Action.fold()

        # get_legal_actions returns Action objects, just pick one randomly
        return random.choice(legal_actions)


class CallingStation:
    """
    Calling station (Fish) - calls almost everything.

    Characteristics:
    - Very high VPIP (60-80%)
    - Very low PFR (5-10%)
    - Rarely folds postflop
    - Almost never raises
    """

    def __init__(self, player_id: str = "fish", call_threshold: float = 0.90):
        self._player_id = player_id
        self.call_threshold = call_threshold  # % of time to call

    @property
    def player_id(self) -> str:
        return self._player_id

    def decide(self, game_state: GameState) -> Action:
        """Fish logic: call most things, rarely raise."""
        player = game_state.players.get(self._player_id)
        if not player:
            return Action.fold()

        to_call = game_state.to_call
        legal = game_state.get_legal_actions()
        legal_types = {a.action_type for a in legal}

        # If can check, always check
        if ActionType.CHECK in legal_types:
            return Action.check()

        # Call most of the time
        if random.random() < self.call_threshold:
            if ActionType.CALL in legal_types:
                return Action.call(to_call)
            elif ActionType.ALL_IN in legal_types and to_call >= player.stack:
                return Action.all_in(player.stack)

        # Occasionally raise with very strong hands
        if game_state.street == Street.PREFLOP:
            if player.hole_cards and self._is_premium(player.hole_cards):
                if random.random() < 0.3:  # 30% of time with premiums
                    if ActionType.RAISE in legal_types:
                        return Action.raise_to(to_call + game_state.table.big_blind * 3)

        # Default: call if possible, else fold
        if ActionType.CALL in legal_types:
            return Action.call(to_call)
        return Action.fold()

    def _is_premium(self, cards: HoleCards) -> bool:
        """Check if hand is premium."""
        return cards.notation in ('AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo')


class TightPassive:
    """
    Tight-passive player (Nit) - only plays premium hands, rarely raises.

    Characteristics:
    - Low VPIP (10-15%)
    - Low PFR (8-12%)
    - High fold frequency
    - Only bets/raises with very strong hands
    """

    def __init__(self, player_id: str = "nit"):
        self._player_id = player_id
        self.premium_hands = {
            'AA', 'KK', 'QQ', 'JJ', 'TT', '99',
            'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'KQs'
        }

    @property
    def player_id(self) -> str:
        return self._player_id

    def decide(self, game_state: GameState) -> Action:
        """Nit logic: only play premiums, fold marginal hands."""
        player = game_state.players.get(self._player_id)
        if not player or not player.hole_cards:
            return Action.fold()

        hand = player.hole_cards.notation
        to_call = game_state.to_call
        legal = game_state.get_legal_actions()
        legal_types = {a.action_type for a in legal}

        # Preflop
        if game_state.street == Street.PREFLOP:
            if hand in self.premium_hands:
                # Raise with premiums
                if ActionType.RAISE in legal_types:
                    return Action.raise_to(to_call + game_state.table.big_blind * 3)
                elif ActionType.CALL in legal_types:
                    return Action.call(to_call)
            elif hand in {'88', '77', '66', 'ATs', 'KJs'}:
                # Call with medium hands
                if to_call <= game_state.table.big_blind * 2:
                    if ActionType.CALL in legal_types:
                        return Action.call(to_call)
            # Fold everything else
            return Action.fold() if ActionType.FOLD in legal_types else Action.check()

        # Postflop: check or fold unless very strong
        if ActionType.CHECK in legal_types:
            return Action.check()

        # Only continue with top pair or better
        if player.hole_cards and game_state.board.cards:
            result = evaluate_hand(player.hole_cards, game_state.board)
            if result.rank.value >= 2:  # Pair or better
                if random.random() < 0.6:  # 60% continue
                    if ActionType.CALL in legal_types:
                        return Action.call(to_call)

        return Action.fold()


class LooseAggressive:
    """
    Loose-aggressive player (LAG) - plays many hands aggressively.

    Characteristics:
    - High VPIP (35-45%)
    - High PFR (25-35%)
    - Frequent bluffs
    - Likes to bet and raise
    """

    def __init__(self, player_id: str = "lag", aggression: float = 2.5):
        self._player_id = player_id
        self.aggression = aggression
        self.open_range = 0.35  # 35% of hands

    @property
    def player_id(self) -> str:
        return self._player_id

    def decide(self, game_state: GameState) -> Action:
        """LAG logic: play many hands, bet/raise frequently."""
        player = game_state.players.get(self._player_id)
        if not player:
            return Action.fold()

        to_call = game_state.to_call
        legal = game_state.get_legal_actions()
        legal_types = {a.action_type for a in legal}
        pot = game_state.pot.total

        # Preflop
        if game_state.street == Street.PREFLOP:
            # Open wide range
            if to_call <= game_state.table.big_blind and random.random() < self.open_range:
                if ActionType.RAISE in legal_types:
                    return Action.raise_to(game_state.table.big_blind * 3)

            # 3-bet fairly often
            if to_call > game_state.table.big_blind and random.random() < 0.15:
                if ActionType.RAISE in legal_types:
                    return Action.raise_to(to_call * 3)

            # Call with decent hands
            if random.random() < 0.40:
                if ActionType.CALL in legal_types:
                    return Action.call(to_call)

            return Action.fold() if ActionType.FOLD in legal_types else Action.check()

        # Postflop: bet/raise frequently
        if ActionType.CHECK in legal_types:
            # Bet often when checked to
            if random.random() < 0.65:
                if ActionType.BET in legal_types:
                    bet_size = pot * random.choice([0.5, 0.67, 0.75])
                    return Action.bet(bet_size)
            return Action.check()

        # Facing a bet: raise or call aggressively
        if random.random() < 0.25:  # 25% raise
            if ActionType.RAISE in legal_types:
                return Action.raise_to(to_call * 2.5)

        if random.random() < 0.60:  # 60% call
            if ActionType.CALL in legal_types:
                return Action.call(to_call)

        return Action.fold()


class TagBot:
    """
    Tight-aggressive bot (TAG) - solid fundamental strategy.

    Characteristics:
    - Medium VPIP (20-25%)
    - Medium-high PFR (18-22%)
    - Position aware
    - Bets for value, some bluffs
    """

    def __init__(self, player_id: str = "tag"):
        self._player_id = player_id

    @property
    def player_id(self) -> str:
        return self._player_id

    def decide(self, game_state: GameState) -> Action:
        """TAG logic: play solid positional poker."""
        player = game_state.players.get(self._player_id)
        if not player or not player.hole_cards:
            return Action.fold()

        to_call = game_state.to_call
        legal = game_state.get_legal_actions()
        legal_types = {a.action_type for a in legal}
        position = player.position

        # Preflop
        if game_state.street == Street.PREFLOP:
            return self._preflop_decision(player, game_state, legal_types)

        # Postflop
        return self._postflop_decision(player, game_state, legal_types)

    def _preflop_decision(self, player: PlayerState, game: GameState,
                          legal_types: set[ActionType]) -> Action:
        """Make preflop decision based on position and hand."""
        from ..tools.gto import ActionRecommendation

        hand = player.hole_cards
        to_call = game.to_call
        bb = game.table.big_blind

        # Check GTO ranges
        if to_call <= bb:
            # No raise in front - consider opening
            rec = should_open(hand, player.position)
            if rec.primary_action == ActionRecommendation.RAISE:
                if ActionType.RAISE in legal_types:
                    return Action.raise_to(bb * 2.5)

        elif to_call > bb:
            # Facing a raise - consider 3-betting
            rec = should_3bet(hand, player.position, Position.CO)  # Assume CO opened
            if rec.primary_action == ActionRecommendation.RAISE:
                if ActionType.RAISE in legal_types:
                    return Action.raise_to(to_call * 3)
            elif rec.primary_action == ActionRecommendation.CALL:
                if ActionType.CALL in legal_types:
                    return Action.call(to_call)

        # Default fold
        return Action.fold() if ActionType.FOLD in legal_types else Action.check()

    def _postflop_decision(self, player: PlayerState, game: GameState,
                           legal_types: set[ActionType]) -> Action:
        """Make postflop decision based on hand strength."""
        to_call = game.to_call
        pot = game.pot.total

        # Evaluate hand
        result = evaluate_hand(player.hole_cards, game.board)
        hand_strength = result.rank.value  # 1-10 scale

        # Strong hand (two pair+) - bet/raise for value
        if hand_strength >= 3:
            if ActionType.CHECK in legal_types:
                if ActionType.BET in legal_types:
                    return Action.bet(pot * 0.67)
            else:
                if random.random() < 0.4 and ActionType.RAISE in legal_types:
                    return Action.raise_to(to_call * 2.5)
                elif ActionType.CALL in legal_types:
                    return Action.call(to_call)

        # Medium hand (pair) - check/call
        elif hand_strength >= 2:
            if ActionType.CHECK in legal_types:
                return Action.check()
            elif to_call < pot * 0.5 and ActionType.CALL in legal_types:
                return Action.call(to_call)

        # Weak hand - check/fold, occasional bluff
        if ActionType.CHECK in legal_types:
            if random.random() < 0.25:  # 25% bluff
                if ActionType.BET in legal_types:
                    return Action.bet(pot * 0.5)
            return Action.check()

        return Action.fold()


def create_opponent(opponent_type: str, player_id: Optional[str] = None) -> object:
    """Factory function to create opponents by type."""
    types = {
        "random": RandomPlayer,
        "fish": CallingStation,
        "calling_station": CallingStation,
        "nit": TightPassive,
        "tight_passive": TightPassive,
        "lag": LooseAggressive,
        "loose_aggressive": LooseAggressive,
        "tag": TagBot,
        "tight_aggressive": TagBot,
    }

    cls = types.get(opponent_type.lower())
    if cls is None:
        raise ValueError(f"Unknown opponent type: {opponent_type}. "
                        f"Available: {list(types.keys())}")

    pid = player_id or opponent_type
    return cls(player_id=pid)
