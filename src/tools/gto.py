"""
GTO (Game Theory Optimal) advisor for poker decisions.

Provides theoretically sound baseline recommendations for:
- Preflop opening ranges by position
- 3-bet and 4-bet ranges
- C-bet frequencies
- Postflop action recommendations

Note: True GTO requires solved equilibria. This module provides
approximations based on widely-accepted GTO principles.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

from ..core.primitives import Position, Street, HoleCards, Action, ActionType
from ..core.game_state import GameState
from .equity import HandRange


class ActionRecommendation(Enum):
    """Recommended action types."""
    FOLD = auto()
    CALL = auto()
    RAISE = auto()
    MIXED = auto()  # Mixed strategy


@dataclass
class GTORecommendation:
    """
    GTO-based action recommendation.

    Attributes:
        primary_action: Main recommended action
        frequency: How often to take this action (for mixed strategies)
        reasoning: Explanation of the recommendation
        alternative: Secondary action if using mixed strategy
        alt_frequency: Frequency of alternative action
        ev_estimate: Estimated EV if known
    """
    primary_action: ActionRecommendation
    frequency: float = 1.0
    reasoning: str = ""
    alternative: Optional[ActionRecommendation] = None
    alt_frequency: float = 0.0
    ev_estimate: Optional[float] = None

    def __str__(self) -> str:
        if self.frequency >= 0.95:
            return f"{self.primary_action.name}: {self.reasoning}"
        return (f"{self.primary_action.name} ({self.frequency:.0%}) / "
                f"{self.alternative.name} ({self.alt_frequency:.0%}): {self.reasoning}")


@dataclass
class PreflopRange:
    """
    Preflop range for a specific situation.

    Attributes:
        open_raise: Hands to open raise
        call: Hands to call (if facing raise)
        three_bet: Hands to 3-bet
        four_bet: Hands to 4-bet
    """
    open_raise: HandRange = field(default_factory=HandRange)
    call: HandRange = field(default_factory=HandRange)
    three_bet: HandRange = field(default_factory=HandRange)
    four_bet: HandRange = field(default_factory=HandRange)


class GTOAdvisor:
    """
    Provides GTO-based recommendations for poker decisions.

    Based on widely-accepted GTO approximations for 6-max NLHE.
    """

    # Preflop opening ranges by position (approximate GTO ranges)
    OPEN_RANGES = {
        Position.UTG: {
            'range': "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,A5s,A4s,"
                    "KQs,KJs,QJs,JTs,T9s,98s,87s,76s,AKo,AQo",
            'pct': 15
        },
        Position.HJ: {
            'range': "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AQs,AJs,ATs,A9s,A5s,A4s,A3s,"
                    "KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s,AKo,AQo,AJo,KQo",
            'pct': 19
        },
        Position.CO: {
            'range': "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
                    "KQs,KJs,KTs,K9s,QJs,QTs,Q9s,JTs,J9s,T9s,T8s,98s,97s,87s,86s,76s,75s,65s,54s,"
                    "AKo,AQo,AJo,ATo,KQo,KJo,QJo",
            'pct': 27
        },
        Position.BTN: {
            'range': "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
                    "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
                    "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,K3s,K2s,"
                    "QJs,QTs,Q9s,Q8s,Q7s,Q6s,JTs,J9s,J8s,J7s,T9s,T8s,T7s,"
                    "98s,97s,96s,87s,86s,85s,76s,75s,65s,64s,54s,53s,43s,"
                    "AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,"
                    "KQo,KJo,KTo,K9o,QJo,QTo,JTo",
            'pct': 45
        },
        Position.SB: {
            'range': "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
                    "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
                    "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,QJs,QTs,Q9s,Q8s,"
                    "JTs,J9s,J8s,T9s,T8s,98s,97s,87s,76s,65s,54s,"
                    "AKo,AQo,AJo,ATo,A9o,KQo,KJo,KTo,QJo,QTo,JTo",
            'pct': 40
        },
    }

    # 3-bet ranges by position vs position
    THREE_BET_RANGES = {
        # vs UTG open
        (Position.HJ, Position.UTG): "AA,KK,QQ,AKs,AKo",
        (Position.CO, Position.UTG): "AA,KK,QQ,JJ,AKs,AKo,AQs",
        (Position.BTN, Position.UTG): "AA,KK,QQ,JJ,TT,AKs,AKo,AQs,AQo,AJs,A5s,A4s",
        (Position.SB, Position.UTG): "AA,KK,QQ,JJ,AKs,AKo,AQs",
        (Position.BB, Position.UTG): "AA,KK,QQ,JJ,AKs,AKo,AQs,AJs,A5s",

        # vs HJ open
        (Position.CO, Position.HJ): "AA,KK,QQ,JJ,AKs,AKo,AQs,A5s",
        (Position.BTN, Position.HJ): "AA,KK,QQ,JJ,TT,99,AKs,AKo,AQs,AQo,AJs,ATs,A5s,A4s,KQs",
        (Position.SB, Position.HJ): "AA,KK,QQ,JJ,TT,AKs,AKo,AQs,AJs,A5s",
        (Position.BB, Position.HJ): "AA,KK,QQ,JJ,TT,AKs,AKo,AQs,AQo,AJs,A5s,A4s",

        # vs CO open
        (Position.BTN, Position.CO): "AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,AJs,ATs,A5s,A4s,A3s,KQs,KJs",
        (Position.SB, Position.CO): "AA,KK,QQ,JJ,TT,99,AKs,AKo,AQs,AQo,AJs,A5s,A4s,KQs",
        (Position.BB, Position.CO): "AA,KK,QQ,JJ,TT,99,AKs,AKo,AQs,AQo,AJs,ATs,A5s,A4s,A3s,KQs",

        # vs BTN open
        (Position.SB, Position.BTN): "AA,KK,QQ,JJ,TT,99,88,77,AKs,AKo,AQs,AQo,AJs,AJo,ATs,A9s,A8s,A5s,A4s,A3s,A2s,KQs,KQo,KJs,KTs,QJs,JTs",
        (Position.BB, Position.BTN): "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AKo,AQs,AQo,AJs,AJo,ATs,A9s,A5s,A4s,A3s,A2s,KQs,KQo,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s",
    }

    # 4-bet ranges (simplified)
    FOUR_BET_VALUE = "AA,KK,QQ,AKs"
    FOUR_BET_BLUFF = "A5s,A4s"  # Suited aces as bluffs

    def __init__(self):
        self._cached_ranges: dict[str, HandRange] = {}

    def get_open_range(self, position: Position) -> HandRange:
        """Get preflop opening range for a position."""
        if position not in self.OPEN_RANGES:
            return HandRange()

        cache_key = f"open_{position.value}"
        if cache_key not in self._cached_ranges:
            range_str = self.OPEN_RANGES[position]['range']
            self._cached_ranges[cache_key] = HandRange.from_string(range_str)

        return self._cached_ranges[cache_key]

    def get_3bet_range(self, hero_pos: Position, villain_pos: Position) -> HandRange:
        """Get 3-bet range for hero position vs villain position."""
        key = (hero_pos, villain_pos)
        if key not in self.THREE_BET_RANGES:
            return HandRange()

        cache_key = f"3bet_{hero_pos.value}_{villain_pos.value}"
        if cache_key not in self._cached_ranges:
            range_str = self.THREE_BET_RANGES[key]
            self._cached_ranges[cache_key] = HandRange.from_string(range_str)

        return self._cached_ranges[cache_key]

    def should_open(self, hand: HoleCards, position: Position) -> GTORecommendation:
        """
        Check if hand should open raise from position.

        Args:
            hand: Hero's hole cards
            position: Hero's position

        Returns:
            GTORecommendation
        """
        open_range = self.get_open_range(position)

        if hand.notation in open_range.hands:
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,
                reasoning=f"{hand.notation} is in {position.value} opening range"
            )
        else:
            return GTORecommendation(
                primary_action=ActionRecommendation.FOLD,
                reasoning=f"{hand.notation} is outside {position.value} opening range"
            )

    def should_3bet(
        self,
        hand: HoleCards,
        hero_pos: Position,
        villain_pos: Position
    ) -> GTORecommendation:
        """
        Check if hand should 3-bet vs an open.

        Args:
            hand: Hero's hole cards
            hero_pos: Hero's position
            villain_pos: Original raiser's position

        Returns:
            GTORecommendation
        """
        three_bet_range = self.get_3bet_range(hero_pos, villain_pos)
        open_range = self.get_open_range(hero_pos)

        notation = hand.notation

        if notation in three_bet_range.hands:
            # Premium hands - always 3-bet
            if notation in ('AA', 'KK', 'QQ', 'AKs', 'AKo'):
                return GTORecommendation(
                    primary_action=ActionRecommendation.RAISE,
                    frequency=1.0,
                    reasoning=f"{notation} is a premium 3-bet vs {villain_pos.value}"
                )
            # Bluff hands - mixed strategy
            if notation in ('A5s', 'A4s', 'A3s', 'A2s'):
                return GTORecommendation(
                    primary_action=ActionRecommendation.RAISE,
                    frequency=0.5,
                    reasoning=f"{notation} is a 3-bet bluff candidate",
                    alternative=ActionRecommendation.FOLD,
                    alt_frequency=0.5
                )
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,
                reasoning=f"{notation} is in 3-bet range vs {villain_pos.value}"
            )

        # Call range - strong hands that don't 3-bet
        call_hands = {'JJ', 'TT', '99', 'AQo', 'AJs', 'ATs', 'KQs', 'KJs', 'QJs', 'JTs'}
        if notation in call_hands and notation in open_range.hands:
            return GTORecommendation(
                primary_action=ActionRecommendation.CALL,
                reasoning=f"{notation} is a flat call vs {villain_pos.value}"
            )

        return GTORecommendation(
            primary_action=ActionRecommendation.FOLD,
            reasoning=f"{notation} is outside 3-bet/call range vs {villain_pos.value}"
        )

    def should_4bet(self, hand: HoleCards, vs_3bet_pos: Position) -> GTORecommendation:
        """
        Check if hand should 4-bet after facing a 3-bet.

        Args:
            hand: Hero's hole cards
            vs_3bet_pos: Position of 3-bettor

        Returns:
            GTORecommendation
        """
        notation = hand.notation

        # Always 4-bet premiums
        if notation in ('AA', 'KK'):
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,
                reasoning=f"{notation} is always a 4-bet for value"
            )

        # Usually 4-bet QQ, AKs
        if notation in ('QQ', 'AKs'):
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,
                frequency=0.8,
                reasoning=f"{notation} is primarily a 4-bet",
                alternative=ActionRecommendation.CALL,
                alt_frequency=0.2
            )

        # AKo - position dependent
        if notation == 'AKo':
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,
                frequency=0.6,
                reasoning=f"{notation} is mixed 4-bet/call",
                alternative=ActionRecommendation.CALL,
                alt_frequency=0.4
            )

        # Bluff 4-bets with suited wheel aces
        if notation in ('A5s', 'A4s'):
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,
                frequency=0.3,
                reasoning=f"{notation} can be used as 4-bet bluff",
                alternative=ActionRecommendation.FOLD,
                alt_frequency=0.7
            )

        # JJ, TT - usually call
        if notation in ('JJ', 'TT'):
            return GTORecommendation(
                primary_action=ActionRecommendation.CALL,
                frequency=0.7,
                reasoning=f"{notation} is usually a call facing 3-bet",
                alternative=ActionRecommendation.RAISE,
                alt_frequency=0.3
            )

        return GTORecommendation(
            primary_action=ActionRecommendation.FOLD,
            reasoning=f"{notation} is a fold vs 3-bet"
        )

    def cbet_recommendation(
        self,
        state: GameState,
        was_preflop_aggressor: bool,
        board_texture: str = "neutral"
    ) -> GTORecommendation:
        """
        Get c-bet recommendation.

        Args:
            state: Current game state
            was_preflop_aggressor: Whether hero was the preflop raiser
            board_texture: "dry", "wet", "neutral"

        Returns:
            GTORecommendation for c-bet
        """
        if not was_preflop_aggressor:
            return GTORecommendation(
                primary_action=ActionRecommendation.FOLD,
                reasoning="Not preflop aggressor - check recommended"
            )

        # GTO c-bet frequencies vary by board texture
        if board_texture == "dry":
            # High c-bet frequency on dry boards
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,  # Bet
                frequency=0.75,
                reasoning="Dry board - high c-bet frequency",
                alternative=ActionRecommendation.CALL,  # Check
                alt_frequency=0.25
            )
        elif board_texture == "wet":
            # Lower c-bet frequency on wet boards
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,
                frequency=0.45,
                reasoning="Wet board - reduced c-bet frequency",
                alternative=ActionRecommendation.CALL,
                alt_frequency=0.55
            )
        else:
            return GTORecommendation(
                primary_action=ActionRecommendation.RAISE,
                frequency=0.60,
                reasoning="Neutral board - moderate c-bet frequency",
                alternative=ActionRecommendation.CALL,
                alt_frequency=0.40
            )

    def analyze_preflop(self, state: GameState) -> GTORecommendation:
        """
        Analyze preflop situation and provide GTO recommendation.

        Args:
            state: Current game state

        Returns:
            GTORecommendation
        """
        hero = state.hero
        if hero.hole_cards is None:
            return GTORecommendation(
                primary_action=ActionRecommendation.FOLD,
                reasoning="No hole cards available"
            )

        # Check if facing action
        if state.to_call > state.table.big_blind:
            # Facing raise - check 3-bet or fold
            aggressor_pos = self._find_aggressor_position(state)
            if aggressor_pos:
                return self.should_3bet(hero.hole_cards, hero.position, aggressor_pos)

        elif state.to_call == state.table.big_blind:
            # No raise yet - consider opening
            return self.should_open(hero.hole_cards, hero.position)

        elif state.to_call == 0:
            # No bet to call (we're BB and action checked to us, or postflop)
            return GTORecommendation(
                primary_action=ActionRecommendation.CALL,  # Check
                reasoning="No bet to call - check is an option"
            )

        return GTORecommendation(
            primary_action=ActionRecommendation.FOLD,
            reasoning="Default recommendation"
        )

    def _find_aggressor_position(self, state: GameState) -> Optional[Position]:
        """Find the position of the last aggressor."""
        for action in reversed(state.action_history):
            if action.action.action_type.is_aggressive:
                return action.position
        return None


# Singleton
_advisor = GTOAdvisor()


def get_open_range(position: Position) -> HandRange:
    """Get opening range for position."""
    return _advisor.get_open_range(position)


def should_open(hand: HoleCards, position: Position) -> GTORecommendation:
    """Check if hand should open."""
    return _advisor.should_open(hand, position)


def should_3bet(hand: HoleCards, hero_pos: Position, villain_pos: Position) -> GTORecommendation:
    """Check if hand should 3-bet."""
    return _advisor.should_3bet(hand, hero_pos, villain_pos)


def should_4bet(hand: HoleCards, vs_3bet_pos: Position) -> GTORecommendation:
    """Check if hand should 4-bet."""
    return _advisor.should_4bet(hand, vs_3bet_pos)
