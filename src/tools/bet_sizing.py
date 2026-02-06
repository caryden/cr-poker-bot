"""
Bet sizing heuristics for poker.

Provides GTO-informed bet sizing recommendations based on:
- Board texture
- Hand strength category
- Stack-to-pot ratio (SPR)
- Street and position

This module helps the agent choose appropriate bet sizes.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from ..core.primitives import Street, Position
from ..core.game_state import GameState
from .board_texture import BoardTexture, BoardWetness, analyze_board


class HandStrengthCategory(Enum):
    """Categories for hand strength."""
    TRASH = auto()         # No equity, pure bluffs
    WEAK_DRAW = auto()     # Gutshot, backdoor draws
    STRONG_DRAW = auto()   # OESD, flush draw
    MARGINAL = auto()      # Weak pairs, weak showdown
    MEDIUM = auto()        # Medium pairs, decent showdown
    STRONG = auto()        # Top pair good kicker, overpair
    VERY_STRONG = auto()   # Two pair, sets, straights
    MONSTER = auto()       # Full house+


class BetPurpose(Enum):
    """Purpose of the bet."""
    VALUE = auto()         # Get called by worse
    BLUFF = auto()         # Get better to fold
    PROTECTION = auto()    # Deny equity to draws
    THIN_VALUE = auto()    # Value from marginal hands
    MERGE = auto()         # Value and protection combined


@dataclass
class BetSizing:
    """
    Recommended bet sizing.

    Attributes:
        size_pot_fraction: Bet size as fraction of pot (0.33, 0.5, 0.67, 1.0, etc.)
        size_bb: Bet size in big blinds (for preflop)
        purpose: Primary purpose of the bet
        reasoning: Explanation for the sizing
        alternative_size: Secondary sizing option
        frequency: How often to use this sizing (for mixed strategies)
    """
    size_pot_fraction: float
    size_bb: Optional[float] = None
    purpose: BetPurpose = BetPurpose.VALUE
    reasoning: str = ""
    alternative_size: Optional[float] = None
    frequency: float = 1.0

    def get_amount(self, pot: float, big_blind: float = 1.0) -> float:
        """Calculate actual bet amount."""
        if self.size_bb is not None:
            return self.size_bb * big_blind
        return pot * self.size_pot_fraction

    def __str__(self) -> str:
        if self.size_bb:
            return f"{self.size_bb:.1f}BB ({self.purpose.name}): {self.reasoning}"
        pct = int(self.size_pot_fraction * 100)
        return f"{pct}% pot ({self.purpose.name}): {self.reasoning}"


class BetSizingAdvisor:
    """Provides bet sizing recommendations."""

    # Preflop standard sizings (in BB)
    PREFLOP_OPEN = {
        Position.UTG: 2.5,
        Position.HJ: 2.5,
        Position.CO: 2.5,
        Position.BTN: 2.5,
        Position.SB: 3.0,  # Larger from SB
    }

    PREFLOP_3BET_MULTIPLIER = 3.0  # 3x the open
    PREFLOP_4BET_MULTIPLIER = 2.5  # 2.5x the 3bet

    def get_preflop_sizing(
        self,
        position: Position,
        facing_raise: bool = False,
        raise_amount: float = 0
    ) -> BetSizing:
        """
        Get preflop bet sizing.

        Args:
            position: Hero's position
            facing_raise: Whether facing a raise
            raise_amount: Amount of raise to call (in BB)

        Returns:
            BetSizing recommendation
        """
        if not facing_raise:
            # Opening raise
            size = self.PREFLOP_OPEN.get(position, 2.5)
            return BetSizing(
                size_pot_fraction=0,
                size_bb=size,
                purpose=BetPurpose.VALUE,
                reasoning=f"Standard open from {position.value}"
            )

        # 3-bet sizing
        if raise_amount <= 3.0:  # Facing open
            size = raise_amount * self.PREFLOP_3BET_MULTIPLIER
            return BetSizing(
                size_pot_fraction=0,
                size_bb=size,
                purpose=BetPurpose.VALUE,
                reasoning="Standard 3-bet sizing (3x)"
            )

        # 4-bet sizing
        size = raise_amount * self.PREFLOP_4BET_MULTIPLIER
        return BetSizing(
            size_pot_fraction=0,
            size_bb=size,
            purpose=BetPurpose.VALUE,
            reasoning="Standard 4-bet sizing (2.5x)"
        )

    def get_flop_sizing(
        self,
        texture: BoardTexture,
        hand_strength: HandStrengthCategory,
        is_aggressor: bool,
        spr: float
    ) -> BetSizing:
        """
        Get flop bet sizing.

        Args:
            texture: Board texture analysis
            hand_strength: Hero's hand strength
            is_aggressor: Whether hero was preflop aggressor
            spr: Stack-to-pot ratio

        Returns:
            BetSizing recommendation
        """
        # Low SPR - commit with strong hands
        if spr < 3:
            if hand_strength in (HandStrengthCategory.STRONG, HandStrengthCategory.VERY_STRONG, HandStrengthCategory.MONSTER):
                return BetSizing(
                    size_pot_fraction=0.75,
                    purpose=BetPurpose.VALUE,
                    reasoning="Low SPR - bet large for value and commitment"
                )
            elif hand_strength in (HandStrengthCategory.STRONG_DRAW,):
                return BetSizing(
                    size_pot_fraction=0.67,
                    purpose=BetPurpose.MERGE,
                    reasoning="Low SPR with draw - semi-bluff large"
                )

        # High card/dry boards - small bets
        if texture.is_static:
            if is_aggressor:
                # C-bet small on dry boards
                if hand_strength in (HandStrengthCategory.STRONG, HandStrengthCategory.VERY_STRONG, HandStrengthCategory.MONSTER):
                    return BetSizing(
                        size_pot_fraction=0.33,
                        purpose=BetPurpose.VALUE,
                        reasoning="Dry board - small value bet, range advantage",
                        alternative_size=0.25
                    )
                elif hand_strength == HandStrengthCategory.TRASH:
                    return BetSizing(
                        size_pot_fraction=0.33,
                        purpose=BetPurpose.BLUFF,
                        reasoning="Dry board - small c-bet bluff"
                    )
                else:
                    return BetSizing(
                        size_pot_fraction=0.33,
                        purpose=BetPurpose.PROTECTION,
                        reasoning="Dry board - small bet with medium strength"
                    )

        # Wet/connected boards - larger bets
        if texture.is_draw_heavy:
            if hand_strength in (HandStrengthCategory.VERY_STRONG, HandStrengthCategory.MONSTER):
                return BetSizing(
                    size_pot_fraction=0.75,
                    purpose=BetPurpose.VALUE,
                    reasoning="Wet board - bet large to deny equity and build pot"
                )
            elif hand_strength == HandStrengthCategory.STRONG:
                return BetSizing(
                    size_pot_fraction=0.67,
                    purpose=BetPurpose.PROTECTION,
                    reasoning="Wet board - bet for protection against draws"
                )
            elif hand_strength == HandStrengthCategory.STRONG_DRAW:
                return BetSizing(
                    size_pot_fraction=0.67,
                    purpose=BetPurpose.MERGE,
                    reasoning="Wet board - semi-bluff with draw"
                )

        # Default medium sizing
        return BetSizing(
            size_pot_fraction=0.5,
            purpose=BetPurpose.VALUE if hand_strength.value >= HandStrengthCategory.MEDIUM.value else BetPurpose.BLUFF,
            reasoning="Standard half-pot bet"
        )

    def get_turn_sizing(
        self,
        texture: BoardTexture,
        hand_strength: HandStrengthCategory,
        bet_flop: bool,
        spr: float
    ) -> BetSizing:
        """
        Get turn bet sizing.

        Args:
            texture: Board texture analysis
            hand_strength: Hero's hand strength
            bet_flop: Whether hero bet the flop
            spr: Stack-to-pot ratio

        Returns:
            BetSizing recommendation
        """
        # Low SPR - consider shoving
        if spr < 2:
            if hand_strength in (HandStrengthCategory.STRONG, HandStrengthCategory.VERY_STRONG, HandStrengthCategory.MONSTER):
                return BetSizing(
                    size_pot_fraction=1.0,
                    purpose=BetPurpose.VALUE,
                    reasoning="Very low SPR - pot or shove for max value"
                )

        # After betting flop, size up on turn
        if bet_flop:
            if hand_strength in (HandStrengthCategory.VERY_STRONG, HandStrengthCategory.MONSTER):
                return BetSizing(
                    size_pot_fraction=0.75,
                    purpose=BetPurpose.VALUE,
                    reasoning="Continuing aggression with strong hand"
                )
            elif hand_strength == HandStrengthCategory.STRONG:
                return BetSizing(
                    size_pot_fraction=0.67,
                    purpose=BetPurpose.VALUE,
                    reasoning="Second barrel for value"
                )
            elif hand_strength == HandStrengthCategory.STRONG_DRAW:
                return BetSizing(
                    size_pot_fraction=0.67,
                    purpose=BetPurpose.MERGE,
                    reasoning="Second barrel semi-bluff with draw"
                )

        # Delayed c-bet / probe
        return BetSizing(
            size_pot_fraction=0.5,
            purpose=BetPurpose.VALUE if hand_strength.value >= HandStrengthCategory.MEDIUM.value else BetPurpose.BLUFF,
            reasoning="Standard turn sizing"
        )

    def get_river_sizing(
        self,
        texture: BoardTexture,
        hand_strength: HandStrengthCategory,
        pot: float,
        stack: float
    ) -> BetSizing:
        """
        Get river bet sizing.

        Args:
            texture: Board texture analysis
            hand_strength: Hero's hand strength
            pot: Current pot size
            stack: Hero's remaining stack

        Returns:
            BetSizing recommendation
        """
        spr = stack / pot if pot > 0 else 10

        # Monster hands - go for max value
        if hand_strength == HandStrengthCategory.MONSTER:
            if spr <= 1:
                return BetSizing(
                    size_pot_fraction=1.0,
                    purpose=BetPurpose.VALUE,
                    reasoning="Nuts/near-nuts - shove for max value"
                )
            return BetSizing(
                size_pot_fraction=0.75,
                purpose=BetPurpose.VALUE,
                reasoning="Very strong hand - large value bet"
            )

        # Very strong hands
        if hand_strength == HandStrengthCategory.VERY_STRONG:
            return BetSizing(
                size_pot_fraction=0.67,
                purpose=BetPurpose.VALUE,
                reasoning="Strong value hand - standard value sizing"
            )

        # Strong hands - thin value
        if hand_strength == HandStrengthCategory.STRONG:
            return BetSizing(
                size_pot_fraction=0.5,
                purpose=BetPurpose.THIN_VALUE,
                reasoning="Thin value bet - get called by worse"
            )

        # Bluffs - polarized sizing
        if hand_strength in (HandStrengthCategory.TRASH, HandStrengthCategory.WEAK_DRAW):
            # Use large bluffs to balance with value range
            return BetSizing(
                size_pot_fraction=0.75,
                purpose=BetPurpose.BLUFF,
                reasoning="Polarized bluff - needs to look like value"
            )

        # Medium hands - consider checking
        return BetSizing(
            size_pot_fraction=0.33,
            purpose=BetPurpose.THIN_VALUE,
            reasoning="Marginal hand - small blocking bet or check"
        )

    def get_raise_sizing(
        self,
        facing_bet: float,
        pot: float,
        street: Street,
        hand_strength: HandStrengthCategory
    ) -> BetSizing:
        """
        Get raise sizing when facing a bet.

        Args:
            facing_bet: Amount of bet to call
            pot: Pot before the bet
            street: Current street
            hand_strength: Hero's hand strength

        Returns:
            BetSizing recommendation
        """
        pot_with_bet = pot + facing_bet

        # Standard raise is 3x on flop, 2.5x on turn/river
        if street == Street.FLOP:
            multiplier = 3.0
        else:
            multiplier = 2.5

        raise_to = facing_bet * multiplier

        if hand_strength == HandStrengthCategory.MONSTER:
            return BetSizing(
                size_pot_fraction=raise_to / pot_with_bet,
                purpose=BetPurpose.VALUE,
                reasoning="Value raise with monster"
            )
        elif hand_strength in (HandStrengthCategory.VERY_STRONG, HandStrengthCategory.STRONG):
            return BetSizing(
                size_pot_fraction=raise_to / pot_with_bet,
                purpose=BetPurpose.VALUE,
                reasoning="Value raise"
            )
        elif hand_strength == HandStrengthCategory.STRONG_DRAW:
            return BetSizing(
                size_pot_fraction=raise_to / pot_with_bet,
                purpose=BetPurpose.MERGE,
                reasoning="Semi-bluff raise with draw"
            )
        else:
            # Bluff raise
            return BetSizing(
                size_pot_fraction=raise_to / pot_with_bet,
                purpose=BetPurpose.BLUFF,
                reasoning="Bluff raise"
            )


# Singleton
_advisor = BetSizingAdvisor()


def get_preflop_sizing(
    position: Position,
    facing_raise: bool = False,
    raise_amount: float = 0
) -> BetSizing:
    """Get preflop bet sizing."""
    return _advisor.get_preflop_sizing(position, facing_raise, raise_amount)


def get_postflop_sizing(
    game_state: GameState,
    hand_strength: HandStrengthCategory,
    is_aggressor: bool = False,
    bet_previous_street: bool = False
) -> BetSizing:
    """
    Get postflop bet sizing based on game state.

    Args:
        game_state: Current game state
        hand_strength: Hero's hand strength category
        is_aggressor: Whether hero was preflop aggressor
        bet_previous_street: Whether hero bet the previous street

    Returns:
        BetSizing recommendation
    """
    texture = analyze_board(game_state.board)
    pot = game_state.pot.total
    stack = game_state.hero.stack
    spr = stack / pot if pot > 0 else 10

    if game_state.street == Street.FLOP:
        return _advisor.get_flop_sizing(texture, hand_strength, is_aggressor, spr)
    elif game_state.street == Street.TURN:
        return _advisor.get_turn_sizing(texture, hand_strength, bet_previous_street, spr)
    elif game_state.street == Street.RIVER:
        return _advisor.get_river_sizing(texture, hand_strength, pot, stack)
    else:
        # Preflop fallback
        return get_preflop_sizing(game_state.hero.position)


def get_value_sizing(pot: float, street: Street) -> float:
    """Get standard value bet sizing for a street."""
    if street == Street.FLOP:
        return pot * 0.33  # Small on flop
    elif street == Street.TURN:
        return pot * 0.67
    else:
        return pot * 0.75


def get_bluff_sizing(pot: float, street: Street) -> float:
    """Get standard bluff sizing for a street."""
    # Bluffs should match value sizing for balance
    return get_value_sizing(pot, street)
