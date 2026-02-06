"""
Game state representation for 6-max No-Limit Texas Hold'em.

Tracks the complete state of a poker hand including player stacks,
actions, pot, and board.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

from .primitives import (
    Card, HoleCards, Board, Pot, Action, ActionType,
    Street, Position
)


class PlayerStatus(Enum):
    """Status of a player in the current hand."""
    ACTIVE = auto()      # Still in the hand, can act
    FOLDED = auto()      # Folded this hand
    ALL_IN = auto()      # All-in, waiting for showdown
    SITTING_OUT = auto() # Not playing this hand


@dataclass
class PlayerState:
    """
    State of a single player at the table.

    Attributes:
        player_id: Unique identifier for the player
        position: Current position at the table
        stack: Current chip stack
        hole_cards: Player's hole cards (None if unknown/mucked)
        status: Current status in the hand
        bet_this_street: Amount bet on current street
        total_invested: Total chips invested in this hand
    """
    player_id: str
    position: Position
    stack: float
    hole_cards: Optional[HoleCards] = None
    status: PlayerStatus = PlayerStatus.ACTIVE
    bet_this_street: float = 0.0
    total_invested: float = 0.0

    @property
    def is_active(self) -> bool:
        """Can this player still act?"""
        return self.status == PlayerStatus.ACTIVE

    @property
    def is_in_hand(self) -> bool:
        """Is player still contesting the pot?"""
        return self.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)

    def reset_street_bet(self) -> None:
        """Reset bet amount for new street."""
        self.bet_this_street = 0.0


@dataclass
class ActionRecord:
    """Record of an action taken during the hand."""
    player_id: str
    position: Position
    street: Street
    action: Action
    pot_before: float
    pot_after: float


@dataclass
class TableState:
    """
    State of the table independent of any specific hand.

    Attributes:
        table_id: Unique table identifier
        max_players: Maximum players (6 for 6-max)
        small_blind: Small blind amount
        big_blind: Big blind amount
        ante: Ante amount (0 if no ante)
        btn_seat: Current button seat (0-5)
    """
    table_id: str
    max_players: int = 6
    small_blind: float = 0.5
    big_blind: float = 1.0
    ante: float = 0.0
    btn_seat: int = 0

    def rotate_button(self) -> None:
        """Move button to next seat."""
        self.btn_seat = (self.btn_seat + 1) % self.max_players


@dataclass
class GameState:
    """
    Complete state of a poker hand.

    This is the primary state object passed to the agent for decision-making.
    It contains all information needed to understand the current situation.

    Attributes:
        hand_id: Unique identifier for this hand
        table: Table configuration
        players: Dict mapping player_id to PlayerState
        hero_id: The player_id of our agent
        board: Community cards
        pot: Current pot(s)
        street: Current betting round
        current_bet: Current bet to call
        min_raise: Minimum raise amount
        action_history: List of all actions taken
        acting_player: Player who must act next (None if hand over)
    """
    hand_id: str
    table: TableState
    players: dict[str, PlayerState]
    hero_id: str
    board: Board = field(default_factory=Board)
    pot: Pot = field(default_factory=Pot)
    street: Street = Street.PREFLOP
    current_bet: float = 0.0
    min_raise: float = 0.0
    action_history: list[ActionRecord] = field(default_factory=list)
    acting_player: Optional[str] = None

    def __post_init__(self):
        if self.min_raise == 0.0:
            self.min_raise = self.table.big_blind

    @property
    def hero(self) -> PlayerState:
        """Get hero's state."""
        return self.players[self.hero_id]

    @property
    def villains(self) -> list[PlayerState]:
        """Get all villain states."""
        return [p for pid, p in self.players.items() if pid != self.hero_id]

    @property
    def active_players(self) -> list[PlayerState]:
        """Players who can still act."""
        return [p for p in self.players.values() if p.is_active]

    @property
    def players_in_hand(self) -> list[PlayerState]:
        """Players still contesting the pot."""
        return [p for p in self.players.values() if p.is_in_hand]

    @property
    def num_active(self) -> int:
        """Number of players who can still act."""
        return len(self.active_players)

    @property
    def num_in_hand(self) -> int:
        """Number of players still in the hand."""
        return len(self.players_in_hand)

    @property
    def is_heads_up(self) -> bool:
        """True if only 2 players remain."""
        return self.num_in_hand == 2

    @property
    def to_call(self) -> float:
        """Amount hero needs to call."""
        return max(0, self.current_bet - self.hero.bet_this_street)

    @property
    def pot_odds(self) -> float:
        """Current pot odds as a ratio (call / (pot + call))."""
        call_amount = self.to_call
        if call_amount == 0:
            return 0.0
        return call_amount / (self.pot.total + call_amount)

    @property
    def effective_stack(self) -> float:
        """Effective stack (smaller of hero stack and smallest villain stack)."""
        villain_stacks = [p.stack for p in self.villains if p.is_in_hand]
        if not villain_stacks:
            return self.hero.stack
        return min(self.hero.stack, min(villain_stacks))

    @property
    def spr(self) -> float:
        """Stack-to-Pot Ratio."""
        if self.pot.total == 0:
            return float('inf')
        return self.effective_stack / self.pot.total

    def get_player_by_position(self, position: Position) -> Optional[PlayerState]:
        """Get player at a specific position."""
        for player in self.players.values():
            if player.position == position:
                return player
        return None

    def get_legal_actions(self) -> list[Action]:
        """
        Get list of legal actions for the acting player.

        Returns actions with appropriate min/max amounts.
        """
        if self.acting_player is None:
            return []

        player = self.players[self.acting_player]
        if not player.is_active:
            return []

        actions = []
        to_call = self.current_bet - player.bet_this_street

        # Can always fold if facing a bet
        if to_call > 0:
            actions.append(Action.fold())

        # Check if no bet to call
        if to_call <= 0:
            actions.append(Action.check())

        # Call if facing a bet and have chips
        if to_call > 0 and player.stack > 0:
            call_amount = min(to_call, player.stack)
            actions.append(Action.call(call_amount))

        # Bet if no current bet and have chips
        if self.current_bet == 0 and player.stack > 0:
            min_bet = self.table.big_blind
            if player.stack <= min_bet:
                actions.append(Action.all_in(player.stack))
            else:
                actions.append(Action.bet(min_bet))
                if player.stack > min_bet:
                    actions.append(Action.all_in(player.stack))

        # Raise if facing a bet and have chips beyond call
        if self.current_bet > 0 and player.stack > to_call:
            min_raise_to = self.current_bet + self.min_raise
            remaining = player.stack - to_call

            if remaining <= self.min_raise:
                # Can only go all-in
                actions.append(Action.all_in(player.stack))
            else:
                actions.append(Action.raise_to(min_raise_to))
                actions.append(Action.all_in(player.stack))

        return actions

    def is_hero_turn(self) -> bool:
        """Is it hero's turn to act?"""
        return self.acting_player == self.hero_id

    def street_actions(self, street: Optional[Street] = None) -> list[ActionRecord]:
        """Get actions for a specific street (default: current street)."""
        target = street or self.street
        return [a for a in self.action_history if a.street == target]

    def player_actions(self, player_id: str) -> list[ActionRecord]:
        """Get all actions by a specific player."""
        return [a for a in self.action_history if a.player_id == player_id]

    def last_aggressor(self) -> Optional[str]:
        """Get player_id of last aggressor on current street."""
        for action in reversed(self.street_actions()):
            if action.action.action_type.is_aggressive:
                return action.player_id
        return None

    def to_dict(self) -> dict:
        """Serialize state to dictionary for LLM context."""
        return {
            'hand_id': self.hand_id,
            'street': str(self.street),
            'board': str(self.board),
            'pot': self.pot.total,
            'current_bet': self.current_bet,
            'to_call': self.to_call,
            'pot_odds': f"{self.pot_odds:.1%}",
            'spr': f"{self.spr:.1f}",
            'hero': {
                'position': str(self.hero.position),
                'stack': self.hero.stack,
                'cards': str(self.hero.hole_cards) if self.hero.hole_cards else '??',
                'invested': self.hero.total_invested,
            },
            'villains': [
                {
                    'id': v.player_id,
                    'position': str(v.position),
                    'stack': v.stack,
                    'status': v.status.name,
                    'invested': v.total_invested,
                }
                for v in self.villains if v.is_in_hand
            ],
            'action_history': [
                {
                    'player': a.player_id,
                    'position': str(a.position),
                    'street': str(a.street),
                    'action': str(a.action),
                }
                for a in self.action_history
            ]
        }


def create_6max_game(
    hand_id: str,
    hero_id: str,
    hero_position: Position,
    hero_cards: HoleCards,
    stacks: dict[Position, float],
    small_blind: float = 0.5,
    big_blind: float = 1.0,
) -> GameState:
    """
    Factory function to create a fresh 6-max game state.

    Args:
        hand_id: Unique hand identifier
        hero_id: Hero's player ID
        hero_position: Hero's position
        hero_cards: Hero's hole cards
        stacks: Stack sizes by position
        small_blind: Small blind amount
        big_blind: Big blind amount

    Returns:
        Initialized GameState ready for preflop action
    """
    table = TableState(
        table_id=f"table_{hand_id}",
        small_blind=small_blind,
        big_blind=big_blind,
    )

    # Create players for each position with stack
    players = {}
    for i, position in enumerate(Position.all_positions()):
        if position not in stacks:
            continue

        player_id = hero_id if position == hero_position else f"villain_{position.value}"
        hole_cards = hero_cards if position == hero_position else None

        player = PlayerState(
            player_id=player_id,
            position=position,
            stack=stacks[position],
            hole_cards=hole_cards,
        )
        players[player_id] = player

    # Post blinds
    sb_player = None
    bb_player = None
    for p in players.values():
        if p.position == Position.SB:
            sb_player = p
        elif p.position == Position.BB:
            bb_player = p

    pot = Pot()

    if sb_player:
        sb_amount = min(small_blind, sb_player.stack)
        sb_player.stack -= sb_amount
        sb_player.bet_this_street = sb_amount
        sb_player.total_invested = sb_amount
        pot.add(sb_amount)

    if bb_player:
        bb_amount = min(big_blind, bb_player.stack)
        bb_player.stack -= bb_amount
        bb_player.bet_this_street = bb_amount
        bb_player.total_invested = bb_amount
        pot.add(bb_amount)

    # Find first to act (UTG preflop)
    acting_player = None
    for position in Position.all_positions():
        player = next((p for p in players.values() if p.position == position), None)
        if player and player.is_active:
            acting_player = player.player_id
            break

    return GameState(
        hand_id=hand_id,
        table=table,
        players=players,
        hero_id=hero_id,
        pot=pot,
        street=Street.PREFLOP,
        current_bet=big_blind,
        min_raise=big_blind,
        acting_player=acting_player,
    )
