"""
PHH (Poker Hand History) format utilities.

Implements the PHH specification for recording and replaying poker hands.
See: https://phh.readthedocs.io/

PHH is TOML-based with standardized fields for game state and actions.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from src.core.primitives import Action, ActionType, Position, HoleCards, Card
from src.core import Street


# PHH Action notation
# d dh p1 AcKd  - deal hole cards to player 1
# d db Jc3d5c  - deal board cards
# p1 f         - player 1 folds
# p1 cc        - player 1 checks/calls
# p1 cbr 100   - player 1 bets/raises to 100
# p1 sm        - player 1 shows mucked cards


@dataclass
class PHHHand:
    """A single hand in PHH format."""

    # Required fields
    variant: str = "NT"  # NT = No-Limit Texas Hold'em
    antes: list[int] = field(default_factory=list)
    blinds_or_straddles: list[int] = field(default_factory=list)
    min_bet: int = 0
    starting_stacks: list[int] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    # Optional fields
    players: list[str] = field(default_factory=list)
    author: str = ""
    event: str = ""
    year: int = 0
    month: int = 0
    day: int = 0
    hand_num: int = 0

    # Internal tracking (not written to PHH)
    _player_cards: dict = field(default_factory=dict)
    _board_cards: list = field(default_factory=list)

    def add_hole_cards(self, player_idx: int, cards: HoleCards, hidden: bool = False):
        """Record hole cards dealt to a player."""
        if hidden:
            card_str = "????"
        else:
            card_str = f"{cards.card1}{cards.card2}"
        self.actions.append(f"d dh p{player_idx + 1} {card_str}")
        self._player_cards[player_idx] = cards

    def add_board(self, cards: list[Card], street: Street):
        """Record board cards dealt."""
        card_str = "".join(str(c) for c in cards)
        if street == Street.FLOP:
            self.actions.append(f"d db {card_str}")
        else:
            # Turn and river are single cards
            self.actions.append(f"d db {card_str}")
        self._board_cards.extend(cards)

    def add_action(self, player_idx: int, action: Action, player_name: str = ""):
        """Record a player action."""
        p = f"p{player_idx + 1}"

        if action.action_type == ActionType.FOLD:
            self.actions.append(f"{p} f")
        elif action.action_type == ActionType.CHECK:
            self.actions.append(f"{p} cc")
        elif action.action_type == ActionType.CALL:
            self.actions.append(f"{p} cc")
        elif action.action_type in (ActionType.BET, ActionType.RAISE):
            self.actions.append(f"{p} cbr {int(action.amount)}")
        elif action.action_type == ActionType.ALL_IN:
            self.actions.append(f"{p} cbr {int(action.amount)}")

    def add_showdown(self, player_idx: int, cards: HoleCards):
        """Record a player showing cards at showdown."""
        card_str = f"{cards.card1.notation}{cards.card2.notation}"
        self.actions.append(f"p{player_idx + 1} sm {card_str}")

    def to_phh(self) -> str:
        """Convert to PHH format string."""
        lines = []

        # Header comment
        if self.event:
            lines.append(f'# {self.event}')
        if self.hand_num:
            lines.append(f'# Hand #{self.hand_num}')
        lines.append('')

        # Required fields
        lines.append(f'variant = "{self.variant}"')

        if self.antes:
            lines.append(f'antes = {self.antes}')

        if self.blinds_or_straddles:
            lines.append(f'blinds_or_straddles = {self.blinds_or_straddles}')

        if self.min_bet:
            lines.append(f'min_bet = {self.min_bet}')

        lines.append(f'starting_stacks = {self.starting_stacks}')

        # Actions
        actions_str = ',\n    '.join(f'"{a}"' for a in self.actions)
        lines.append(f'actions = [\n    {actions_str},\n]')

        # Optional fields
        if self.players:
            players_str = ', '.join(f'"{p}"' for p in self.players)
            lines.append(f'players = [{players_str}]')

        if self.author:
            lines.append(f'author = "{self.author}"')

        if self.year:
            lines.append(f'year = {self.year}')
            if self.month:
                lines.append(f'month = {self.month}')
            if self.day:
                lines.append(f'day = {self.day}')

        return '\n'.join(lines)

    def save(self, filepath: str):
        """Save to a .phh file."""
        with open(filepath, 'w') as f:
            f.write(self.to_phh())


@dataclass
class PHHSession:
    """A collection of hands (e.g., a tournament or session)."""

    hands: list[PHHHand] = field(default_factory=list)
    event: str = ""
    author: str = "cr-poker-bot"

    def add_hand(self, hand: PHHHand):
        """Add a hand to the session."""
        hand.hand_num = len(self.hands) + 1
        hand.event = self.event
        hand.author = self.author
        self.hands.append(hand)

    def save_all(self, directory: str):
        """Save all hands to individual .phh files."""
        os.makedirs(directory, exist_ok=True)
        for i, hand in enumerate(self.hands):
            filepath = os.path.join(directory, f"hand_{i+1:04d}.phh")
            hand.save(filepath)

    def to_summary(self) -> str:
        """Get a summary of the session."""
        lines = [
            f"PHH Session: {self.event}",
            f"Hands: {len(self.hands)}",
            f"Author: {self.author}",
        ]
        return '\n'.join(lines)


def parse_phh_action(action_str: str) -> tuple[str, str, Optional[int]]:
    """
    Parse a PHH action string.

    Returns: (actor, action_type, amount)
    - actor: 'd' for dealer, 'p1', 'p2', etc for players
    - action_type: 'dh' (deal hole), 'db' (deal board), 'f', 'cc', 'cbr', 'sm'
    - amount: bet/raise amount if applicable
    """
    parts = action_str.strip().split()
    actor = parts[0]
    action_type = parts[1] if len(parts) > 1 else ""

    # Extract amount if present
    amount = None
    if action_type == 'cbr' and len(parts) > 2:
        try:
            amount = int(parts[2])
        except ValueError:
            pass

    return actor, action_type, amount


def load_phh(filepath: str) -> PHHHand:
    """
    Load a PHH file.

    Note: This is a simplified parser. For full compliance,
    use the official pokerkit PHH parser.
    """
    import re

    hand = PHHHand()

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse variant
    match = re.search(r'variant\s*=\s*"([^"]+)"', content)
    if match:
        hand.variant = match.group(1)

    # Parse starting_stacks
    match = re.search(r'starting_stacks\s*=\s*\[([^\]]+)\]', content)
    if match:
        hand.starting_stacks = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]

    # Parse blinds
    match = re.search(r'blinds_or_straddles\s*=\s*\[([^\]]+)\]', content)
    if match:
        hand.blinds_or_straddles = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]

    # Parse actions
    match = re.search(r'actions\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if match:
        actions_str = match.group(1)
        hand.actions = re.findall(r'"([^"]+)"', actions_str)

    # Parse players
    match = re.search(r'players\s*=\s*\[([^\]]+)\]', content)
    if match:
        hand.players = re.findall(r'"([^"]+)"', match.group(1))

    return hand


# Position to player index mapping for 6-max
POSITION_TO_IDX_6MAX = {
    Position.SB: 0,
    Position.BB: 1,
    Position.UTG: 2,
    Position.HJ: 3,
    Position.CO: 4,
    Position.BTN: 5,
}


def create_phh_from_game(
    players: list[str],
    starting_stacks: list[int],
    blinds: tuple[int, int],
    min_bet: int,
    hole_cards: dict[int, HoleCards],
    board_cards: list[Card],
    actions: list[tuple[int, Action]],  # (player_idx, action)
    event: str = "",
    hero_idx: int = 0,
) -> PHHHand:
    """
    Create a PHH hand from game data.

    Args:
        players: List of player names in seat order
        starting_stacks: List of starting stacks
        blinds: (small_blind, big_blind)
        min_bet: Minimum bet size
        hole_cards: Dict of player_idx -> HoleCards
        board_cards: List of board cards in order
        actions: List of (player_idx, Action) tuples in order
        event: Event name
        hero_idx: Index of hero player (their cards shown, others hidden)
    """
    hand = PHHHand(
        variant="NT",
        blinds_or_straddles=[blinds[0], blinds[1]] + [0] * (len(players) - 2),
        min_bet=min_bet,
        starting_stacks=starting_stacks,
        players=players,
        event=event,
    )

    now = datetime.now()
    hand.year = now.year
    hand.month = now.month
    hand.day = now.day

    # Deal hole cards (hero's shown, villains hidden)
    for idx in range(len(players)):
        if idx in hole_cards:
            hand.add_hole_cards(idx, hole_cards[idx], hidden=(idx != hero_idx))

    # Track current street for board dealing
    current_street = Street.PREFLOP
    board_dealt = 0

    for player_idx, action in actions:
        # Check if we need to deal board cards before this action
        # (In a real implementation, we'd track streets more carefully)
        hand.add_action(player_idx, action, players[player_idx] if player_idx < len(players) else "")

    # Deal remaining board cards
    if board_cards:
        if len(board_cards) >= 3:
            hand.add_board(board_cards[:3], Street.FLOP)
        if len(board_cards) >= 4:
            hand.add_board([board_cards[3]], Street.TURN)
        if len(board_cards) >= 5:
            hand.add_board([board_cards[4]], Street.RIVER)

    return hand
