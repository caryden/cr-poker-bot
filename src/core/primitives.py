"""
Core primitive types for poker game representation.

Defines the fundamental building blocks: cards, positions, streets, and actions.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Suit(Enum):
    """Card suits."""
    CLUBS = 'c'
    DIAMONDS = 'd'
    HEARTS = 'h'
    SPADES = 's'

    def __str__(self) -> str:
        return self.value


class Rank(Enum):
    """Card ranks (2-A)."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        if self.value <= 10:
            return str(self.value) if self.value != 10 else 'T'
        return {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}[self.value]

    @classmethod
    def from_char(cls, c: str) -> 'Rank':
        """Parse rank from character (2-9, T, J, Q, K, A)."""
        mapping = {
            '2': cls.TWO, '3': cls.THREE, '4': cls.FOUR, '5': cls.FIVE,
            '6': cls.SIX, '7': cls.SEVEN, '8': cls.EIGHT, '9': cls.NINE,
            'T': cls.TEN, 't': cls.TEN, '10': cls.TEN,
            'J': cls.JACK, 'j': cls.JACK,
            'Q': cls.QUEEN, 'q': cls.QUEEN,
            'K': cls.KING, 'k': cls.KING,
            'A': cls.ACE, 'a': cls.ACE
        }
        if c not in mapping:
            raise ValueError(f"Invalid rank character: {c}")
        return mapping[c]


@dataclass(frozen=True)
class Card:
    """
    A single playing card.

    Immutable and hashable for use in sets and as dict keys.
    """
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self})"

    @classmethod
    def from_str(cls, s: str) -> 'Card':
        """
        Parse card from string like 'As', 'Th', '2c'.

        Args:
            s: Card string (rank + suit)

        Returns:
            Card instance
        """
        s = s.strip()
        if len(s) < 2:
            raise ValueError(f"Invalid card string: {s}")

        rank_char = s[:-1]
        suit_char = s[-1].lower()

        rank = Rank.from_char(rank_char)
        suit_map = {'c': Suit.CLUBS, 'd': Suit.DIAMONDS,
                    'h': Suit.HEARTS, 's': Suit.SPADES}

        if suit_char not in suit_map:
            raise ValueError(f"Invalid suit character: {suit_char}")

        return cls(rank, suit_map[suit_char])


@dataclass(frozen=True)
class HoleCards:
    """
    A player's two hole cards.

    Cards are stored in canonical order (higher rank first, then by suit).
    """
    card1: Card
    card2: Card

    def __post_init__(self):
        # Ensure canonical ordering
        if (self.card1.rank.value, self.card1.suit.value) < \
           (self.card2.rank.value, self.card2.suit.value):
            object.__setattr__(self, 'card1', self.card2)
            object.__setattr__(self, 'card2', self.card1)

    def __str__(self) -> str:
        return f"{self.card1}{self.card2}"

    def __iter__(self):
        yield self.card1
        yield self.card2

    @property
    def is_pair(self) -> bool:
        """True if hole cards are a pocket pair."""
        return self.card1.rank == self.card2.rank

    @property
    def is_suited(self) -> bool:
        """True if hole cards are suited."""
        return self.card1.suit == self.card2.suit

    @property
    def gap(self) -> int:
        """Gap between ranks (0 for connectors, 1 for one-gappers, etc.)."""
        return abs(self.card1.rank.value - self.card2.rank.value) - 1

    @property
    def notation(self) -> str:
        """
        Standard hand notation (e.g., 'AKs', 'QQ', 'T9o').
        """
        r1, r2 = str(self.card1.rank), str(self.card2.rank)
        if self.is_pair:
            return f"{r1}{r2}"
        suffix = 's' if self.is_suited else 'o'
        return f"{r1}{r2}{suffix}"

    @classmethod
    def from_str(cls, s: str) -> 'HoleCards':
        """
        Parse hole cards from string like 'AsKh' or 'As Kh'.
        """
        s = s.replace(' ', '')
        if len(s) == 4:
            return cls(Card.from_str(s[:2]), Card.from_str(s[2:]))
        raise ValueError(f"Invalid hole cards string: {s}")


class Street(Enum):
    """Betting rounds in Texas Hold'em."""
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()

    def __str__(self) -> str:
        return self.name.lower()


class Position(Enum):
    """
    Table positions for 6-max poker.

    Positions are listed in order of action preflop.
    BTN acts last postflop.
    """
    UTG = 'UTG'      # Under the Gun (first to act preflop)
    HJ = 'HJ'        # Hijack
    CO = 'CO'        # Cutoff
    BTN = 'BTN'      # Button (dealer)
    SB = 'SB'        # Small Blind
    BB = 'BB'        # Big Blind

    def __str__(self) -> str:
        return self.value

    @property
    def preflop_order(self) -> int:
        """Order of action preflop (0 = first)."""
        order = {
            Position.UTG: 0,
            Position.HJ: 1,
            Position.CO: 2,
            Position.BTN: 3,
            Position.SB: 4,
            Position.BB: 5,
        }
        return order[self]

    @property
    def postflop_order(self) -> int:
        """Order of action postflop (0 = first). SB acts first if still in."""
        order = {
            Position.SB: 0,
            Position.BB: 1,
            Position.UTG: 2,
            Position.HJ: 3,
            Position.CO: 4,
            Position.BTN: 5,
        }
        return order[self]

    @classmethod
    def all_positions(cls) -> list['Position']:
        """Return all positions in preflop action order."""
        return [cls.UTG, cls.HJ, cls.CO, cls.BTN, cls.SB, cls.BB]


class ActionType(Enum):
    """Types of actions a player can take."""
    FOLD = 'fold'
    CHECK = 'check'
    CALL = 'call'
    BET = 'bet'
    RAISE = 'raise'
    ALL_IN = 'all_in'

    def __str__(self) -> str:
        return self.value

    @property
    def is_aggressive(self) -> bool:
        """True if this is an aggressive action (bet/raise/all-in)."""
        return self in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)

    @property
    def is_passive(self) -> bool:
        """True if this is a passive action (check/call)."""
        return self in (ActionType.CHECK, ActionType.CALL)


@dataclass(frozen=True)
class Action:
    """
    A player action in the game.

    Attributes:
        action_type: The type of action taken
        amount: Bet/raise amount (0 for fold/check, call amount for call)
        is_all_in: Whether this action puts the player all-in
    """
    action_type: ActionType
    amount: float = 0.0
    is_all_in: bool = False

    def __str__(self) -> str:
        if self.action_type in (ActionType.FOLD, ActionType.CHECK):
            return str(self.action_type)
        if self.action_type == ActionType.ALL_IN:
            return f"all-in {self.amount:.0f}"
        return f"{self.action_type} {self.amount:.0f}"

    @classmethod
    def fold(cls) -> 'Action':
        return cls(ActionType.FOLD)

    @classmethod
    def check(cls) -> 'Action':
        return cls(ActionType.CHECK)

    @classmethod
    def call(cls, amount: float) -> 'Action':
        return cls(ActionType.CALL, amount)

    @classmethod
    def bet(cls, amount: float, is_all_in: bool = False) -> 'Action':
        return cls(ActionType.BET, amount, is_all_in)

    @classmethod
    def raise_to(cls, amount: float, is_all_in: bool = False) -> 'Action':
        return cls(ActionType.RAISE, amount, is_all_in)

    @classmethod
    def all_in(cls, amount: float) -> 'Action':
        return cls(ActionType.ALL_IN, amount, is_all_in=True)


@dataclass
class Pot:
    """
    Represents the pot(s) in a hand.

    Handles main pot and side pots for all-in situations.
    """
    main: float = 0.0
    side_pots: list[tuple[float, set[str]]] = None  # (amount, eligible_players)

    def __post_init__(self):
        if self.side_pots is None:
            self.side_pots = []

    @property
    def total(self) -> float:
        """Total of all pots."""
        return self.main + sum(sp[0] for sp in self.side_pots)

    def add(self, amount: float) -> None:
        """Add chips to the main pot."""
        self.main += amount


@dataclass
class Board:
    """
    Community cards on the board.
    """
    cards: list[Card] = None

    def __post_init__(self):
        if self.cards is None:
            self.cards = []

    def __str__(self) -> str:
        return ' '.join(str(c) for c in self.cards) if self.cards else '(empty)'

    def __len__(self) -> int:
        return len(self.cards)

    @property
    def street(self) -> Street:
        """Infer current street from board cards."""
        n = len(self.cards)
        if n == 0:
            return Street.PREFLOP
        elif n == 3:
            return Street.FLOP
        elif n == 4:
            return Street.TURN
        elif n == 5:
            return Street.RIVER
        raise ValueError(f"Invalid board size: {n}")

    @property
    def flop(self) -> Optional[tuple[Card, Card, Card]]:
        """Return flop cards if dealt."""
        if len(self.cards) >= 3:
            return (self.cards[0], self.cards[1], self.cards[2])
        return None

    @property
    def turn(self) -> Optional[Card]:
        """Return turn card if dealt."""
        if len(self.cards) >= 4:
            return self.cards[3]
        return None

    @property
    def river(self) -> Optional[Card]:
        """Return river card if dealt."""
        if len(self.cards) >= 5:
            return self.cards[4]
        return None

    def deal_flop(self, c1: Card, c2: Card, c3: Card) -> None:
        """Deal the flop."""
        if len(self.cards) != 0:
            raise ValueError("Flop already dealt")
        self.cards = [c1, c2, c3]

    def deal_turn(self, card: Card) -> None:
        """Deal the turn."""
        if len(self.cards) != 3:
            raise ValueError("Cannot deal turn - need flop first")
        self.cards.append(card)

    def deal_river(self, card: Card) -> None:
        """Deal the river."""
        if len(self.cards) != 4:
            raise ValueError("Cannot deal river - need turn first")
        self.cards.append(card)

    @classmethod
    def from_str(cls, s: str) -> 'Board':
        """Parse board from string like 'As Kh Qd' or 'AsKhQd'."""
        s = s.strip()
        if not s:
            return cls()

        # Handle space-separated or concatenated
        if ' ' in s:
            cards = [Card.from_str(c) for c in s.split()]
        else:
            # Assume 2 chars per card
            cards = [Card.from_str(s[i:i+2]) for i in range(0, len(s), 2)]

        return cls(cards)
