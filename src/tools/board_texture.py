"""
Board texture analysis for poker decision making.

Provides classification of board textures to inform betting strategies:
- Dry vs wet boards
- Connectedness
- Flush potential
- Paired boards
- High card strength

These classifications help determine optimal bet sizing and frequencies.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from ..core.primitives import Card, Board, Rank, Suit


class BoardWetness(Enum):
    """How draw-heavy the board is."""
    VERY_DRY = auto()      # K72 rainbow
    DRY = auto()           # AQ4 rainbow
    NEUTRAL = auto()       # JT5 two-tone
    WET = auto()           # 987 two-tone
    VERY_WET = auto()      # 876 monotone


class BoardPairedness(Enum):
    """Paired board status."""
    UNPAIRED = auto()
    PAIRED = auto()        # One pair on board
    TRIPS = auto()         # Three of a kind on board
    TWO_PAIR = auto()      # Two pair on board
    QUADS = auto()         # Four of a kind on board


class BoardHighness(Enum):
    """How high the board cards are."""
    LOW = auto()           # Highest card <= 8
    MEDIUM = auto()        # Highest card 9-J
    HIGH = auto()          # Highest card Q-A


@dataclass
class BoardTexture:
    """
    Complete board texture analysis.

    Attributes:
        wetness: How draw-heavy the board is
        pairedness: Whether board is paired
        highness: High card category
        flush_possible: Whether flush is possible
        flush_draw_possible: Whether flush draw exists
        straight_possible: Whether straight is possible
        straight_draw_possible: Whether OESD/gutshot exists
        monotone: All same suit
        rainbow: All different suits
        connected: Cards form sequences
        high_card: Highest card on board
        num_cards: Number of community cards
    """
    wetness: BoardWetness
    pairedness: BoardPairedness
    highness: BoardHighness
    flush_possible: bool
    flush_draw_possible: bool
    straight_possible: bool
    straight_draw_possible: bool
    monotone: bool
    rainbow: bool
    connected: bool
    high_card: Optional[Rank]
    num_cards: int

    def __str__(self) -> str:
        parts = [self.wetness.name.replace('_', ' ').title()]

        if self.pairedness != BoardPairedness.UNPAIRED:
            parts.append(self.pairedness.name.title())

        if self.monotone:
            parts.append("Monotone")
        elif self.rainbow:
            parts.append("Rainbow")
        else:
            parts.append("Two-Tone")

        if self.connected:
            parts.append("Connected")

        return ", ".join(parts)

    @property
    def is_draw_heavy(self) -> bool:
        """Check if board has significant draw potential."""
        return self.wetness in (BoardWetness.WET, BoardWetness.VERY_WET)

    @property
    def is_static(self) -> bool:
        """Check if board is unlikely to change hand strengths on later streets."""
        return self.wetness in (BoardWetness.VERY_DRY, BoardWetness.DRY)

    @property
    def favors_aggressor(self) -> bool:
        """Check if board texture favors the preflop aggressor's range."""
        # High boards and dry boards favor the raiser
        return self.highness == BoardHighness.HIGH or self.is_static


class BoardAnalyzer:
    """Analyzes poker board textures."""

    def analyze(self, board: Board) -> BoardTexture:
        """
        Perform complete board texture analysis.

        Args:
            board: The community cards

        Returns:
            BoardTexture with all classifications
        """
        cards = board.cards
        if not cards:
            return self._empty_texture()

        ranks = [c.rank for c in cards]
        suits = [c.suit for c in cards]
        rank_values = sorted([r.value for r in ranks], reverse=True)

        # Suit analysis
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suit_count = max(suit_counts.values())

        monotone = max_suit_count == len(cards) and len(cards) >= 3
        rainbow = len(suit_counts) == len(cards)
        flush_possible = max_suit_count >= 3 and len(cards) >= 3
        flush_draw_possible = max_suit_count >= 2 and len(cards) < 5

        # Rank analysis
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        max_rank_count = max(rank_counts.values())

        # Pairedness
        if max_rank_count == 4:
            pairedness = BoardPairedness.QUADS
        elif max_rank_count == 3:
            pairedness = BoardPairedness.TRIPS
        elif max_rank_count == 2:
            pair_count = sum(1 for c in rank_counts.values() if c == 2)
            pairedness = BoardPairedness.TWO_PAIR if pair_count >= 2 else BoardPairedness.PAIRED
        else:
            pairedness = BoardPairedness.UNPAIRED

        # Highness
        high_card = max(ranks, key=lambda r: r.value)
        if high_card.value >= 12:  # Queen+
            highness = BoardHighness.HIGH
        elif high_card.value >= 9:  # 9-J
            highness = BoardHighness.MEDIUM
        else:
            highness = BoardHighness.LOW

        # Connectedness
        connected, straight_possible, straight_draw_possible = self._analyze_connectivity(rank_values)

        # Overall wetness
        wetness = self._calculate_wetness(
            flush_possible, flush_draw_possible,
            straight_possible, straight_draw_possible,
            connected, monotone, pairedness
        )

        return BoardTexture(
            wetness=wetness,
            pairedness=pairedness,
            highness=highness,
            flush_possible=flush_possible,
            flush_draw_possible=flush_draw_possible,
            straight_possible=straight_possible,
            straight_draw_possible=straight_draw_possible,
            monotone=monotone,
            rainbow=rainbow,
            connected=connected,
            high_card=high_card,
            num_cards=len(cards)
        )

    def _empty_texture(self) -> BoardTexture:
        """Return texture for empty board."""
        return BoardTexture(
            wetness=BoardWetness.NEUTRAL,
            pairedness=BoardPairedness.UNPAIRED,
            highness=BoardHighness.MEDIUM,
            flush_possible=False,
            flush_draw_possible=False,
            straight_possible=False,
            straight_draw_possible=False,
            monotone=False,
            rainbow=True,
            connected=False,
            high_card=None,
            num_cards=0
        )

    def _analyze_connectivity(self, rank_values: list[int]) -> tuple[bool, bool, bool]:
        """
        Analyze how connected the board is.

        Returns:
            (connected, straight_possible, straight_draw_possible)
        """
        if len(rank_values) < 2:
            return False, False, False

        unique_ranks = sorted(set(rank_values), reverse=True)

        # Check for gaps between consecutive cards
        gaps = []
        for i in range(len(unique_ranks) - 1):
            gap = unique_ranks[i] - unique_ranks[i+1] - 1
            gaps.append(gap)

        # Also check wheel potential (A can be low)
        if 14 in unique_ranks:  # Ace
            wheel_check = unique_ranks + [1]  # Add ace as low
            unique_ranks_with_wheel = sorted(set(rank_values) | {1}, reverse=True)
        else:
            unique_ranks_with_wheel = unique_ranks

        # Connected: max gap of 2 between cards
        max_gap = max(gaps) if gaps else 0
        connected = max_gap <= 2 and len(unique_ranks) >= 2

        # Straight possible: 5 cards within 5-rank span
        straight_possible = False
        if len(unique_ranks_with_wheel) >= 3:
            for i in range(len(unique_ranks_with_wheel) - 2):
                window = unique_ranks_with_wheel[i:i+5] if i+5 <= len(unique_ranks_with_wheel) else unique_ranks_with_wheel[i:]
                if len(window) >= 3:
                    span = window[0] - window[-1]
                    if span <= 4:
                        straight_possible = True
                        break

        # Straight draw possible: 3+ cards within 5-rank span
        straight_draw_possible = connected and not straight_possible

        return connected, straight_possible, straight_draw_possible

    def _calculate_wetness(
        self,
        flush_possible: bool,
        flush_draw_possible: bool,
        straight_possible: bool,
        straight_draw_possible: bool,
        connected: bool,
        monotone: bool,
        pairedness: BoardPairedness
    ) -> BoardWetness:
        """Calculate overall board wetness."""
        wet_factors = 0

        if flush_possible:
            wet_factors += 3
        elif flush_draw_possible:
            wet_factors += 2

        if straight_possible:
            wet_factors += 2
        elif straight_draw_possible:
            wet_factors += 1

        if connected:
            wet_factors += 1

        if monotone:
            wet_factors += 2

        # Paired boards are drier (fewer combos hit)
        if pairedness != BoardPairedness.UNPAIRED:
            wet_factors -= 1

        if wet_factors <= 0:
            return BoardWetness.VERY_DRY
        elif wet_factors <= 1:
            return BoardWetness.DRY
        elif wet_factors <= 3:
            return BoardWetness.NEUTRAL
        elif wet_factors <= 5:
            return BoardWetness.WET
        else:
            return BoardWetness.VERY_WET


# Singleton analyzer
_analyzer = BoardAnalyzer()


def analyze_board(board: Board) -> BoardTexture:
    """Analyze board texture."""
    return _analyzer.analyze(board)


def get_texture_description(board: Board) -> str:
    """Get human-readable board texture description."""
    texture = analyze_board(board)
    return str(texture)


def is_favorable_cbet_board(board: Board) -> bool:
    """Check if board favors c-betting."""
    texture = analyze_board(board)
    return texture.favors_aggressor


def get_draw_potential(board: Board) -> dict[str, bool]:
    """Get draw possibilities on the board."""
    texture = analyze_board(board)
    return {
        'flush_possible': texture.flush_possible,
        'flush_draw': texture.flush_draw_possible,
        'straight_possible': texture.straight_possible,
        'straight_draw': texture.straight_draw_possible
    }
