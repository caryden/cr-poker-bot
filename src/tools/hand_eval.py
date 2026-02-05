"""
Hand evaluation for Texas Hold'em.

Evaluates 5-card poker hands and finds the best 5-card hand
from 7 cards (2 hole cards + 5 community cards).
"""

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Optional
from collections import Counter

from ..core.primitives import Card, Rank, Suit, HoleCards, Board


class HandRank(IntEnum):
    """
    Poker hand rankings from lowest to highest.

    Higher value = stronger hand.
    """
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

    def __str__(self) -> str:
        return self.name.replace('_', ' ').title()


@dataclass(frozen=True)
class HandEvaluation:
    """
    Result of evaluating a poker hand.

    Attributes:
        rank: The hand ranking (pair, flush, etc.)
        primary: Primary rank values for comparison (e.g., pair rank)
        kickers: Kicker cards for tiebreaking
        description: Human-readable description
        cards: The 5 cards making up this hand
    """
    rank: HandRank
    primary: tuple[int, ...]  # Primary ranking values
    kickers: tuple[int, ...]  # Kicker values for tiebreaking
    description: str
    cards: tuple[Card, ...]

    @property
    def score(self) -> tuple:
        """
        Comparable score tuple.

        Higher tuple = stronger hand (uses natural tuple comparison).
        """
        return (self.rank.value,) + self.primary + self.kickers

    def __lt__(self, other: 'HandEvaluation') -> bool:
        return self.score < other.score

    def __le__(self, other: 'HandEvaluation') -> bool:
        return self.score <= other.score

    def __gt__(self, other: 'HandEvaluation') -> bool:
        return self.score > other.score

    def __ge__(self, other: 'HandEvaluation') -> bool:
        return self.score >= other.score

    def __eq__(self, other: 'HandEvaluation') -> bool:
        return self.score == other.score

    def __str__(self) -> str:
        return self.description


class HandEvaluator:
    """
    Evaluates poker hands.

    Supports:
    - 5-card hand evaluation
    - Best 5 from 7 cards (Texas Hold'em)
    - Hand comparison
    """

    # Ace-low straight: A-2-3-4-5 (wheel)
    WHEEL = (14, 5, 4, 3, 2)

    def evaluate_5(self, cards: list[Card]) -> HandEvaluation:
        """
        Evaluate a 5-card poker hand.

        Args:
            cards: Exactly 5 cards

        Returns:
            HandEvaluation with rank and comparison info
        """
        if len(cards) != 5:
            raise ValueError(f"Expected 5 cards, got {len(cards)}")

        cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)
        ranks = [c.rank.value for c in cards]
        suits = [c.suit for c in cards]

        is_flush = len(set(suits)) == 1
        is_straight, straight_high = self._check_straight(ranks)

        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)
        # Sort by count desc, then by rank desc
        sorted_ranks = sorted(rank_counts.keys(),
                             key=lambda r: (rank_counts[r], r),
                             reverse=True)

        card_tuple = tuple(cards)

        # Check hand types from highest to lowest
        if is_straight and is_flush:
            if straight_high == 14 and min(ranks) == 10:
                return HandEvaluation(
                    HandRank.ROYAL_FLUSH,
                    (14,), (), "Royal Flush", card_tuple
                )
            return HandEvaluation(
                HandRank.STRAIGHT_FLUSH,
                (straight_high,), (),
                f"Straight Flush, {self._rank_name(straight_high)} high",
                card_tuple
            )

        if counts == [4, 1]:
            quad_rank = sorted_ranks[0]
            kicker = sorted_ranks[1]
            return HandEvaluation(
                HandRank.FOUR_OF_A_KIND,
                (quad_rank,), (kicker,),
                f"Four of a Kind, {self._rank_name(quad_rank)}s",
                card_tuple
            )

        if counts == [3, 2]:
            trips_rank = sorted_ranks[0]
            pair_rank = sorted_ranks[1]
            return HandEvaluation(
                HandRank.FULL_HOUSE,
                (trips_rank, pair_rank), (),
                f"Full House, {self._rank_name(trips_rank)}s full of {self._rank_name(pair_rank)}s",
                card_tuple
            )

        if is_flush:
            return HandEvaluation(
                HandRank.FLUSH,
                tuple(ranks), (),
                f"Flush, {self._rank_name(ranks[0])} high",
                card_tuple
            )

        if is_straight:
            return HandEvaluation(
                HandRank.STRAIGHT,
                (straight_high,), (),
                f"Straight, {self._rank_name(straight_high)} high",
                card_tuple
            )

        if counts == [3, 1, 1]:
            trips_rank = sorted_ranks[0]
            kickers = tuple(sorted_ranks[1:])
            return HandEvaluation(
                HandRank.THREE_OF_A_KIND,
                (trips_rank,), kickers,
                f"Three of a Kind, {self._rank_name(trips_rank)}s",
                card_tuple
            )

        if counts == [2, 2, 1]:
            high_pair = sorted_ranks[0]
            low_pair = sorted_ranks[1]
            kicker = sorted_ranks[2]
            return HandEvaluation(
                HandRank.TWO_PAIR,
                (high_pair, low_pair), (kicker,),
                f"Two Pair, {self._rank_name(high_pair)}s and {self._rank_name(low_pair)}s",
                card_tuple
            )

        if counts == [2, 1, 1, 1]:
            pair_rank = sorted_ranks[0]
            kickers = tuple(sorted_ranks[1:])
            return HandEvaluation(
                HandRank.PAIR,
                (pair_rank,), kickers,
                f"Pair of {self._rank_name(pair_rank)}s",
                card_tuple
            )

        # High card
        return HandEvaluation(
            HandRank.HIGH_CARD,
            (ranks[0],), tuple(ranks[1:]),
            f"High Card, {self._rank_name(ranks[0])}",
            card_tuple
        )

    def evaluate_holdem(self, hole_cards: HoleCards, board: Board) -> HandEvaluation:
        """
        Find the best 5-card hand from hole cards + board.

        Args:
            hole_cards: Player's 2 hole cards
            board: Community cards (3-5 cards)

        Returns:
            Best possible HandEvaluation
        """
        all_cards = list(hole_cards) + board.cards

        if len(all_cards) < 5:
            raise ValueError(f"Need at least 5 cards, got {len(all_cards)}")

        best = None
        for combo in combinations(all_cards, 5):
            evaluation = self.evaluate_5(list(combo))
            if best is None or evaluation > best:
                best = evaluation

        return best

    def evaluate_7(self, cards: list[Card]) -> HandEvaluation:
        """
        Find the best 5-card hand from 7 cards.

        Args:
            cards: Exactly 7 cards

        Returns:
            Best possible HandEvaluation
        """
        if len(cards) != 7:
            raise ValueError(f"Expected 7 cards, got {len(cards)}")

        best = None
        for combo in combinations(cards, 5):
            evaluation = self.evaluate_5(list(combo))
            if best is None or evaluation > best:
                best = evaluation

        return best

    def compare(self, hand1: HandEvaluation, hand2: HandEvaluation) -> int:
        """
        Compare two hands.

        Returns:
            1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        if hand1 > hand2:
            return 1
        elif hand1 < hand2:
            return -1
        return 0

    def _check_straight(self, ranks: list[int]) -> tuple[bool, int]:
        """
        Check if ranks form a straight.

        Returns:
            (is_straight, high_card) tuple
        """
        unique = sorted(set(ranks), reverse=True)
        if len(unique) != 5:
            return False, 0

        # Check normal straight
        if unique[0] - unique[4] == 4:
            return True, unique[0]

        # Check wheel (A-2-3-4-5)
        if tuple(unique) == (14, 5, 4, 3, 2):
            return True, 5  # 5-high straight

        return False, 0

    def _rank_name(self, rank_value: int) -> str:
        """Get readable name for a rank value."""
        names = {
            14: 'Ace', 13: 'King', 12: 'Queen', 11: 'Jack',
            10: 'Ten', 9: 'Nine', 8: 'Eight', 7: 'Seven',
            6: 'Six', 5: 'Five', 4: 'Four', 3: 'Three', 2: 'Two'
        }
        return names.get(rank_value, str(rank_value))


# Singleton instance for convenience
_evaluator = HandEvaluator()


def evaluate_hand(hole_cards: HoleCards, board: Board) -> HandEvaluation:
    """
    Convenience function to evaluate a Hold'em hand.

    Args:
        hole_cards: Player's hole cards
        board: Community cards

    Returns:
        Best 5-card hand evaluation
    """
    return _evaluator.evaluate_holdem(hole_cards, board)


def compare_hands(hand1: HandEvaluation, hand2: HandEvaluation) -> int:
    """
    Compare two evaluated hands.

    Returns:
        1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    return _evaluator.compare(hand1, hand2)
