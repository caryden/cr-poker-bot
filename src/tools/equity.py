"""
Equity calculator for Texas Hold'em.

Uses Monte Carlo simulation to estimate hand equity against
opponent ranges or random hands.
"""

from dataclasses import dataclass
from typing import Optional
import random
from itertools import combinations

from ..core.primitives import Card, Rank, Suit, HoleCards, Board
from .hand_eval import HandEvaluator, HandEvaluation


@dataclass
class EquityResult:
    """
    Result of an equity calculation.

    Attributes:
        equity: Win probability (0.0 to 1.0)
        win_pct: Win percentage
        tie_pct: Tie percentage
        lose_pct: Lose percentage
        simulations: Number of simulations run
        confidence: Statistical confidence in result
    """
    equity: float
    win_pct: float
    tie_pct: float
    lose_pct: float
    simulations: int
    confidence: float = 0.95

    def __str__(self) -> str:
        return f"Equity: {self.equity:.1%} (W:{self.win_pct:.1%} T:{self.tie_pct:.1%} L:{self.lose_pct:.1%})"


class Deck:
    """A deck of 52 playing cards."""

    def __init__(self):
        self.cards: list[Card] = []
        self.reset()

    def reset(self) -> None:
        """Reset deck to full 52 cards."""
        self.cards = [
            Card(rank, suit)
            for suit in Suit
            for rank in Rank
        ]

    def remove(self, cards: list[Card]) -> None:
        """Remove specific cards from deck."""
        for card in cards:
            if card in self.cards:
                self.cards.remove(card)

    def draw(self, n: int = 1) -> list[Card]:
        """Draw n random cards from deck."""
        if n > len(self.cards):
            raise ValueError(f"Cannot draw {n} cards, only {len(self.cards)} remaining")
        drawn = random.sample(self.cards, n)
        for card in drawn:
            self.cards.remove(card)
        return drawn

    def shuffle(self) -> None:
        """Shuffle remaining cards."""
        random.shuffle(self.cards)

    def __len__(self) -> int:
        return len(self.cards)


class HandRange:
    """
    Represents a range of possible hands.

    Can be constructed from:
    - Specific hands: "AKs", "QQ"
    - Range notation: "JJ+", "AQs+", "ATo+"
    - Percentage: top 20% of hands
    """

    # Precomputed hand rankings by strength (approximate)
    HAND_RANKINGS = [
        # Premium pairs
        'AA', 'KK', 'QQ', 'JJ', 'TT',
        # Big aces
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo',
        # Medium pairs
        '99', '88', '77',
        # Broadway
        'KQs', 'KQo', 'ATs', 'ATo', 'KJs', 'KJo',
        # Small-medium pairs
        '66', '55', '44', '33', '22',
        # Suited connectors
        'QJs', 'JTs', 'T9s', '98s', '87s', '76s', '65s',
        # More aces
        'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
        # More broadway
        'KTs', 'QTs', 'J9s', 'T8s', '97s', '86s', '75s', '64s', '54s',
        # Offsuit broadway
        'QJo', 'JTo', 'KTo', 'QTo',
        # Rest of aces
        'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o',
        # Remaining
        'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s',
        'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s',
        'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s',
        'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s',
        '96s', '95s', '94s', '93s', '92s',
        '85s', '84s', '83s', '82s',
        '74s', '73s', '72s',
        '63s', '62s',
        '53s', '52s',
        '43s', '42s',
        '32s',
    ]

    def __init__(self, hands: Optional[set[str]] = None):
        """
        Initialize with a set of hand notations.

        Args:
            hands: Set of hand notations like {'AA', 'AKs', 'AKo'}
        """
        self.hands = hands or set()

    @classmethod
    def from_string(cls, range_str: str) -> 'HandRange':
        """
        Parse range from string notation.

        Examples:
            "AA,KK,QQ" - specific hands
            "JJ+" - pairs JJ and above
            "AQs+" - suited aces AQ and above
            "22-77" - pair range
        """
        hands = set()
        range_str = range_str.replace(' ', '')

        for part in range_str.split(','):
            part = part.strip()
            if not part:
                continue

            if '+' in part:
                hands.update(cls._parse_plus_notation(part.replace('+', '')))
            elif '-' in part:
                hands.update(cls._parse_range_notation(part))
            else:
                hands.add(part)

        return cls(hands)

    @classmethod
    def from_percentage(cls, pct: float) -> 'HandRange':
        """
        Create range from top X% of hands.

        Args:
            pct: Percentage (0-100)
        """
        # Each hand notation represents multiple combos
        # Pairs: 6 combos, Suited: 4 combos, Offsuit: 12 combos
        # Total: 1326 combos
        target_combos = int(1326 * pct / 100)

        hands = set()
        combos = 0

        for hand in cls.HAND_RANKINGS:
            if len(hand) == 2:  # Pair
                combos += 6
            elif hand.endswith('s'):  # Suited
                combos += 4
            else:  # Offsuit
                combos += 12

            hands.add(hand)
            if combos >= target_combos:
                break

        return cls(hands)

    @classmethod
    def _parse_plus_notation(cls, base: str) -> set[str]:
        """Parse JJ+ or AQs+ notation."""
        hands = set()

        if len(base) == 2 and base[0] == base[1]:
            # Pair notation: JJ+
            pair_rank = base[0]
            rank_order = 'AKQJT98765432'
            start_idx = rank_order.index(pair_rank)
            for i in range(start_idx + 1):
                r = rank_order[i]
                hands.add(f'{r}{r}')
        elif len(base) == 3:
            # Suited/offsuit: AQs+
            high, low, suited = base[0], base[1], base[2]
            rank_order = 'AKQJT98765432'
            high_idx = rank_order.index(high)
            low_idx = rank_order.index(low)

            # Add hands from low to A (second card)
            for i in range(high_idx + 1, low_idx + 1):
                hands.add(f'{high}{rank_order[i]}{suited}')

        return hands

    @classmethod
    def _parse_range_notation(cls, range_str: str) -> set[str]:
        """Parse 22-77 or ATs-AQs notation."""
        parts = range_str.split('-')
        if len(parts) != 2:
            return set()

        start, end = parts
        hands = set()

        if len(start) == 2 and start[0] == start[1]:
            # Pair range: 22-77
            rank_order = 'AKQJT98765432'
            start_idx = rank_order.index(start[0])
            end_idx = rank_order.index(end[0])

            for i in range(min(start_idx, end_idx), max(start_idx, end_idx) + 1):
                r = rank_order[i]
                hands.add(f'{r}{r}')

        return hands

    def get_combos(self, dead_cards: list[Card] = None) -> list[HoleCards]:
        """
        Generate all specific card combinations for this range.

        Args:
            dead_cards: Cards that cannot be used

        Returns:
            List of HoleCards combinations
        """
        dead = set(dead_cards) if dead_cards else set()
        combos = []

        for hand in self.hands:
            combos.extend(self._notation_to_combos(hand, dead))

        return combos

    def _notation_to_combos(self, notation: str,
                           dead: set[Card]) -> list[HoleCards]:
        """Convert notation like 'AKs' to actual card combinations."""
        combos = []

        if len(notation) == 2:
            # Pair: AA
            rank = Rank.from_char(notation[0])
            for s1, s2 in combinations(Suit, 2):
                c1 = Card(rank, s1)
                c2 = Card(rank, s2)
                if c1 not in dead and c2 not in dead:
                    combos.append(HoleCards(c1, c2))

        elif len(notation) == 3:
            rank1 = Rank.from_char(notation[0])
            rank2 = Rank.from_char(notation[1])
            suited = notation[2] == 's'

            if suited:
                # Suited: same suit
                for suit in Suit:
                    c1 = Card(rank1, suit)
                    c2 = Card(rank2, suit)
                    if c1 not in dead and c2 not in dead:
                        combos.append(HoleCards(c1, c2))
            else:
                # Offsuit: different suits
                for s1 in Suit:
                    for s2 in Suit:
                        if s1 != s2:
                            c1 = Card(rank1, s1)
                            c2 = Card(rank2, s2)
                            if c1 not in dead and c2 not in dead:
                                combos.append(HoleCards(c1, c2))

        return combos

    def __len__(self) -> int:
        return len(self.hands)

    def __contains__(self, hand: str) -> bool:
        return hand in self.hands

    def __str__(self) -> str:
        return ','.join(sorted(self.hands))


class EquityCalculator:
    """
    Calculate hand equity using Monte Carlo simulation.
    """

    def __init__(self, evaluator: Optional[HandEvaluator] = None):
        self.evaluator = evaluator or HandEvaluator()

    def calculate_vs_random(
        self,
        hero_cards: HoleCards,
        board: Board,
        num_opponents: int = 1,
        simulations: int = 10000
    ) -> EquityResult:
        """
        Calculate equity against random hands.

        Args:
            hero_cards: Hero's hole cards
            board: Current board (can be empty for preflop)
            num_opponents: Number of opponents
            simulations: Number of Monte Carlo iterations

        Returns:
            EquityResult with win/tie/lose percentages
        """
        wins = 0
        ties = 0
        losses = 0

        dead_cards = list(hero_cards) + board.cards
        cards_to_deal = 5 - len(board.cards)

        for _ in range(simulations):
            deck = Deck()
            deck.remove(dead_cards)

            # Deal remaining board cards
            runout = deck.draw(cards_to_deal)
            full_board = Board(board.cards + runout)

            # Deal opponent hands
            opponent_hands = []
            for _ in range(num_opponents):
                opp_cards = deck.draw(2)
                opponent_hands.append(HoleCards(opp_cards[0], opp_cards[1]))

            # Evaluate all hands
            hero_eval = self.evaluator.evaluate_holdem(hero_cards, full_board)
            opponent_evals = [
                self.evaluator.evaluate_holdem(opp, full_board)
                for opp in opponent_hands
            ]

            # Compare
            best_opponent = max(opponent_evals)
            if hero_eval > best_opponent:
                wins += 1
            elif hero_eval < best_opponent:
                losses += 1
            else:
                ties += 1

        total = wins + ties + losses
        win_pct = wins / total
        tie_pct = ties / total
        lose_pct = losses / total

        # Equity includes full wins plus share of ties
        equity = win_pct + tie_pct / 2

        return EquityResult(
            equity=equity,
            win_pct=win_pct,
            tie_pct=tie_pct,
            lose_pct=lose_pct,
            simulations=simulations
        )

    def calculate_vs_range(
        self,
        hero_cards: HoleCards,
        board: Board,
        opponent_range: HandRange,
        simulations: int = 10000
    ) -> EquityResult:
        """
        Calculate equity against a specific range.

        Args:
            hero_cards: Hero's hole cards
            board: Current board
            opponent_range: Opponent's hand range
            simulations: Number of iterations

        Returns:
            EquityResult
        """
        wins = 0
        ties = 0
        losses = 0

        dead_cards = list(hero_cards) + board.cards
        cards_to_deal = 5 - len(board.cards)

        # Get valid opponent combos
        opponent_combos = opponent_range.get_combos(dead_cards)
        if not opponent_combos:
            raise ValueError("No valid opponent hands in range")

        for _ in range(simulations):
            deck = Deck()
            deck.remove(dead_cards)

            # Pick random opponent hand from range
            opp_hand = random.choice(opponent_combos)

            # Check if opponent cards available
            opp_cards = list(opp_hand)
            if any(c not in deck.cards for c in opp_cards):
                continue  # Skip this iteration

            deck.remove(opp_cards)

            # Deal remaining board
            runout = deck.draw(cards_to_deal)
            full_board = Board(board.cards + runout)

            # Evaluate
            hero_eval = self.evaluator.evaluate_holdem(hero_cards, full_board)
            opp_eval = self.evaluator.evaluate_holdem(opp_hand, full_board)

            if hero_eval > opp_eval:
                wins += 1
            elif hero_eval < opp_eval:
                losses += 1
            else:
                ties += 1

        total = wins + ties + losses
        if total == 0:
            return EquityResult(0.5, 0.5, 0.0, 0.5, 0)

        win_pct = wins / total
        tie_pct = ties / total
        lose_pct = losses / total
        equity = win_pct + tie_pct / 2

        return EquityResult(
            equity=equity,
            win_pct=win_pct,
            tie_pct=tie_pct,
            lose_pct=lose_pct,
            simulations=total
        )

    def calculate_matchup(
        self,
        hand1: HoleCards,
        hand2: HoleCards,
        board: Optional[Board] = None,
        simulations: int = 10000
    ) -> tuple[EquityResult, EquityResult]:
        """
        Calculate equity for a specific hand matchup.

        Args:
            hand1: First hand
            hand2: Second hand
            board: Optional board cards
            simulations: Number of iterations

        Returns:
            Tuple of (hand1_equity, hand2_equity)
        """
        board = board or Board()
        wins1, wins2, ties = 0, 0, 0

        dead_cards = list(hand1) + list(hand2) + board.cards
        cards_to_deal = 5 - len(board.cards)

        for _ in range(simulations):
            deck = Deck()
            deck.remove(dead_cards)

            runout = deck.draw(cards_to_deal)
            full_board = Board(board.cards + runout)

            eval1 = self.evaluator.evaluate_holdem(hand1, full_board)
            eval2 = self.evaluator.evaluate_holdem(hand2, full_board)

            if eval1 > eval2:
                wins1 += 1
            elif eval2 > eval1:
                wins2 += 1
            else:
                ties += 1

        total = wins1 + wins2 + ties
        eq1 = (wins1 + ties / 2) / total
        eq2 = (wins2 + ties / 2) / total

        result1 = EquityResult(
            equity=eq1,
            win_pct=wins1 / total,
            tie_pct=ties / total,
            lose_pct=wins2 / total,
            simulations=simulations
        )
        result2 = EquityResult(
            equity=eq2,
            win_pct=wins2 / total,
            tie_pct=ties / total,
            lose_pct=wins1 / total,
            simulations=simulations
        )

        return result1, result2


# Singleton instance
_calculator = EquityCalculator()


def calculate_equity(
    hero_cards: HoleCards,
    board: Board,
    num_opponents: int = 1,
    simulations: int = 10000
) -> EquityResult:
    """
    Convenience function to calculate equity vs random hands.
    """
    return _calculator.calculate_vs_random(hero_cards, board, num_opponents, simulations)


def calculate_equity_vs_range(
    hero_cards: HoleCards,
    board: Board,
    opponent_range: HandRange,
    simulations: int = 10000
) -> EquityResult:
    """
    Convenience function to calculate equity vs a range.
    """
    return _calculator.calculate_vs_range(hero_cards, board, opponent_range, simulations)
