"""
Pot odds and betting math utilities.

Provides calculations for:
- Pot odds
- Implied odds
- Expected Value (EV)
- Minimum Defense Frequency (MDF)
- Bet sizing relative to pot
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from ..core.game_state import GameState


class BetSize(Enum):
    """Common bet sizing categories."""
    QUARTER_POT = 0.25
    THIRD_POT = 0.33
    HALF_POT = 0.5
    TWO_THIRDS_POT = 0.67
    THREE_QUARTERS_POT = 0.75
    POT = 1.0
    OVERBET_1_5X = 1.5
    OVERBET_2X = 2.0

    @property
    def name_str(self) -> str:
        names = {
            0.25: "1/4 pot",
            0.33: "1/3 pot",
            0.5: "1/2 pot",
            0.67: "2/3 pot",
            0.75: "3/4 pot",
            1.0: "pot",
            1.5: "1.5x pot",
            2.0: "2x pot"
        }
        return names.get(self.value, f"{self.value}x pot")


@dataclass
class PotOddsResult:
    """
    Result of pot odds calculation.

    Attributes:
        pot_odds: Required equity to break even (0.0 to 1.0)
        pot_odds_ratio: Odds expressed as ratio (e.g., "3:1")
        to_call: Amount needed to call
        pot_after_call: Pot size after calling
        break_even_equity: Minimum equity needed to call profitably
    """
    pot_odds: float
    pot_odds_ratio: str
    to_call: float
    pot_after_call: float
    break_even_equity: float

    def __str__(self) -> str:
        return f"Pot odds: {self.pot_odds:.1%} ({self.pot_odds_ratio}) - Need {self.break_even_equity:.1%} equity"


@dataclass
class EVResult:
    """
    Expected Value calculation result.

    Attributes:
        ev: Expected value in chips
        ev_bb: EV in big blinds
        is_profitable: Whether EV is positive
        action_description: Description of the action analyzed
    """
    ev: float
    ev_bb: float
    is_profitable: bool
    action_description: str

    def __str__(self) -> str:
        sign = "+" if self.ev >= 0 else ""
        return f"{self.action_description}: EV = {sign}{self.ev:.2f} ({sign}{self.ev_bb:.2f} BB)"


@dataclass
class MDFResult:
    """
    Minimum Defense Frequency result.

    Attributes:
        mdf: Minimum frequency to defend (0.0 to 1.0)
        fold_frequency: Maximum fold frequency (1 - mdf)
        bet_size_ratio: Bet size as fraction of pot
    """
    mdf: float
    fold_frequency: float
    bet_size_ratio: float

    def __str__(self) -> str:
        return f"MDF: {self.mdf:.1%} (can fold up to {self.fold_frequency:.1%})"


class PotOddsCalculator:
    """
    Calculator for pot odds and betting math.
    """

    def pot_odds(self, pot: float, to_call: float) -> PotOddsResult:
        """
        Calculate pot odds for a call.

        Pot odds = to_call / (pot + to_call)

        Args:
            pot: Current pot size
            to_call: Amount to call

        Returns:
            PotOddsResult with odds and break-even equity
        """
        if to_call <= 0:
            return PotOddsResult(
                pot_odds=0.0,
                pot_odds_ratio="free",
                to_call=0,
                pot_after_call=pot,
                break_even_equity=0.0
            )

        pot_after = pot + to_call
        odds = to_call / pot_after

        # Calculate ratio (pot : call)
        ratio = pot / to_call
        ratio_str = f"{ratio:.1f}:1"

        return PotOddsResult(
            pot_odds=odds,
            pot_odds_ratio=ratio_str,
            to_call=to_call,
            pot_after_call=pot_after,
            break_even_equity=odds
        )

    def pot_odds_from_state(self, state: GameState) -> PotOddsResult:
        """
        Calculate pot odds from game state.
        """
        return self.pot_odds(state.pot.total, state.to_call)

    def implied_odds(
        self,
        pot: float,
        to_call: float,
        expected_future_winnings: float
    ) -> PotOddsResult:
        """
        Calculate implied odds including expected future winnings.

        Args:
            pot: Current pot
            to_call: Amount to call
            expected_future_winnings: Expected additional winnings if we hit

        Returns:
            PotOddsResult with implied odds
        """
        effective_pot = pot + expected_future_winnings
        return self.pot_odds(effective_pot, to_call)

    def expected_value_call(
        self,
        pot: float,
        to_call: float,
        equity: float,
        bb_size: float = 1.0
    ) -> EVResult:
        """
        Calculate EV of calling.

        EV = (equity * pot_won) - ((1 - equity) * amount_lost)

        Args:
            pot: Current pot size
            to_call: Amount to call
            equity: Our equity (0.0 to 1.0)
            bb_size: Big blind size for BB conversion

        Returns:
            EVResult
        """
        # If we win, we win the pot (not including our call which is sunk)
        pot_won = pot  # We get the pot
        # If we lose, we lose our call
        amount_lost = to_call

        ev = (equity * pot_won) - ((1 - equity) * amount_lost)
        ev_bb = ev / bb_size

        return EVResult(
            ev=ev,
            ev_bb=ev_bb,
            is_profitable=ev > 0,
            action_description=f"Call {to_call:.0f}"
        )

    def expected_value_bet(
        self,
        pot: float,
        bet_size: float,
        fold_equity: float,
        equity_when_called: float,
        bb_size: float = 1.0
    ) -> EVResult:
        """
        Calculate EV of betting.

        EV = (fold_eq * pot) + ((1 - fold_eq) * (eq_called * new_pot - (1 - eq_called) * bet))

        Args:
            pot: Current pot
            bet_size: Our bet size
            fold_equity: Probability opponent folds
            equity_when_called: Our equity when called
            bb_size: Big blind size

        Returns:
            EVResult
        """
        # When they fold, we win current pot
        ev_fold = fold_equity * pot

        # When they call
        new_pot = pot + bet_size  # Pot after they call (they match our bet)
        ev_win_called = equity_when_called * new_pot
        ev_lose_called = (1 - equity_when_called) * bet_size
        ev_called = (1 - fold_equity) * (ev_win_called - ev_lose_called)

        ev = ev_fold + ev_called
        ev_bb = ev / bb_size

        return EVResult(
            ev=ev,
            ev_bb=ev_bb,
            is_profitable=ev > 0,
            action_description=f"Bet {bet_size:.0f}"
        )

    def minimum_defense_frequency(self, bet_size: float, pot: float) -> MDFResult:
        """
        Calculate Minimum Defense Frequency.

        MDF = pot / (pot + bet)

        This is how often we must defend to prevent opponent from
        profitably bluffing with any two cards.

        Args:
            bet_size: Opponent's bet size
            pot: Pot before the bet

        Returns:
            MDFResult
        """
        if bet_size <= 0:
            return MDFResult(mdf=1.0, fold_frequency=0.0, bet_size_ratio=0.0)

        mdf = pot / (pot + bet_size)
        fold_freq = 1 - mdf
        ratio = bet_size / pot if pot > 0 else 0

        return MDFResult(
            mdf=mdf,
            fold_frequency=fold_freq,
            bet_size_ratio=ratio
        )

    def required_fold_equity(
        self,
        pot: float,
        bet_size: float,
        equity_when_called: float = 0.0
    ) -> float:
        """
        Calculate required fold equity for a bluff to be profitable.

        For a pure bluff (0 equity when called):
        Required FE = bet / (pot + bet)

        Args:
            pot: Current pot
            bet_size: Our bet size
            equity_when_called: Our equity if called (0 for pure bluff)

        Returns:
            Required fold equity (0.0 to 1.0)
        """
        if equity_when_called >= 1.0:
            return 0.0  # Don't need fold equity with nuts

        # Break-even fold equity
        # EV = FE * pot + (1-FE) * (eq * (pot + bet) - bet)
        # 0 = FE * pot + (1-FE) * (eq * (pot + bet) - bet)
        # Solving for FE when eq = 0:
        # FE = bet / (pot + bet)

        if equity_when_called == 0:
            return bet_size / (pot + bet_size)

        # With some equity:
        pot_with_bet = pot + bet_size
        ev_when_called = equity_when_called * pot_with_bet - bet_size

        if ev_when_called >= 0:
            return 0.0  # Profitable even when called

        # FE * pot = -ev_when_called * (1 - FE)
        required = -ev_when_called / (pot - ev_when_called)
        return max(0.0, min(1.0, required))

    def bet_size_for_pot_odds(self, pot: float, target_odds: float) -> float:
        """
        Calculate bet size to give opponent specific pot odds.

        Args:
            pot: Current pot
            target_odds: Desired pot odds (e.g., 0.25 for 25%)

        Returns:
            Required bet size
        """
        # odds = bet / (pot + bet)
        # odds * (pot + bet) = bet
        # odds * pot + odds * bet = bet
        # odds * pot = bet - odds * bet
        # odds * pot = bet * (1 - odds)
        # bet = odds * pot / (1 - odds)

        if target_odds >= 1:
            return float('inf')

        return (target_odds * pot) / (1 - target_odds)

    def stack_to_pot_ratio(self, effective_stack: float, pot: float) -> float:
        """
        Calculate Stack-to-Pot Ratio (SPR).

        Args:
            effective_stack: Smaller of the relevant stacks
            pot: Current pot

        Returns:
            SPR value
        """
        if pot <= 0:
            return float('inf')
        return effective_stack / pot

    def spr_from_state(self, state: GameState) -> float:
        """Calculate SPR from game state."""
        return state.spr

    def commitment_threshold(self, spr: float) -> str:
        """
        Interpret SPR for commitment decisions.

        Args:
            spr: Stack-to-Pot Ratio

        Returns:
            Description of commitment level
        """
        if spr < 2:
            return "Very low SPR - often committed with top pair+"
        elif spr < 4:
            return "Low SPR - committed with strong hands"
        elif spr < 8:
            return "Medium SPR - can play postflop poker"
        elif spr < 13:
            return "High SPR - deep stacked play"
        else:
            return "Very high SPR - extremely deep"


# Singleton
_calculator = PotOddsCalculator()


def pot_odds(pot: float, to_call: float) -> PotOddsResult:
    """Calculate pot odds."""
    return _calculator.pot_odds(pot, to_call)


def expected_value_call(pot: float, to_call: float, equity: float) -> EVResult:
    """Calculate EV of calling."""
    return _calculator.expected_value_call(pot, to_call, equity)


def minimum_defense_frequency(bet_size: float, pot: float) -> MDFResult:
    """Calculate MDF."""
    return _calculator.minimum_defense_frequency(bet_size, pot)


def required_fold_equity(pot: float, bet_size: float) -> float:
    """Calculate required fold equity for bluff."""
    return _calculator.required_fold_equity(pot, bet_size)
