"""
Opponent memory system for tracking raw statistics and hand histories.

Raw statistics (frequentist counts) feed into the subjective logic belief system.
This separation allows:
- Stats to accumulate precisely over many hands
- Beliefs to be computed from stats with appropriate uncertainty
- Hand histories to provide context for LLM reasoning
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from typing import Optional
from enum import Enum, auto

from .primitives import Action, ActionType, Position, Street, HoleCards
from .subjective_logic import Opinion, MultinomialOpinion
from .beliefs import (
    VillainBeliefs, PLAYER_TYPES, DEFAULT_PLAYER_TYPE_BASE_RATES,
    BeliefCategory
)


@dataclass
class HandSummary:
    """Summary of a single hand for memory."""
    hand_id: str
    timestamp: datetime
    position: Position
    hole_cards: Optional[HoleCards]  # Only if shown

    # Key actions taken
    preflop_action: Optional[ActionType] = None
    postflop_aggression: int = 0  # Count of bets/raises postflop
    went_to_showdown: bool = False
    won_pot: bool = False
    pot_size: float = 0.0

    # Context
    facing_raise_preflop: bool = False
    was_3bet_pot: bool = False

    def is_notable(self) -> bool:
        """Is this hand worth remembering in detail?"""
        # Large pots, showdowns, unusual plays
        return (
            self.pot_size > 50 or  # Big pot (in BB)
            self.went_to_showdown or
            self.postflop_aggression >= 3  # Very aggressive line
        )


@dataclass
class OpponentStats:
    """
    Raw aggregate statistics for an opponent.

    These are frequentist counts that feed into subjective logic beliefs.
    The conversion to beliefs applies appropriate uncertainty based on sample size.
    """
    # Preflop stats
    vpip_hands: int = 0
    vpip_opportunities: int = 0
    pfr_hands: int = 0
    pfr_opportunities: int = 0

    # 3-bet stats
    three_bet_made: int = 0
    three_bet_opportunities: int = 0
    fold_to_3bet: int = 0
    faced_3bet: int = 0

    # C-bet stats
    cbet_made: int = 0
    cbet_opportunities: int = 0
    fold_to_cbet: int = 0
    faced_cbet: int = 0

    # Postflop aggression
    postflop_bets: int = 0
    postflop_raises: int = 0
    postflop_calls: int = 0
    postflop_folds: int = 0

    # Showdown stats
    went_to_showdown: int = 0
    could_showdown: int = 0  # Hands where player reached river
    won_at_showdown: int = 0

    # Bluffing indicators
    showed_bluff: int = 0  # Showed weak hand after aggression
    showed_value: int = 0  # Showed strong hand after aggression

    @property
    def total_hands(self) -> int:
        """Total hands observed."""
        return self.vpip_opportunities

    @property
    def vpip(self) -> float:
        """Voluntarily Put $ In Pot percentage."""
        if self.vpip_opportunities == 0:
            return 0.0
        return self.vpip_hands / self.vpip_opportunities

    @property
    def pfr(self) -> float:
        """Pre-Flop Raise percentage."""
        if self.pfr_opportunities == 0:
            return 0.0
        return self.pfr_hands / self.pfr_opportunities

    @property
    def three_bet_pct(self) -> float:
        """3-bet percentage."""
        if self.three_bet_opportunities == 0:
            return 0.0
        return self.three_bet_made / self.three_bet_opportunities

    @property
    def fold_to_3bet_pct(self) -> float:
        """Fold to 3-bet percentage."""
        if self.faced_3bet == 0:
            return 0.0
        return self.fold_to_3bet / self.faced_3bet

    @property
    def cbet_pct(self) -> float:
        """Continuation bet percentage."""
        if self.cbet_opportunities == 0:
            return 0.0
        return self.cbet_made / self.cbet_opportunities

    @property
    def fold_to_cbet_pct(self) -> float:
        """Fold to c-bet percentage."""
        if self.faced_cbet == 0:
            return 0.0
        return self.fold_to_cbet / self.faced_cbet

    @property
    def aggression_factor(self) -> float:
        """
        Aggression Factor = (bets + raises) / calls.

        Higher = more aggressive. Typical values:
        - < 1: Passive
        - 1-2: Slightly aggressive
        - 2-3: Aggressive
        - > 3: Very aggressive / maniac
        """
        if self.postflop_calls == 0:
            return float('inf') if (self.postflop_bets + self.postflop_raises) > 0 else 0.0
        return (self.postflop_bets + self.postflop_raises) / self.postflop_calls

    @property
    def wtsd(self) -> float:
        """Went To ShowDown percentage."""
        if self.could_showdown == 0:
            return 0.0
        return self.went_to_showdown / self.could_showdown

    @property
    def won_at_sd(self) -> float:
        """Won $ at ShowDown percentage."""
        if self.went_to_showdown == 0:
            return 0.0
        return self.won_at_showdown / self.went_to_showdown

    def get_sample_quality(self) -> str:
        """Assess quality of statistical sample."""
        hands = self.total_hands
        if hands < 20:
            return "tiny"
        elif hands < 50:
            return "small"
        elif hands < 100:
            return "medium"
        elif hands < 300:
            return "good"
        else:
            return "large"


class OpponentMemory:
    """
    Complete memory for a single opponent.

    Combines:
    - Raw statistics (frequentist counts)
    - Recent hand histories (for LLM context)
    - Notable hands (big pots, unusual plays)
    """

    def __init__(self, opponent_id: str, max_recent_hands: int = 20):
        self.opponent_id = opponent_id
        self.stats = OpponentStats()
        self.recent_hands: deque[HandSummary] = deque(maxlen=max_recent_hands)
        self.notable_hands: list[HandSummary] = []
        self.first_seen: Optional[datetime] = None
        self.last_seen: Optional[datetime] = None

    def record_preflop_action(self, action: ActionType, position: Position,
                              facing_raise: bool = False) -> None:
        """Record a preflop action."""
        now = datetime.now()
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now

        # VPIP opportunity (everyone gets one per hand)
        self.stats.vpip_opportunities += 1
        self.stats.pfr_opportunities += 1

        if action in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            self.stats.vpip_hands += 1

        if action in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            self.stats.pfr_hands += 1

            # 3-bet tracking
            if facing_raise:
                self.stats.three_bet_opportunities += 1
                self.stats.three_bet_made += 1

        # Fold to 3-bet
        if facing_raise and action == ActionType.FOLD:
            self.stats.faced_3bet += 1
            self.stats.fold_to_3bet += 1
        elif facing_raise and action != ActionType.FOLD:
            self.stats.faced_3bet += 1

    def record_cbet_opportunity(self, made_cbet: bool) -> None:
        """Record a continuation bet opportunity."""
        self.stats.cbet_opportunities += 1
        if made_cbet:
            self.stats.cbet_made += 1

    def record_faced_cbet(self, folded: bool) -> None:
        """Record facing a continuation bet."""
        self.stats.faced_cbet += 1
        if folded:
            self.stats.fold_to_cbet += 1

    def record_postflop_action(self, action: ActionType) -> None:
        """Record a postflop action."""
        if action == ActionType.BET:
            self.stats.postflop_bets += 1
        elif action == ActionType.RAISE:
            self.stats.postflop_raises += 1
        elif action == ActionType.CALL:
            self.stats.postflop_calls += 1
        elif action == ActionType.FOLD:
            self.stats.postflop_folds += 1

    def record_showdown(self, won: bool, showed_strong: bool) -> None:
        """Record showdown result."""
        self.stats.could_showdown += 1
        self.stats.went_to_showdown += 1
        if won:
            self.stats.won_at_showdown += 1

        # Track what they showed (for bluff frequency)
        if showed_strong:
            self.stats.showed_value += 1
        else:
            self.stats.showed_bluff += 1

    def record_reached_river_no_showdown(self) -> None:
        """Record reaching river but not going to showdown."""
        self.stats.could_showdown += 1

    def add_hand_summary(self, summary: HandSummary) -> None:
        """Add a hand summary to recent history."""
        self.recent_hands.append(summary)
        if summary.is_notable():
            self.notable_hands.append(summary)
            # Keep only last 20 notable hands
            if len(self.notable_hands) > 20:
                self.notable_hands = self.notable_hands[-20:]

    def to_beliefs(self) -> VillainBeliefs:
        """
        Convert raw statistics to subjective logic beliefs.

        Uses Opinion.from_frequency() to map counts to beliefs with
        appropriate uncertainty based on sample size.
        """
        beliefs = VillainBeliefs(self.opponent_id)

        # Infer player type from stats
        beliefs.player_type = self._infer_player_type()

        # Aggression beliefs
        af = self.stats.aggression_factor
        if self.stats.postflop_calls > 0 or (self.stats.postflop_bets + self.stats.postflop_raises) > 0:
            # Map AF to aggressive belief
            # AF < 1 = passive, AF > 2 = aggressive
            if af > 2:
                agg_successes = min(int(af * 10), 50)
                agg_failures = 10
            elif af < 1:
                agg_successes = 10
                agg_failures = min(int((2 - af) * 20), 50)
            else:
                agg_successes = int(af * 15)
                agg_failures = int((2 - af) * 15)

            beliefs.add_belief(
                "is aggressive",
                Opinion.from_frequency(agg_successes, agg_failures, base_rate=0.4),
                BeliefCategory.AGGRESSION
            )

        # Bluff frequency beliefs
        total_showed = self.stats.showed_bluff + self.stats.showed_value
        if total_showed > 0:
            beliefs.add_belief(
                "bluffs often (high frequency)",
                Opinion.from_frequency(
                    self.stats.showed_bluff,
                    self.stats.showed_value,
                    base_rate=0.3
                ),
                BeliefCategory.BLUFF_FREQUENCY
            )

        # Fold to c-bet belief
        if self.stats.faced_cbet > 0:
            beliefs.add_belief(
                "folds to c-bet often",
                Opinion.from_frequency(
                    self.stats.fold_to_cbet,
                    self.stats.faced_cbet - self.stats.fold_to_cbet,
                    base_rate=0.45
                ),
                BeliefCategory.TENDENCY
            )

        # Fold to 3-bet belief
        if self.stats.faced_3bet > 0:
            beliefs.add_belief(
                "folds to 3-bet often",
                Opinion.from_frequency(
                    self.stats.fold_to_3bet,
                    self.stats.faced_3bet - self.stats.fold_to_3bet,
                    base_rate=0.55
                ),
                BeliefCategory.TENDENCY
            )

        return beliefs

    def _infer_player_type(self) -> MultinomialOpinion:
        """
        Infer player type from statistics.

        Uses VPIP/PFR matrix and aggression factor:
        - NIT: Low VPIP (<15%), low PFR
        - TAG: Low-medium VPIP (15-25%), high PFR relative to VPIP
        - LAG: High VPIP (>30%), high PFR and aggression
        - Fish: High VPIP, low PFR (calling station)
        - Maniac: Very high VPIP and extreme aggression
        """
        if self.stats.total_hands < 5:
            # Not enough data - return vacuous
            return MultinomialOpinion.vacuous(PLAYER_TYPES, DEFAULT_PLAYER_TYPE_BASE_RATES)

        vpip = self.stats.vpip
        pfr = self.stats.pfr
        af = self.stats.aggression_factor

        # Calculate evidence for each type
        evidence: dict[str, float] = {t: 0.0 for t in PLAYER_TYPES}

        # NIT: Very tight
        if vpip < 0.02:
            # Extremely tight (nearly 0% VPIP) - almost certainly NIT
            evidence["NIT"] += 0.85
        elif vpip < 0.10:
            # Very tight
            evidence["NIT"] += 0.6
        elif vpip < 0.15:
            evidence["NIT"] += 0.4
        elif vpip < 0.20:
            evidence["NIT"] += 0.2

        # TAG: Tight but aggressive
        if 0.15 <= vpip <= 0.28 and pfr > vpip * 0.7:
            evidence["TAG"] += 0.3
            if af > 2:
                evidence["TAG"] += 0.2

        # LAG: Loose and aggressive
        if vpip > 0.28 and pfr > 0.20:
            evidence["LAG"] += 0.3
            if af > 2.5:
                evidence["LAG"] += 0.2

        # Fish: Loose and passive (high VPIP, low PFR, low AF)
        if vpip > 0.30 and pfr < vpip * 0.5:
            evidence["Fish"] += 0.3
            if af < 1.5:
                evidence["Fish"] += 0.2

        # Maniac: Extreme aggression
        if vpip > 0.40 and af > 3:
            evidence["Maniac"] += 0.4

        # Normalize evidence and create opinion
        total_evidence = sum(evidence.values())
        if total_evidence == 0:
            return MultinomialOpinion.vacuous(PLAYER_TYPES, DEFAULT_PLAYER_TYPE_BASE_RATES)

        # Scale by sample size (more hands = more certainty)
        sample_factor = min(1.0, self.stats.total_hands / 100)

        # Create beliefs from evidence
        beliefs = {t: evidence[t] * sample_factor * 0.7 for t in PLAYER_TYPES}
        uncertainty = 1.0 - sum(beliefs.values())

        return MultinomialOpinion(beliefs, uncertainty, DEFAULT_PLAYER_TYPE_BASE_RATES)

    def get_summary(self) -> str:
        """Generate natural language summary for LLM context."""
        stats = self.stats
        quality = stats.get_sample_quality()

        lines = [
            f"Opponent: {self.opponent_id} ({stats.total_hands} hands, {quality} sample)",
            f"  VPIP: {stats.vpip:.0%} | PFR: {stats.pfr:.0%}",
        ]

        if stats.three_bet_opportunities > 0:
            lines.append(f"  3-bet: {stats.three_bet_pct:.0%}")

        if stats.faced_3bet > 0:
            lines.append(f"  Fold to 3-bet: {stats.fold_to_3bet_pct:.0%}")

        if stats.cbet_opportunities > 0:
            lines.append(f"  C-bet: {stats.cbet_pct:.0%}")

        if stats.faced_cbet > 0:
            lines.append(f"  Fold to C-bet: {stats.fold_to_cbet_pct:.0%}")

        af = stats.aggression_factor
        if af != float('inf'):
            lines.append(f"  Aggression Factor: {af:.1f}")

        if stats.went_to_showdown > 0:
            lines.append(f"  WTSD: {stats.wtsd:.0%} | W$SD: {stats.won_at_sd:.0%}")

        # Player type inference
        beliefs = self.to_beliefs()
        best_type, prob = beliefs.get_most_likely_player_type()
        lines.append(f"  Likely type: {best_type} ({prob:.0%})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "opponent_id": self.opponent_id,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "stats": {
                "vpip_hands": self.stats.vpip_hands,
                "vpip_opportunities": self.stats.vpip_opportunities,
                "pfr_hands": self.stats.pfr_hands,
                "pfr_opportunities": self.stats.pfr_opportunities,
                "three_bet_made": self.stats.three_bet_made,
                "three_bet_opportunities": self.stats.three_bet_opportunities,
                "fold_to_3bet": self.stats.fold_to_3bet,
                "faced_3bet": self.stats.faced_3bet,
                "cbet_made": self.stats.cbet_made,
                "cbet_opportunities": self.stats.cbet_opportunities,
                "fold_to_cbet": self.stats.fold_to_cbet,
                "faced_cbet": self.stats.faced_cbet,
                "postflop_bets": self.stats.postflop_bets,
                "postflop_raises": self.stats.postflop_raises,
                "postflop_calls": self.stats.postflop_calls,
                "postflop_folds": self.stats.postflop_folds,
                "went_to_showdown": self.stats.went_to_showdown,
                "could_showdown": self.stats.could_showdown,
                "won_at_showdown": self.stats.won_at_showdown,
                "showed_bluff": self.stats.showed_bluff,
                "showed_value": self.stats.showed_value,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpponentMemory":
        """Deserialize from dictionary."""
        memory = cls(data["opponent_id"])

        if data.get("first_seen"):
            memory.first_seen = datetime.fromisoformat(data["first_seen"])
        if data.get("last_seen"):
            memory.last_seen = datetime.fromisoformat(data["last_seen"])

        stats_data = data.get("stats", {})
        for key, value in stats_data.items():
            if hasattr(memory.stats, key):
                setattr(memory.stats, key, value)

        return memory


class TableMemory:
    """Memory for all opponents at a table."""

    def __init__(self):
        self.opponents: dict[str, OpponentMemory] = {}

    def get_opponent(self, opponent_id: str) -> OpponentMemory:
        """Get or create memory for an opponent."""
        if opponent_id not in self.opponents:
            self.opponents[opponent_id] = OpponentMemory(opponent_id)
        return self.opponents[opponent_id]

    def get_all_summaries(self) -> str:
        """Get summaries for all opponents."""
        if not self.opponents:
            return "No opponent data available."

        lines = ["=== Opponent Profiles ==="]
        for memory in sorted(self.opponents.values(),
                           key=lambda m: m.stats.total_hands,
                           reverse=True):
            lines.append("")
            lines.append(memory.get_summary())

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "opponents": {
                oid: mem.to_dict() for oid, mem in self.opponents.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TableMemory":
        """Deserialize from dictionary."""
        memory = cls()
        for oid, mem_data in data.get("opponents", {}).items():
            memory.opponents[oid] = OpponentMemory.from_dict(mem_data)
        return memory
