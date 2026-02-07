"""
Belief state management and revision via observation-consistency mapping.

Observations don't directly update beliefs. Instead, each observation O_i
is mapped to each belief B_j through a consistency opinion — an SL opinion
about the proposition "O_i is consistent with B_j".

Flow:
  Observation O_i  →  ω_consistency(O_i, B_j)  →  SL Fusion  →  B_j (revised)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
from enum import Enum, auto

from .subjective_logic import (
    Opinion, Belief, sort_beliefs_by_knowledge,
    MultinomialOpinion, MultinomialBelief
)
from .primitives import Action, ActionType, Position, Street, HoleCards, Card
from .game_state import GameState
from ..logging_config import get_logger

logger = get_logger(__name__)


class ObservationType(Enum):
    """Types of observable events in a poker game."""
    ACTION = auto()      # Bet, raise, call, check, fold
    SHOWDOWN = auto()    # Cards revealed at showdown
    TIMING = auto()      # Time taken to act
    SIZING = auto()      # Bet size relative to pot
    SEQUENCE = auto()    # Pattern of actions over streets


@dataclass
class GameContext:
    """Context in which an observation occurred."""
    street: Street
    position: Position
    pot_size: float
    to_call: float
    num_players: int
    is_heads_up: bool
    facing_raise: bool = False
    raise_count: int = 0
    spr: float = 0.0


@dataclass
class Observation:
    """
    A single observed event in the game.

    Observations are the raw inputs to belief revision.
    """
    obs_id: str
    obs_type: ObservationType
    player_id: str
    action: Optional[Action] = None
    revealed_cards: Optional[HoleCards] = None
    context: Optional[GameContext] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Derived metrics
    bet_size_bb: Optional[float] = None
    pot_fraction: Optional[float] = None
    time_taken_ms: Optional[int] = None

    def __str__(self) -> str:
        if self.obs_type == ObservationType.ACTION and self.action:
            return f"{self.player_id}@{self.context.position}: {self.action}"
        elif self.obs_type == ObservationType.SHOWDOWN and self.revealed_cards:
            return f"{self.player_id} shows {self.revealed_cards}"
        return f"{self.obs_type.name}: {self.player_id}"


@dataclass
class ConsistencyOpinion:
    """
    Opinion about: "Observation O is consistent with Belief B"

    This is the bridge between what we see and what we believe.
    High b = observation strongly supports the belief
    High d = observation strongly contradicts the belief
    High u = observation is ambiguous w.r.t. the belief
    """
    observation: Observation
    belief_label: str
    opinion: Opinion
    reasoning: str = ""

    def __str__(self) -> str:
        return f"P({self.observation} ~ {self.belief_label}) = {self.opinion}"


# ============== Belief Categories ==============

class BeliefCategory:
    """Standard belief categories for opponent modeling."""
    PLAYER_TYPE = "player_type"
    AGGRESSION = "aggression"
    BLUFF_FREQUENCY = "bluff_frequency"
    POSITIONAL_AWARENESS = "positional_awareness"
    HAND_RANGE = "hand_range"
    TENDENCY = "tendency"


# Player type categories (mutually exclusive - use multinomial SL)
PLAYER_TYPES = ["TAG", "LAG", "NIT", "Fish", "Maniac"]

# Default base rates for player types (sum to 1.0)
DEFAULT_PLAYER_TYPE_BASE_RATES = {
    "TAG": 0.20,     # Tight-Aggressive (competent regulars)
    "LAG": 0.15,     # Loose-Aggressive (sophisticated players)
    "NIT": 0.10,     # Very tight (risk-averse)
    "Fish": 0.45,    # Recreational (most common in low stakes)
    "Maniac": 0.10,  # Hyper-aggressive (rare)
}


# ============== Observation-Belief Mapper ==============

class ObservationBeliefMapper:
    """
    Map observations to consistency opinions for each belief.

    Key insight: An observation may be:
    - Strongly consistent (high b) with some beliefs
    - Strongly inconsistent (high d) with others
    - Ambiguous (high u) for most

    This is more principled than direct belief updates.
    """

    def compute_consistency(self, obs: Observation,
                           belief: Belief) -> ConsistencyOpinion:
        """
        Compute how consistent observation is with a belief.

        Returns opinion about "obs is consistent with belief.label"
        """
        if obs.obs_type == ObservationType.ACTION:
            return self._action_consistency(obs, belief)
        elif obs.obs_type == ObservationType.SHOWDOWN:
            return self._showdown_consistency(obs, belief)
        elif obs.obs_type == ObservationType.TIMING:
            return self._timing_consistency(obs, belief)
        elif obs.obs_type == ObservationType.SIZING:
            return self._sizing_consistency(obs, belief)
        else:
            # Unknown - maximally uncertain
            return ConsistencyOpinion(
                observation=obs,
                belief_label=belief.label,
                opinion=Opinion.vacuous(),
                reasoning="Unknown observation type"
            )

    def _action_consistency(self, obs: Observation,
                           belief: Belief) -> ConsistencyOpinion:
        """How consistent is this action with the belief?"""
        action = obs.action
        ctx = obs.context

        # Default to uncertain
        opinion = Opinion.vacuous()
        reasoning = "No strong signal"

        if belief.category == BeliefCategory.AGGRESSION:
            opinion, reasoning = self._aggression_consistency(action, ctx, belief)
        elif belief.category == BeliefCategory.PLAYER_TYPE:
            opinion, reasoning = self._player_type_consistency(action, ctx, belief)
        elif belief.category == BeliefCategory.BLUFF_FREQUENCY:
            opinion, reasoning = self._bluff_freq_consistency(action, ctx, belief)
        elif belief.category == BeliefCategory.POSITIONAL_AWARENESS:
            opinion, reasoning = self._positional_consistency(action, ctx, belief)

        return ConsistencyOpinion(
            observation=obs,
            belief_label=belief.label,
            opinion=opinion,
            reasoning=reasoning
        )

    def _aggression_consistency(self, action: Action, ctx: GameContext,
                                belief: Belief) -> tuple[Opinion, str]:
        """Consistency with aggression-related beliefs."""
        is_aggressive_action = action.action_type.is_aggressive
        is_passive_action = action.action_type.is_passive

        if "aggressive" in belief.label.lower() or "LAG" in belief.label:
            if is_aggressive_action:
                # 3-bet/raise is consistent with aggressive
                if ctx.facing_raise:
                    return (Opinion(0.7, 0.1, 0.2, 0.5),
                            "Aggression consistent with aggressive profile")
                else:
                    return (Opinion(0.5, 0.1, 0.4, 0.5),
                            "Bet consistent with aggressive profile")
            elif action.action_type == ActionType.FOLD:
                return (Opinion(0.1, 0.4, 0.5, 0.5),
                        "Fold mildly inconsistent with aggressive profile")
            elif is_passive_action:
                return (Opinion(0.2, 0.3, 0.5, 0.5),
                        "Passive play somewhat inconsistent with aggressive")

        elif "passive" in belief.label.lower() or "calling station" in belief.label.lower():
            if action.action_type == ActionType.CALL:
                return (Opinion(0.6, 0.1, 0.3, 0.5),
                        "Call consistent with passive profile")
            elif is_aggressive_action:
                return (Opinion(0.1, 0.5, 0.4, 0.5),
                        "Aggression inconsistent with passive profile")

        return (Opinion.vacuous(), "No clear signal for aggression belief")

    def _player_type_consistency(self, action: Action, ctx: GameContext,
                                 belief: Belief) -> tuple[Opinion, str]:
        """Consistency with player type beliefs (legacy binomial - for backwards compat)."""
        label_lower = belief.label.lower()

        # Nit/tight beliefs
        if "nit" in label_lower or "tight" in label_lower:
            if action.action_type == ActionType.FOLD:
                return (Opinion(0.4, 0.1, 0.5, 0.3),
                        "Fold consistent with tight style")
            elif action.action_type.is_aggressive and ctx.street == Street.PREFLOP:
                return (Opinion(0.3, 0.2, 0.5, 0.3),
                        "Preflop raise from tight player - ambiguous")

        # LAG beliefs
        if "lag" in label_lower:
            if action.action_type.is_aggressive:
                return (Opinion(0.6, 0.1, 0.3, 0.4),
                        "Aggression consistent with LAG")
            elif action.action_type == ActionType.FOLD:
                return (Opinion(0.2, 0.4, 0.4, 0.4),
                        "Fold somewhat inconsistent with LAG")

        # TAG beliefs
        if "tag" in label_lower:
            if action.action_type.is_aggressive and ctx.street == Street.PREFLOP:
                return (Opinion(0.5, 0.1, 0.4, 0.4),
                        "Preflop aggression consistent with TAG")

        # Fish/recreational beliefs
        if "fish" in label_lower or "recreational" in label_lower:
            if action.action_type == ActionType.CALL:
                return (Opinion(0.5, 0.1, 0.4, 0.4),
                        "Calling consistent with recreational player")

        return (Opinion.vacuous(), "No clear signal for player type")

    def compute_player_type_evidence(self, obs: Observation) -> Optional[tuple[str, float, str]]:
        """
        Compute evidence for multinomial player type from an observation.

        Returns:
            Optional tuple of (player_type, evidence_strength, reasoning)
            Returns None if observation provides no player type signal.
        """
        if obs.obs_type == ObservationType.ACTION:
            return self._action_to_player_type(obs.action, obs.context)
        elif obs.obs_type == ObservationType.SHOWDOWN:
            return self._showdown_to_player_type(obs.revealed_cards, obs)
        return None

    def _action_to_player_type(self, action: Action,
                               ctx: Optional[GameContext]) -> Optional[tuple[str, float, str]]:
        """Map action to player type evidence."""
        if action is None or ctx is None:
            return None

        # Preflop actions are most indicative of player type
        if ctx.street == Street.PREFLOP:
            if action.action_type == ActionType.FOLD:
                # Folding preflop suggests tight (NIT or TAG)
                return ("NIT", 0.15, "Preflop fold suggests tight style")

            elif action.action_type == ActionType.CALL:
                # Limping/calling preflop suggests Fish
                return ("Fish", 0.20, "Preflop limp/call suggests recreational")

            elif action.action_type.is_aggressive:
                if ctx.facing_raise:
                    # 3-betting suggests LAG or TAG
                    if ctx.raise_count >= 2:
                        # 4-bet+ suggests LAG or Maniac
                        return ("LAG", 0.25, "4-bet+ suggests loose-aggressive or maniac")
                    else:
                        # 3-bet is somewhat balanced between TAG and LAG
                        return ("TAG", 0.10, "3-bet mildly suggests TAG (could be LAG too)")
                else:
                    # Open raise - balanced action, weak signal
                    return None

        # Postflop actions
        else:
            if action.action_type.is_aggressive:
                # Postflop aggression slightly suggests LAG
                return ("LAG", 0.10, "Postflop aggression mildly suggests LAG")
            elif action.action_type == ActionType.CALL:
                # Postflop calling suggests Fish (calling station behavior)
                return ("Fish", 0.15, "Postflop calling suggests recreational")
            elif action.action_type == ActionType.FOLD:
                # Postflop fold is weak signal for NIT
                return ("NIT", 0.08, "Postflop fold weakly suggests tight style")

        return None

    def _showdown_to_player_type(self, cards: Optional[HoleCards],
                                  obs: Observation) -> Optional[tuple[str, float, str]]:
        """Infer player type from showdown cards."""
        if cards is None:
            return None

        is_premium = cards.notation in ('AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs')
        is_trash = cards.card1.rank.value <= 7 and cards.card2.rank.value <= 7 and not cards.is_pair

        if is_trash:
            # Showing trash suggests LAG, Maniac, or Fish
            # If they were aggressive, more likely LAG/Maniac
            # If they were passive, more likely Fish
            return ("LAG", 0.30, f"Showing {cards.notation} (weak hand) suggests LAG/Maniac")

        elif is_premium:
            # Showing premium could be TAG or NIT
            return ("TAG", 0.15, f"Showing {cards.notation} (premium) suggests TAG or NIT")

        return None

    def _bluff_freq_consistency(self, action: Action, ctx: GameContext,
                                belief: Belief) -> tuple[Opinion, str]:
        """Consistency with bluff frequency beliefs."""
        label_lower = belief.label.lower()

        if "high bluff" in label_lower or "bluffs often" in label_lower:
            # Aggression on scare cards, draws, etc. consistent with bluffing
            if action.action_type.is_aggressive:
                return (Opinion(0.4, 0.1, 0.5, 0.3),
                        "Aggression mildly consistent with high bluff freq")

        if "rarely bluffs" in label_lower or "low bluff" in label_lower:
            if action.action_type.is_aggressive:
                return (Opinion(0.2, 0.3, 0.5, 0.3),
                        "Aggression mildly inconsistent with low bluff freq")

        return (Opinion.vacuous(), "No clear bluff frequency signal")

    def _positional_consistency(self, action: Action, ctx: GameContext,
                                belief: Belief) -> tuple[Opinion, str]:
        """Consistency with positional awareness beliefs."""
        label_lower = belief.label.lower()

        if "positionally aware" in label_lower:
            # Late position aggression = positionally aware
            if ctx.position in (Position.CO, Position.BTN) and action.action_type.is_aggressive:
                return (Opinion(0.5, 0.1, 0.4, 0.4),
                        "LP aggression consistent with positional awareness")
            # Early position tightness = positionally aware
            if ctx.position in (Position.UTG, Position.HJ) and action.action_type == ActionType.FOLD:
                return (Opinion(0.4, 0.1, 0.5, 0.4),
                        "EP fold consistent with positional awareness")

        return (Opinion.vacuous(), "No clear positional signal")

    def _showdown_consistency(self, obs: Observation,
                             belief: Belief) -> ConsistencyOpinion:
        """How consistent are revealed cards with the belief?"""
        cards = obs.revealed_cards
        if cards is None:
            return ConsistencyOpinion(
                observation=obs,
                belief_label=belief.label,
                opinion=Opinion.vacuous(),
                reasoning="No cards revealed"
            )

        opinion = Opinion.vacuous()
        reasoning = "No strong signal from showdown"

        # Check for specific hand ranges
        if belief.category == BeliefCategory.HAND_RANGE:
            opinion, reasoning = self._range_consistency(cards, belief)

        # Player type inferences from showdown
        elif belief.category == BeliefCategory.PLAYER_TYPE:
            opinion, reasoning = self._type_from_showdown(cards, obs, belief)

        elif belief.category == BeliefCategory.BLUFF_FREQUENCY:
            opinion, reasoning = self._bluff_from_showdown(cards, obs, belief)

        return ConsistencyOpinion(
            observation=obs,
            belief_label=belief.label,
            opinion=opinion,
            reasoning=reasoning
        )

    def _range_consistency(self, cards: HoleCards,
                          belief: Belief) -> tuple[Opinion, str]:
        """Is this hand consistent with a range belief?"""
        # Check hand strength characteristics
        is_premium = cards.notation in ('AA', 'KK', 'QQ', 'AKs', 'AKo')
        is_trash = cards.card1.rank.value <= 7 and cards.card2.rank.value <= 7 and not cards.is_pair

        if "premium" in belief.label.lower() or "strong range" in belief.label.lower():
            if is_premium:
                return (Opinion(0.8, 0.05, 0.15, 0.3),
                        f"Showdown of {cards.notation} strongly supports premium range")
            elif is_trash:
                return (Opinion(0.05, 0.7, 0.25, 0.3),
                        f"Showdown of {cards.notation} contradicts premium range")

        if "wide range" in belief.label.lower():
            # Wide range is consistent with almost anything
            return (Opinion(0.4, 0.1, 0.5, 0.5),
                    "Most hands consistent with wide range")

        return (Opinion.vacuous(), "No specific range inference")

    def _type_from_showdown(self, cards: HoleCards, obs: Observation,
                            belief: Belief) -> tuple[Opinion, str]:
        """Infer player type from showdown."""
        is_trash = cards.card1.rank.value <= 7 and cards.card2.rank.value <= 7 and not cards.is_pair
        label_lower = belief.label.lower()

        # Showing trash after aggression = LAG or maniac
        if is_trash:
            if "lag" in label_lower or "maniac" in label_lower:
                return (Opinion(0.6, 0.1, 0.3, 0.3),
                        f"Showing {cards.notation} supports LAG/maniac profile")
            if "nit" in label_lower or "tight" in label_lower:
                return (Opinion(0.05, 0.75, 0.2, 0.3),
                        f"Showing {cards.notation} strongly contradicts tight profile")

        return (Opinion.vacuous(), "No clear type inference from showdown")

    def _bluff_from_showdown(self, cards: HoleCards, obs: Observation,
                             belief: Belief) -> tuple[Opinion, str]:
        """Infer bluff frequency from showdown."""
        # Would need action context to know if this was a bluff
        # For now, use hand strength as proxy
        is_trash = cards.card1.rank.value <= 7 and cards.card2.rank.value <= 7 and not cards.is_pair
        label_lower = belief.label.lower()

        if is_trash and ("high bluff" in label_lower or "bluffs often" in label_lower):
            return (Opinion(0.6, 0.1, 0.3, 0.3),
                    f"Showing {cards.notation} (trash) supports high bluff freq")

        if is_trash and ("rarely bluffs" in label_lower or "low bluff" in label_lower):
            return (Opinion(0.1, 0.6, 0.3, 0.3),
                    f"Showing {cards.notation} contradicts low bluff freq")

        return (Opinion.vacuous(), "No clear bluff inference from showdown")

    def _timing_consistency(self, obs: Observation,
                           belief: Belief) -> ConsistencyOpinion:
        """How consistent is timing with the belief?"""
        # Timing tells are weak but can add small evidence
        time_ms = obs.time_taken_ms or 0

        opinion = Opinion.vacuous()
        reasoning = "Timing provides weak evidence"

        # Quick actions often indicate strength or auto-play
        # Long tanks often indicate tough decisions
        if time_ms < 1000:  # Very fast
            if "recreational" in belief.label.lower():
                opinion = Opinion(0.3, 0.2, 0.5, 0.5)
                reasoning = "Fast action mildly consistent with auto-play tendency"

        elif time_ms > 15000:  # Long tank
            if "thinking player" in belief.label.lower():
                opinion = Opinion(0.4, 0.1, 0.5, 0.5)
                reasoning = "Long think consistent with thoughtful player"

        return ConsistencyOpinion(
            observation=obs,
            belief_label=belief.label,
            opinion=opinion,
            reasoning=reasoning
        )

    def _sizing_consistency(self, obs: Observation,
                           belief: Belief) -> ConsistencyOpinion:
        """How consistent is bet sizing with the belief?"""
        pot_frac = obs.pot_fraction or 0

        opinion = Opinion.vacuous()
        reasoning = "No strong sizing signal"

        # Oversized bets
        if pot_frac > 1.5:
            if "polarized" in belief.label.lower():
                opinion = Opinion(0.5, 0.1, 0.4, 0.4)
                reasoning = "Oversized bet consistent with polarized strategy"

        # Small bets
        elif pot_frac < 0.33:
            if "merged" in belief.label.lower() or "thin value" in belief.label.lower():
                opinion = Opinion(0.4, 0.2, 0.4, 0.4)
                reasoning = "Small bet consistent with merged/thin value"

        return ConsistencyOpinion(
            observation=obs,
            belief_label=belief.label,
            opinion=opinion,
            reasoning=reasoning
        )


# ============== Belief Reviser ==============

class SLBeliefReviser:
    """
    Revise beliefs using subjective logic operators.

    Uses cumulative fusion to combine consistency opinions with
    current beliefs, gradually reducing uncertainty as evidence
    accumulates.
    """

    def __init__(self, trust_discount: float = 1.0):
        """
        Args:
            trust_discount: Global discount factor for observations (0-1)
                           Lower = more conservative updates
        """
        self.trust_discount = trust_discount

    def revise(self, current: Opinion, consistency: ConsistencyOpinion) -> Opinion:
        """
        Revise a belief based on an observation-consistency opinion.

        Args:
            current: Current belief opinion
            consistency: Consistency opinion from observation

        Returns:
            Revised belief opinion
        """
        # Apply trust discounting to consistency opinion
        discounted = consistency.opinion.discount_by_factor(self.trust_discount)

        # Cumulative fusion to combine evidence
        revised = current.cumulative_fusion(discounted)

        return revised

    def revise_multiple(self, current: Opinion,
                       consistencies: list[ConsistencyOpinion]) -> Opinion:
        """
        Revise belief based on multiple observations.

        Fuses all consistency opinions sequentially.
        """
        result = current
        for consistency in consistencies:
            result = self.revise(result, consistency)
        return result


# ============== Belief State ==============

@dataclass
class VillainBeliefs:
    """
    Beliefs about a specific opponent.

    Uses multinomial SL for player type (mutually exclusive categories)
    and binomial SL for tendencies (non-mutually exclusive).
    """
    player_id: str
    beliefs: dict[str, Belief] = field(default_factory=dict)
    player_type: Optional[MultinomialOpinion] = None  # Multinomial for mutually exclusive types

    def __post_init__(self):
        """Initialize player type with vacuous opinion if not set."""
        if self.player_type is None:
            self.player_type = MultinomialOpinion.vacuous(
                PLAYER_TYPES, DEFAULT_PLAYER_TYPE_BASE_RATES
            )

    def add_belief(self, label: str, opinion: Opinion,
                   category: str = "general") -> None:
        """Add or update a binomial belief (for tendencies)."""
        self.beliefs[label] = Belief(label, opinion, category)

    def get_belief(self, label: str) -> Optional[Belief]:
        """Get a specific binomial belief."""
        return self.beliefs.get(label)

    def update_player_type(self, evidence_type: str, strength: float = 0.2) -> None:
        """
        Update player type belief based on evidence.

        Args:
            evidence_type: Player type suggested by observation (TAG, LAG, NIT, Fish, Maniac)
            strength: Strength of evidence (0 to 1)
        """
        if evidence_type not in PLAYER_TYPES:
            raise ValueError(f"Unknown player type: {evidence_type}. Must be one of {PLAYER_TYPES}")
        self.player_type = self.player_type.update_from_evidence(evidence_type, strength)

    def get_player_type_distribution(self) -> dict[str, float]:
        """Get projected probability distribution over player types."""
        return self.player_type.all_projected_probabilities()

    def get_most_likely_player_type(self) -> tuple[str, float]:
        """Get most likely player type and its probability."""
        return self.player_type.most_likely_category()

    def all_beliefs(self) -> list[Belief]:
        """Get all binomial beliefs as a list."""
        return list(self.beliefs.values())

    def beliefs_by_category(self, category: str) -> list[Belief]:
        """Get binomial beliefs in a specific category."""
        return [b for b in self.beliefs.values() if b.category == category]

    def sorted_by_knowledge(self) -> list[Belief]:
        """Get binomial beliefs sorted by knowledge (most informative first)."""
        return sort_beliefs_by_knowledge(self.all_beliefs())

    def to_agent_format(self) -> str:
        """Format beliefs using Jøsang verbal mapping."""
        from src.core.subjective_logic import verbal_multinomial
        lines = [f"Beliefs about {self.player_id}:"]

        # Player type (multinomial) - verbal with percentages and confidence
        lines.append(f"  Type: {verbal_multinomial(self.player_type)}")

        # Tendencies (binomial) - verbal likelihood + confidence
        if self.beliefs:
            lines.append("  Tendencies:")
            for belief in self.sorted_by_knowledge():
                lines.append(f"    {belief.to_agent_format()}")

        return "\n".join(lines)


@dataclass
class BeliefState:
    """
    Complete belief state for the agent.

    Contains beliefs about all villains and situational beliefs.
    """
    villain_beliefs: dict[str, VillainBeliefs] = field(default_factory=dict)
    situational_beliefs: dict[str, Belief] = field(default_factory=dict)

    def get_or_create_villain(self, player_id: str) -> VillainBeliefs:
        """Get beliefs for a villain, creating if needed."""
        if player_id not in self.villain_beliefs:
            self.villain_beliefs[player_id] = VillainBeliefs(player_id)
        return self.villain_beliefs[player_id]

    def add_situational_belief(self, label: str, opinion: Opinion,
                               category: str = "situation") -> None:
        """Add a situational belief (not tied to specific player)."""
        self.situational_beliefs[label] = Belief(label, opinion, category)

    def all_beliefs(self) -> list[Belief]:
        """Get all beliefs (villain + situational)."""
        beliefs = []
        for vb in self.villain_beliefs.values():
            beliefs.extend(vb.all_beliefs())
        beliefs.extend(self.situational_beliefs.values())
        return beliefs

    def to_agent_format(self, active_player_ids: set[str] | None = None) -> str:
        """Format complete belief state for agent consumption.

        Args:
            active_player_ids: If provided, only include beliefs for these players.
                Eliminated players are excluded to save prompt tokens.
        """
        lines = ["=== OPPONENT READS ==="]
        lines.append("")

        # Villain beliefs (only active players)
        for villain_beliefs in self.villain_beliefs.values():
            if active_player_ids and villain_beliefs.player_id not in active_player_ids:
                continue
            lines.append(villain_beliefs.to_agent_format())
            lines.append("")

        # Situational beliefs
        if self.situational_beliefs:
            lines.append("Situational beliefs:")
            sorted_sit = sort_beliefs_by_knowledge(list(self.situational_beliefs.values()))
            for belief in sorted_sit:
                lines.append(f"  {belief.to_agent_format()}")

        return "\n".join(lines)


# ============== Belief Revision Engine ==============

class BeliefRevisionEngine:
    """
    Complete belief revision pipeline.

    1. Receives observations
    2. Maps to consistency opinions for each belief
    3. Revises beliefs using SL fusion
    4. Maintains updated belief state
    """

    def __init__(self, trust_discount: float = 0.9):
        self.mapper = ObservationBeliefMapper()
        self.reviser = SLBeliefReviser(trust_discount)
        self.observation_history: list[Observation] = []
        self._processed_action_count: int = 0  # Track how many actions already processed

    def process_observation(self, obs: Observation,
                           belief_state: BeliefState) -> BeliefState:
        """
        Process a single observation and update beliefs.

        Updates both multinomial player type and binomial tendency beliefs.

        Args:
            obs: The observation to process
            belief_state: Current belief state

        Returns:
            Updated belief state
        """
        logger.debug("Processing observation: %s %s from %s",
                      obs.obs_type.value, obs.action, obs.player_id)
        self.observation_history.append(obs)

        # Get beliefs for this player
        if obs.player_id:
            villain_beliefs = belief_state.get_or_create_villain(obs.player_id)

            # Update multinomial player type
            player_type_evidence = self.mapper.compute_player_type_evidence(obs)
            if player_type_evidence is not None:
                evidence_type, strength, _ = player_type_evidence
                # Apply trust discount to evidence strength
                discounted_strength = strength * self.reviser.trust_discount
                villain_beliefs.update_player_type(evidence_type, discounted_strength)

            # Revise each binomial belief based on observation consistency
            for label, belief in list(villain_beliefs.beliefs.items()):
                consistency = self.mapper.compute_consistency(obs, belief)

                # Skip if vacuous (no information)
                if consistency.opinion.is_vacuous:
                    continue

                # Revise belief
                new_opinion = self.reviser.revise(belief.opinion, consistency)
                villain_beliefs.beliefs[label] = Belief(
                    label, new_opinion, belief.category
                )

        return belief_state

    def process_game_state(self, game_state: GameState,
                          belief_state: BeliefState) -> BeliefState:
        """
        Extract observations from game state and process only new actions.
        """
        # Only process actions we haven't seen yet
        all_actions = game_state.action_history
        new_actions = all_actions[self._processed_action_count:]

        for action_record in new_actions:
            if action_record.player_id == game_state.hero_id:
                self._processed_action_count += 1
                continue  # Skip hero's actions

            obs = Observation(
                obs_id=f"{game_state.hand_id}_{len(self.observation_history)}",
                obs_type=ObservationType.ACTION,
                player_id=action_record.player_id,
                action=action_record.action,
                context=GameContext(
                    street=action_record.street,
                    position=action_record.position,
                    pot_size=action_record.pot_after,
                    to_call=0,  # Would need more context
                    num_players=game_state.num_in_hand,
                    is_heads_up=game_state.is_heads_up,
                )
            )
            belief_state = self.process_observation(obs, belief_state)
            self._processed_action_count += 1

        return belief_state

    def reset_action_tracking(self) -> None:
        """Reset action tracking for a new hand."""
        self._processed_action_count = 0


def create_default_villain_beliefs(player_id: str) -> VillainBeliefs:
    """
    Create default (uncertain) beliefs for a new villain.

    Uses multinomial SL for player type (mutually exclusive: TAG, LAG, NIT, Fish, Maniac)
    and binomial SL for tendencies (not mutually exclusive).
    """
    # Player type is initialized with vacuous multinomial in VillainBeliefs.__post_init__
    beliefs = VillainBeliefs(player_id)

    # Tendencies (binomial) - these are NOT mutually exclusive
    # A TAG can be aggressive, a LAG can be positionally aware, etc.

    # Aggression tendencies
    beliefs.add_belief("is aggressive",
                       Opinion.vacuous(base_rate=0.4),
                       BeliefCategory.AGGRESSION)
    beliefs.add_belief("is passive",
                       Opinion.vacuous(base_rate=0.4),
                       BeliefCategory.AGGRESSION)

    # Bluff frequency tendencies
    beliefs.add_belief("bluffs often (high frequency)",
                       Opinion.vacuous(base_rate=0.3),
                       BeliefCategory.BLUFF_FREQUENCY)
    beliefs.add_belief("rarely bluffs (low frequency)",
                       Opinion.vacuous(base_rate=0.3),
                       BeliefCategory.BLUFF_FREQUENCY)

    # Positional awareness
    beliefs.add_belief("is positionally aware",
                       Opinion.vacuous(base_rate=0.4),
                       BeliefCategory.POSITIONAL_AWARENESS)

    return beliefs
