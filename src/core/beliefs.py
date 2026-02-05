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

from .subjective_logic import Opinion, Belief, sort_beliefs_by_knowledge
from .primitives import Action, ActionType, Position, Street, HoleCards, Card
from .game_state import GameState


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
        """Consistency with player type beliefs (TAG, LAG, NIT, etc.)."""
        label_lower = belief.label.lower()

        # Nit/tight beliefs
        if "nit" in label_lower or "tight" in label_lower:
            if action.action_type == ActionType.FOLD:
                return (Opinion(0.4, 0.1, 0.5, 0.3),
                        "Fold consistent with tight style")
            elif action.action_type.is_aggressive and ctx.street == Street.PREFLOP:
                # Tight player raising preflop = strong hand
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
    """Beliefs about a specific opponent."""
    player_id: str
    beliefs: dict[str, Belief] = field(default_factory=dict)

    def add_belief(self, label: str, opinion: Opinion,
                   category: str = "general") -> None:
        """Add or update a belief."""
        self.beliefs[label] = Belief(label, opinion, category)

    def get_belief(self, label: str) -> Optional[Belief]:
        """Get a specific belief."""
        return self.beliefs.get(label)

    def all_beliefs(self) -> list[Belief]:
        """Get all beliefs as a list."""
        return list(self.beliefs.values())

    def beliefs_by_category(self, category: str) -> list[Belief]:
        """Get beliefs in a specific category."""
        return [b for b in self.beliefs.values() if b.category == category]

    def sorted_by_knowledge(self) -> list[Belief]:
        """Get beliefs sorted by knowledge (most informative first)."""
        return sort_beliefs_by_knowledge(self.all_beliefs())

    def to_agent_format(self) -> str:
        """Format beliefs for agent consumption."""
        lines = [f"Beliefs about {self.player_id}:"]
        for belief in self.sorted_by_knowledge():
            lines.append(f"  {belief.to_agent_format()}")
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

    def to_agent_format(self) -> str:
        """Format complete belief state for agent consumption."""
        lines = ["=== BELIEF STATE ==="]
        lines.append("(Sorted by knowledge: b+d, higher = more informative)")
        lines.append("(High d = strong evidence AGAINST, narrows search space)")
        lines.append("")

        # Villain beliefs
        for villain_beliefs in self.villain_beliefs.values():
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

    def process_observation(self, obs: Observation,
                           belief_state: BeliefState) -> BeliefState:
        """
        Process a single observation and update beliefs.

        Args:
            obs: The observation to process
            belief_state: Current belief state

        Returns:
            Updated belief state
        """
        self.observation_history.append(obs)

        # Get beliefs for this player
        if obs.player_id:
            villain_beliefs = belief_state.get_or_create_villain(obs.player_id)

            # Revise each belief based on observation consistency
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
        Extract observations from game state and process.
        """
        # Process recent actions not yet seen
        for action_record in game_state.action_history:
            if action_record.player_id == game_state.hero_id:
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

        return belief_state


def create_default_villain_beliefs(player_id: str) -> VillainBeliefs:
    """
    Create default (uncertain) beliefs for a new villain.

    Starts with vacuous or slightly informed priors.
    """
    beliefs = VillainBeliefs(player_id)

    # Player type beliefs - start vacuous
    beliefs.add_belief("is TAG (Tight-Aggressive)",
                       Opinion.vacuous(base_rate=0.25),
                       BeliefCategory.PLAYER_TYPE)
    beliefs.add_belief("is LAG (Loose-Aggressive)",
                       Opinion.vacuous(base_rate=0.15),
                       BeliefCategory.PLAYER_TYPE)
    beliefs.add_belief("is NIT (Very Tight)",
                       Opinion.vacuous(base_rate=0.15),
                       BeliefCategory.PLAYER_TYPE)
    beliefs.add_belief("is Fish (Recreational)",
                       Opinion.vacuous(base_rate=0.35),
                       BeliefCategory.PLAYER_TYPE)

    # Aggression beliefs
    beliefs.add_belief("is aggressive",
                       Opinion.vacuous(base_rate=0.4),
                       BeliefCategory.AGGRESSION)
    beliefs.add_belief("is passive",
                       Opinion.vacuous(base_rate=0.4),
                       BeliefCategory.AGGRESSION)

    # Bluff frequency
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
