"""
Test-time Recursive Thinking (TRT) strategy module.

Implements strategy-conditioned reasoning where the agent:
1. Explores multiple strategic approaches
2. Self-verifies without external feedback
3. Accumulates knowledge across iterations
4. Converges on a decision

Key insight from TRT: Negative "don't do" learnings (high disbelief)
are more valuable than positive "do" learnings for pruning search space.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto

from ...core.primitives import Action, ActionType
from ...core.game_state import GameState
from ...core.beliefs import BeliefState
from ...core.subjective_logic import Opinion


class Strategy(Enum):
    """
    Strategic approaches for decision-making.

    Each strategy represents a different reasoning lens.
    """
    GTO = "gto"              # Game Theory Optimal baseline
    EXPLOIT = "exploit"      # Exploit opponent tendencies
    DEFENSIVE = "defensive"  # Minimize risk/variance
    TRAPPING = "trapping"    # Slow-play strong hands
    BLUFF_HEAVY = "bluff"    # Aggressive bluffing

    @property
    def description(self) -> str:
        descriptions = {
            Strategy.GTO: "Play theoretically optimal, unexploitable",
            Strategy.EXPLOIT: "Deviate from GTO to exploit opponent leaks",
            Strategy.DEFENSIVE: "Prioritize pot control and risk reduction",
            Strategy.TRAPPING: "Disguise hand strength, induce bluffs",
            Strategy.BLUFF_HEAVY: "Increase bluff frequency, apply pressure",
        }
        return descriptions[self]


@dataclass
class StrategyEvaluation:
    """
    Evaluation of an action under a specific strategy.

    Attributes:
        strategy: The strategy used
        action: Recommended action
        reasoning: Why this action fits the strategy
        ev_estimate: Estimated EV if known
        confidence: Confidence in this evaluation
        contraindications: Reasons NOT to take this action
    """
    strategy: Strategy
    action: Action
    reasoning: str
    ev_estimate: Optional[float] = None
    confidence: float = 0.5
    contraindications: list[str] = field(default_factory=list)

    @property
    def has_contraindications(self) -> bool:
        return len(self.contraindications) > 0


@dataclass
class RolloutResult:
    """
    Result of a strategy-conditioned rollout.

    Attributes:
        strategy: Strategy used
        evaluation: The evaluation result
        iterations: Number of refinement iterations
        converged: Whether reasoning converged
        knowledge_gained: New knowledge from this rollout
    """
    strategy: Strategy
    evaluation: StrategyEvaluation
    iterations: int
    converged: bool
    knowledge_gained: list[str] = field(default_factory=list)


@dataclass
class TRTDecision:
    """
    Final decision from TRT process.

    Attributes:
        action: Chosen action
        primary_strategy: Main strategy driving decision
        rollout_results: Results from all strategy rollouts
        consensus_strength: How much strategies agree
        disbelief_pruned: Strategies/actions pruned by disbelief
    """
    action: Action
    primary_strategy: Strategy
    rollout_results: list[RolloutResult]
    consensus_strength: float
    disbelief_pruned: list[str] = field(default_factory=list)


class TRTEngine:
    """
    Test-time Recursive Thinking engine.

    Performs strategy-conditioned rollouts with self-verification
    and iterative refinement.
    """

    def __init__(
        self,
        strategies: Optional[list[Strategy]] = None,
        max_iterations: int = 3,
        convergence_threshold: float = 0.1
    ):
        """
        Initialize TRT engine.

        Args:
            strategies: Strategies to evaluate (default: all)
            max_iterations: Max refinement iterations per strategy
            convergence_threshold: Change threshold for convergence
        """
        self.strategies = strategies or list(Strategy)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.knowledge_base: list[str] = []

    def decide(
        self,
        game_state: GameState,
        belief_state: Optional[BeliefState] = None,
        evaluate_fn: Optional[Callable[[GameState, Strategy], StrategyEvaluation]] = None
    ) -> TRTDecision:
        """
        Make a decision using TRT process.

        Args:
            game_state: Current game state
            belief_state: Beliefs about opponents
            evaluate_fn: Function to evaluate action under strategy

        Returns:
            TRTDecision with chosen action and reasoning
        """
        evaluate_fn = evaluate_fn or self._default_evaluate

        rollout_results = []
        disbelief_pruned = []

        # Phase 1: Strategy-conditioned rollouts
        for strategy in self.strategies:
            # Check if strategy is contraindicated by beliefs
            if belief_state and self._strategy_contraindicated(strategy, belief_state):
                disbelief_pruned.append(f"{strategy.value}: contraindicated by beliefs")
                continue

            result = self._rollout(game_state, strategy, belief_state, evaluate_fn)
            rollout_results.append(result)

            # Accumulate knowledge
            self.knowledge_base.extend(result.knowledge_gained)

        # Phase 2: Self-verification
        verified_results = self._verify_results(rollout_results, game_state)

        # Phase 3: Consensus and decision
        action, primary_strategy, consensus = self._find_consensus(verified_results)

        return TRTDecision(
            action=action,
            primary_strategy=primary_strategy,
            rollout_results=verified_results,
            consensus_strength=consensus,
            disbelief_pruned=disbelief_pruned
        )

    def _strategy_contraindicated(
        self,
        strategy: Strategy,
        belief_state: BeliefState
    ) -> bool:
        """
        Check if a strategy is contraindicated by belief state.

        High disbelief in relevant beliefs prunes strategies.
        """
        # Example: If we strongly disbelieve villain bluffs often,
        # BLUFF_HEAVY strategy for hero is less useful (no need to balance)
        for villain_beliefs in belief_state.villain_beliefs.values():
            for belief in villain_beliefs.all_beliefs():
                op = belief.opinion

                # High disbelief = strong evidence against
                if op.disbelief > 0.6:
                    if strategy == Strategy.EXPLOIT and "fish" in belief.label.lower():
                        # If villain is NOT a fish, exploitation is harder
                        return True
                    if strategy == Strategy.TRAPPING and "aggressive" in belief.label.lower():
                        # If villain is NOT aggressive, trapping is less effective
                        return True

        return False

    def _rollout(
        self,
        game_state: GameState,
        strategy: Strategy,
        belief_state: Optional[BeliefState],
        evaluate_fn: Callable
    ) -> RolloutResult:
        """
        Perform a strategy-conditioned rollout with refinement.
        """
        prev_evaluation = None
        knowledge = []

        for iteration in range(self.max_iterations):
            evaluation = evaluate_fn(game_state, strategy)

            # Check convergence
            if prev_evaluation and self._converged(prev_evaluation, evaluation):
                return RolloutResult(
                    strategy=strategy,
                    evaluation=evaluation,
                    iterations=iteration + 1,
                    converged=True,
                    knowledge_gained=knowledge
                )

            # Extract knowledge from contraindications (negative learnings)
            if evaluation.has_contraindications:
                for contra in evaluation.contraindications:
                    knowledge.append(f"{strategy.value}: DON'T {contra}")

            prev_evaluation = evaluation

        return RolloutResult(
            strategy=strategy,
            evaluation=evaluation,
            iterations=self.max_iterations,
            converged=False,
            knowledge_gained=knowledge
        )

    def _converged(
        self,
        prev: StrategyEvaluation,
        curr: StrategyEvaluation
    ) -> bool:
        """Check if evaluation has converged."""
        # Same action and similar confidence
        if prev.action.action_type != curr.action.action_type:
            return False
        if abs(prev.confidence - curr.confidence) > self.convergence_threshold:
            return False
        return True

    def _verify_results(
        self,
        results: list[RolloutResult],
        game_state: GameState
    ) -> list[RolloutResult]:
        """
        Self-verify results without external feedback.

        Checks:
        1. GTO consistency
        2. EV sanity
        3. Cross-strategy consistency
        """
        verified = []

        for result in results:
            eval_ = result.evaluation
            passed = True

            # Sanity checks
            if eval_.action.action_type == ActionType.FOLD and game_state.to_call == 0:
                eval_.contraindications.append("Folding when check is free")
                passed = False

            # Cross-verify with GTO if not GTO strategy
            if result.strategy != Strategy.GTO:
                gto_result = next(
                    (r for r in results if r.strategy == Strategy.GTO),
                    None
                )
                if gto_result:
                    # Exploitation should have reason to deviate
                    if eval_.action != gto_result.evaluation.action:
                        if not eval_.reasoning:
                            eval_.contraindications.append(
                                "Deviates from GTO without clear reason"
                            )

            verified.append(result)

        return verified

    def _find_consensus(
        self,
        results: list[RolloutResult]
    ) -> tuple[Action, Strategy, float]:
        """
        Find consensus action across strategies.

        Returns:
            (action, primary_strategy, consensus_strength)
        """
        if not results:
            return Action.fold(), Strategy.DEFENSIVE, 0.0

        # Count actions
        action_counts: dict[ActionType, list[RolloutResult]] = {}
        for result in results:
            action_type = result.evaluation.action.action_type
            if action_type not in action_counts:
                action_counts[action_type] = []
            action_counts[action_type].append(result)

        # Find majority action
        best_action_type = max(action_counts.keys(), key=lambda k: len(action_counts[k]))
        consensus = len(action_counts[best_action_type]) / len(results)

        # Pick highest confidence result with this action
        matching_results = action_counts[best_action_type]
        best_result = max(matching_results, key=lambda r: r.evaluation.confidence)

        return (
            best_result.evaluation.action,
            best_result.strategy,
            consensus
        )

    def _default_evaluate(
        self,
        game_state: GameState,
        strategy: Strategy
    ) -> StrategyEvaluation:
        """
        Default strategy evaluation (simple heuristics).

        In production, this would be an LLM call with strategy conditioning.
        """
        hero = game_state.hero
        to_call = game_state.to_call

        if strategy == Strategy.GTO:
            # Simple GTO approximation
            if to_call == 0:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.check(),
                    reasoning="GTO: Check when facing no bet",
                    confidence=0.7
                )
            elif game_state.pot_odds < 0.25:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.call(to_call),
                    reasoning="GTO: Good pot odds, call",
                    confidence=0.6
                )
            else:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.fold(),
                    reasoning="GTO: Poor pot odds",
                    confidence=0.5
                )

        elif strategy == Strategy.DEFENSIVE:
            if to_call > hero.stack * 0.2:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.fold(),
                    reasoning="Defensive: Large bet relative to stack",
                    confidence=0.6,
                    contraindications=["May be folding best hand"]
                )
            elif to_call == 0:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.check(),
                    reasoning="Defensive: Pot control",
                    confidence=0.7
                )
            else:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.call(to_call),
                    reasoning="Defensive: Small call",
                    confidence=0.5
                )

        elif strategy == Strategy.EXPLOIT:
            # Would use belief state in full implementation
            return StrategyEvaluation(
                strategy=strategy,
                action=Action.call(to_call) if to_call > 0 else Action.check(),
                reasoning="Exploit: Default to call, need more reads",
                confidence=0.4,
                contraindications=["Insufficient information for exploitation"]
            )

        elif strategy == Strategy.TRAPPING:
            if to_call == 0:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.check(),
                    reasoning="Trapping: Disguise strength",
                    confidence=0.5,
                    contraindications=["May miss value"]
                )
            else:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.call(to_call),
                    reasoning="Trapping: Flat call to trap",
                    confidence=0.5
                )

        else:  # BLUFF_HEAVY
            if to_call == 0 and hero.stack > game_state.pot.total:
                bet_size = game_state.pot.total * 0.67
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.bet(bet_size),
                    reasoning="Bluff: Apply pressure",
                    confidence=0.4,
                    contraindications=["Risky without fold equity"]
                )
            else:
                return StrategyEvaluation(
                    strategy=strategy,
                    action=Action.fold() if to_call > 0 else Action.check(),
                    reasoning="Bluff: No good bluff spot",
                    confidence=0.5
                )

    def get_accumulated_knowledge(self) -> list[str]:
        """Get all knowledge accumulated across decisions."""
        return self.knowledge_base.copy()

    def clear_knowledge(self) -> None:
        """Clear accumulated knowledge."""
        self.knowledge_base.clear()
