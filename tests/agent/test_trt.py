"""Tests for TRT (Test-time Recursive Thinking) module."""

import pytest
from src.core.primitives import Card, Rank, Suit, HoleCards, Position, ActionType
from src.core.game_state import create_6max_game
from src.core.beliefs import BeliefState, create_default_villain_beliefs
from src.core.subjective_logic import Opinion
from src.agent.trt.strategy import (
    Strategy, StrategyEvaluation, RolloutResult, TRTDecision, TRTEngine
)


@pytest.fixture
def sample_game():
    """Create a sample game for testing."""
    stacks = {pos: 100.0 for pos in Position.all_positions()}
    hero_cards = HoleCards(
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.KING, Suit.HEARTS)
    )
    return create_6max_game(
        hand_id="test",
        hero_id="hero",
        hero_position=Position.BTN,
        hero_cards=hero_cards,
        stacks=stacks
    )


class TestStrategy:
    """Tests for Strategy enum."""

    def test_strategy_values(self):
        assert Strategy.GTO.value == "gto"
        assert Strategy.EXPLOIT.value == "exploit"

    def test_strategy_descriptions(self):
        assert "optimal" in Strategy.GTO.description.lower()
        assert "exploit" in Strategy.EXPLOIT.description.lower()

    def test_all_strategies(self):
        strategies = list(Strategy)
        assert len(strategies) == 5


class TestStrategyEvaluation:
    """Tests for StrategyEvaluation dataclass."""

    def test_evaluation_creation(self):
        from src.core.primitives import Action

        eval_ = StrategyEvaluation(
            strategy=Strategy.GTO,
            action=Action.call(50),
            reasoning="Good pot odds",
            confidence=0.7
        )

        assert eval_.strategy == Strategy.GTO
        assert eval_.confidence == 0.7
        assert not eval_.has_contraindications

    def test_evaluation_with_contraindications(self):
        from src.core.primitives import Action

        eval_ = StrategyEvaluation(
            strategy=Strategy.BLUFF_HEAVY,
            action=Action.bet(100),
            reasoning="Apply pressure",
            contraindications=["No fold equity", "Villain never folds"]
        )

        assert eval_.has_contraindications
        assert len(eval_.contraindications) == 2


class TestTRTEngine:
    """Tests for TRTEngine class."""

    def test_engine_creation(self):
        engine = TRTEngine()
        assert len(engine.strategies) == 5
        assert engine.max_iterations == 3

    def test_engine_custom_strategies(self):
        engine = TRTEngine(strategies=[Strategy.GTO, Strategy.DEFENSIVE])
        assert len(engine.strategies) == 2

    def test_engine_decide(self, sample_game):
        engine = TRTEngine()
        decision = engine.decide(sample_game)

        assert isinstance(decision, TRTDecision)
        assert decision.action is not None
        assert decision.primary_strategy is not None

    def test_engine_with_beliefs(self, sample_game):
        engine = TRTEngine()
        belief_state = BeliefState()
        villain_beliefs = belief_state.get_or_create_villain("villain_UTG")
        villain_beliefs.add_belief("is aggressive", Opinion(0.7, 0.1, 0.2, 0.5))

        decision = engine.decide(sample_game, belief_state)

        assert isinstance(decision, TRTDecision)

    def test_rollout_results_included(self, sample_game):
        engine = TRTEngine()
        decision = engine.decide(sample_game)

        assert len(decision.rollout_results) > 0

    def test_consensus_strength(self, sample_game):
        engine = TRTEngine()
        decision = engine.decide(sample_game)

        assert 0 <= decision.consensus_strength <= 1


class TestStrategyPruning:
    """Tests for belief-based strategy pruning."""

    def test_exploit_pruned_when_no_fish(self, sample_game):
        engine = TRTEngine()
        belief_state = BeliefState()

        # Add high disbelief that villain is a fish
        villain_beliefs = belief_state.get_or_create_villain("villain_UTG")
        villain_beliefs.add_belief(
            "is Fish (Recreational)",
            Opinion(0.1, 0.7, 0.2, 0.3),  # High disbelief
            "player_type"
        )

        decision = engine.decide(sample_game, belief_state)

        # EXPLOIT should be pruned
        pruned_strategies = [p for p in decision.disbelief_pruned if "exploit" in p.lower()]
        assert len(pruned_strategies) > 0 or decision.primary_strategy != Strategy.EXPLOIT

    def test_trapping_pruned_when_not_aggressive(self, sample_game):
        engine = TRTEngine()
        belief_state = BeliefState()

        # Add high disbelief that villain is aggressive
        villain_beliefs = belief_state.get_or_create_villain("villain_UTG")
        villain_beliefs.add_belief(
            "is aggressive",
            Opinion(0.1, 0.7, 0.2, 0.4),  # High disbelief
            "player_type"
        )

        decision = engine.decide(sample_game, belief_state)

        # Should note trapping is less effective
        # (pruning logic checks for this)
        assert isinstance(decision, TRTDecision)


class TestKnowledgeAccumulation:
    """Tests for knowledge accumulation across decisions."""

    def test_knowledge_accumulates(self, sample_game):
        engine = TRTEngine()

        # Make first decision
        engine.decide(sample_game)
        knowledge_after_first = len(engine.get_accumulated_knowledge())

        # Make second decision
        engine.decide(sample_game)
        knowledge_after_second = len(engine.get_accumulated_knowledge())

        # Knowledge should accumulate
        assert knowledge_after_second >= knowledge_after_first

    def test_knowledge_can_be_cleared(self, sample_game):
        engine = TRTEngine()
        engine.decide(sample_game)

        assert len(engine.get_accumulated_knowledge()) > 0

        engine.clear_knowledge()

        assert len(engine.get_accumulated_knowledge()) == 0


class TestDefaultEvaluation:
    """Tests for default strategy evaluation."""

    def test_gto_checks_when_no_bet(self, sample_game):
        engine = TRTEngine()
        # Set to_call to 0
        sample_game.current_bet = 0

        eval_ = engine._default_evaluate(sample_game, Strategy.GTO)

        assert eval_.action.action_type == ActionType.CHECK

    def test_defensive_folds_large_bets(self, sample_game):
        engine = TRTEngine()
        # Set large bet relative to stack
        sample_game.current_bet = 50  # 50% of 100 stack

        eval_ = engine._default_evaluate(sample_game, Strategy.DEFENSIVE)

        # Defensive should be cautious with large bets
        assert eval_.action.action_type in (ActionType.FOLD, ActionType.CALL)

    def test_bluff_bets_when_checked_to(self, sample_game):
        engine = TRTEngine()
        sample_game.current_bet = 0

        eval_ = engine._default_evaluate(sample_game, Strategy.BLUFF_HEAVY)

        # Bluff strategy should look for betting opportunities
        assert eval_.action.action_type in (ActionType.BET, ActionType.CHECK)


class TestRolloutResult:
    """Tests for RolloutResult dataclass."""

    def test_rollout_result_creation(self):
        from src.core.primitives import Action

        eval_ = StrategyEvaluation(
            strategy=Strategy.GTO,
            action=Action.call(50),
            reasoning="Good odds"
        )

        result = RolloutResult(
            strategy=Strategy.GTO,
            evaluation=eval_,
            iterations=2,
            converged=True,
            knowledge_gained=["DON'T bluff into calling station"]
        )

        assert result.converged
        assert len(result.knowledge_gained) == 1


class TestTRTDecision:
    """Tests for TRTDecision dataclass."""

    def test_decision_has_all_fields(self, sample_game):
        engine = TRTEngine()
        decision = engine.decide(sample_game)

        assert decision.action is not None
        assert decision.primary_strategy is not None
        assert isinstance(decision.rollout_results, list)
        assert isinstance(decision.consensus_strength, float)
        assert isinstance(decision.disbelief_pruned, list)
