"""Tests for ReAct agent."""

import pytest
from src.core.primitives import Card, Rank, Suit, HoleCards, Position, ActionType
from src.core.game_state import create_6max_game
from src.core.beliefs import BeliefState
from src.agent.react import (
    AgentPhase, ReasoningStep, AgentDecision, ReActAgent
)
from src.agent.tools import ToolRegistry, ToolContext


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


class TestReActAgent:
    """Tests for ReActAgent class."""

    def test_agent_creation(self):
        agent = ReActAgent()
        assert agent.tools is not None
        assert agent.max_steps == 10

    def test_agent_makes_decision(self, sample_game):
        agent = ReActAgent(max_steps=5)
        decision = agent.decide(sample_game)

        assert isinstance(decision, AgentDecision)
        assert decision.action is not None
        assert decision.reasoning is not None

    def test_agent_with_belief_state(self, sample_game):
        agent = ReActAgent()
        belief_state = BeliefState()

        decision = agent.decide(sample_game, belief_state)

        assert isinstance(decision, AgentDecision)

    def test_agent_records_steps(self, sample_game):
        agent = ReActAgent(max_steps=5)
        decision = agent.decide(sample_game)

        assert len(decision.steps) > 0

    def test_decision_to_dict(self, sample_game):
        agent = ReActAgent(max_steps=3)
        decision = agent.decide(sample_game)

        d = decision.to_dict()
        assert 'action' in d
        assert 'reasoning' in d
        assert 'confidence' in d


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""

    def test_reasoning_step_creation(self):
        step = ReasoningStep(
            phase=AgentPhase.REASON,
            thought="Analyzing the situation..."
        )

        assert step.phase == AgentPhase.REASON
        assert step.thought == "Analyzing the situation..."
        assert step.tool_call is None

    def test_step_with_tool_result(self):
        from src.agent.tools import ToolCall, ToolResult, ToolName

        call = ToolCall(tool=ToolName.POT_ODDS, params={})
        result = ToolResult(
            tool=ToolName.POT_ODDS,
            success=True,
            result={"odds": 0.33},
            formatted="Pot odds: 33%"
        )

        step = ReasoningStep(
            phase=AgentPhase.OBSERVE,
            thought="Checking pot odds",
            tool_call=call,
            tool_result=result
        )

        assert step.tool_call is not None
        assert step.tool_result.success


class TestAgentDecision:
    """Tests for AgentDecision dataclass."""

    def test_decision_creation(self):
        from src.core.primitives import Action

        decision = AgentDecision(
            action=Action.call(50),
            reasoning="Good pot odds with strong hand",
            confidence=0.8,
            verification_passed=True
        )

        assert decision.action.action_type == ActionType.CALL
        assert decision.confidence == 0.8
        assert decision.verification_passed


class TestAgentParsing:
    """Tests for agent response parsing."""

    def test_parse_tool_call(self, sample_game):
        agent = ReActAgent()
        context = ToolContext(
            game_state=sample_game,
            hero_cards=sample_game.hero.hole_cards
        )

        response = "Let me check pot odds.\n\nTOOL: pot_odds()"
        step = agent._parse_response(response, context)

        assert step.phase == AgentPhase.OBSERVE
        assert step.tool_call is not None

    def test_parse_decision_fold(self, sample_game):
        agent = ReActAgent()
        context = ToolContext(game_state=sample_game)

        response = "I should fold here.\n\nDECISION: fold"
        step = agent._parse_response(response, context)

        assert step.phase == AgentPhase.DECIDE
        assert step.action_proposed is not None
        assert step.action_proposed.action_type == ActionType.FOLD

    def test_parse_decision_call(self, sample_game):
        agent = ReActAgent()
        context = ToolContext(game_state=sample_game)

        response = "Good odds, I'll call.\n\nDECISION: call 50"
        step = agent._parse_response(response, context)

        assert step.phase == AgentPhase.DECIDE
        assert step.action_proposed.action_type == ActionType.CALL

    def test_parse_decision_raise(self, sample_game):
        agent = ReActAgent()
        context = ToolContext(game_state=sample_game)

        response = "I should raise for value.\n\nDECISION: raise 100"
        step = agent._parse_response(response, context)

        assert step.phase == AgentPhase.DECIDE
        assert step.action_proposed.action_type == ActionType.RAISE

    def test_parse_reasoning_only(self, sample_game):
        agent = ReActAgent()
        context = ToolContext(game_state=sample_game)

        response = "Let me think about this situation..."
        step = agent._parse_response(response, context)

        assert step.phase == AgentPhase.REASON
        assert step.tool_call is None
        assert step.action_proposed is None


class TestCustomLLMCall:
    """Tests for custom LLM call function."""

    def test_custom_llm_immediate_decision(self, sample_game):
        def immediate_fold(prompt: str) -> str:
            return "Bad hand, folding.\n\nDECISION: fold"

        agent = ReActAgent(llm_call=immediate_fold)
        decision = agent.decide(sample_game)

        assert decision.action.action_type == ActionType.FOLD
        assert len(decision.steps) == 1

    def test_custom_llm_with_tool_use(self, sample_game):
        call_count = [0]

        def tool_then_decide(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "Checking odds.\n\nTOOL: pot_odds()"
            else:
                return "Odds are good.\n\nDECISION: call 10"

        agent = ReActAgent(llm_call=tool_then_decide)
        decision = agent.decide(sample_game)

        assert decision.action.action_type == ActionType.CALL
        assert len(decision.steps) == 2
