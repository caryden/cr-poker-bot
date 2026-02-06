"""Tests for agent tool interface."""

import pytest
from src.core.primitives import Card, Rank, Suit, HoleCards, Board, Position
from src.core.game_state import create_6max_game
from src.core.beliefs import BeliefState, create_default_villain_beliefs
from src.agent.tools import (
    ToolName, ToolCall, ToolResult, ToolContext, ToolRegistry,
    HandEvalTool, EquityCalcTool, PotOddsTool, GTOAdvisorTool, BeliefQueryTool
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


@pytest.fixture
def sample_context(sample_game):
    """Create tool context for testing."""
    belief_state = BeliefState()
    belief_state.get_or_create_villain("villain_UTG")
    return ToolContext(
        game_state=sample_game,
        belief_state=belief_state,
        hero_cards=sample_game.hero.hole_cards,
        board=sample_game.board
    )


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_registry_has_default_tools(self):
        registry = ToolRegistry()
        assert registry.get(ToolName.HAND_EVAL) is not None
        assert registry.get(ToolName.EQUITY_CALC) is not None
        assert registry.get(ToolName.POT_ODDS) is not None
        assert registry.get(ToolName.GTO_ADVISOR) is not None

    def test_registry_execute(self, sample_context):
        registry = ToolRegistry()
        call = ToolCall(tool=ToolName.POT_ODDS, params={})
        result = registry.execute(call, sample_context)
        assert result.success

    def test_get_tools_prompt(self):
        registry = ToolRegistry()
        prompt = registry.get_tools_prompt()
        assert "hand_eval" in prompt
        assert "equity_calc" in prompt


class TestHandEvalTool:
    """Tests for HandEvalTool."""

    def test_preflop_evaluation(self, sample_context):
        tool = HandEvalTool()
        result = tool.execute({}, sample_context)

        assert result.success
        assert "AKo" in result.formatted

    def test_postflop_evaluation(self, sample_context):
        # Add a flop
        sample_context.board = Board([
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.TWO, Suit.HEARTS),
        ])

        tool = HandEvalTool()
        result = tool.execute({}, sample_context)

        assert result.success
        assert "TWO_PAIR" in result.formatted or "Two Pair" in result.formatted

    def test_no_cards_error(self):
        tool = HandEvalTool()
        context = ToolContext()
        result = tool.execute({}, context)

        assert not result.success
        assert "No hole cards" in result.error


class TestEquityCalcTool:
    """Tests for EquityCalcTool."""

    def test_equity_vs_random(self, sample_context):
        tool = EquityCalcTool()
        result = tool.execute({}, sample_context)

        assert result.success
        assert "%" in result.formatted

    def test_equity_vs_range(self, sample_context):
        tool = EquityCalcTool()
        result = tool.execute({"vs_range": "QQ,JJ,TT"}, sample_context)

        assert result.success
        assert "QQ,JJ,TT" in result.formatted


class TestPotOddsTool:
    """Tests for PotOddsTool."""

    def test_pot_odds_calculation(self, sample_context):
        tool = PotOddsTool()
        result = tool.execute({}, sample_context)

        assert result.success
        assert "Pot:" in result.formatted
        assert "%" in result.formatted


class TestGTOAdvisorTool:
    """Tests for GTOAdvisorTool."""

    def test_gto_open_recommendation(self, sample_context):
        tool = GTOAdvisorTool()
        result = tool.execute({"situation": "open"}, sample_context)

        assert result.success
        # AKo from BTN should be a raise
        assert "RAISE" in result.formatted or "Raise" in result.formatted

    def test_gto_3bet_recommendation(self, sample_context):
        tool = GTOAdvisorTool()
        result = tool.execute({
            "situation": "3bet",
            "villain_position": "UTG"
        }, sample_context)

        assert result.success


class TestBeliefQueryTool:
    """Tests for BeliefQueryTool."""

    def test_query_all_beliefs(self, sample_context):
        # Add some beliefs
        villain_beliefs = sample_context.belief_state.get_or_create_villain("villain_UTG")
        from src.core.subjective_logic import Opinion
        villain_beliefs.add_belief("is aggressive", Opinion(0.6, 0.2, 0.2, 0.5))

        tool = BeliefQueryTool()
        result = tool.execute({"player_id": "all"}, sample_context)

        assert result.success
        assert "villain_UTG" in result.formatted

    def test_no_belief_state(self):
        tool = BeliefQueryTool()
        context = ToolContext()
        result = tool.execute({}, context)

        assert not result.success


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        call = ToolCall(
            tool=ToolName.EQUITY_CALC,
            params={"vs_range": "AA,KK"},
            reasoning="Need to check equity"
        )

        assert call.tool == ToolName.EQUITY_CALC
        assert call.params["vs_range"] == "AA,KK"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result_str(self):
        result = ToolResult(
            tool=ToolName.POT_ODDS,
            success=True,
            result={"pot_odds": 0.33},
            formatted="Pot odds: 33%"
        )

        assert "[pot_odds]" in str(result)
        assert "33%" in str(result)

    def test_error_result_str(self):
        result = ToolResult(
            tool=ToolName.HAND_EVAL,
            success=False,
            result=None,
            error="No cards"
        )

        assert "ERROR" in str(result)
        assert "No cards" in str(result)
