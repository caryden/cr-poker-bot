"""
Tool interface for the poker agent.

Defines the abstract tool interface and concrete implementations
that wrap the poker tools for agent use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum, auto

from ..core.primitives import HoleCards, Board, Position
from ..core.game_state import GameState
from ..core.beliefs import BeliefState
from ..tools.hand_eval import evaluate_hand, HandEvaluation
from ..tools.equity import calculate_equity, calculate_equity_vs_range, HandRange, EquityResult
from ..tools.pot_odds import pot_odds, expected_value_call, minimum_defense_frequency, PotOddsResult, EVResult
from ..tools.gto import should_open, should_3bet, should_4bet, GTORecommendation
from ..tools.board_texture import analyze_board, BoardTexture
from ..tools.bet_sizing import (
    get_postflop_sizing, get_preflop_sizing,
    HandStrengthCategory, BetSizing
)


class ToolName(Enum):
    """Available tools for the agent."""
    HAND_EVAL = "hand_eval"
    EQUITY_CALC = "equity_calc"
    POT_ODDS = "pot_odds"
    EV_CALC = "ev_calc"
    GTO_ADVISOR = "gto_advisor"
    BELIEF_QUERY = "belief_query"
    BOARD_TEXTURE = "board_texture"
    BET_SIZING = "bet_sizing"


@dataclass
class ToolCall:
    """
    A request to invoke a tool.

    Attributes:
        tool: Which tool to invoke
        params: Parameters for the tool
        reasoning: Why the agent is calling this tool
    """
    tool: ToolName
    params: dict[str, Any]
    reasoning: str = ""


@dataclass
class ToolResult:
    """
    Result from a tool invocation.

    Attributes:
        tool: Which tool was invoked
        success: Whether the call succeeded
        result: The tool's output
        error: Error message if failed
        formatted: Human-readable formatted result
    """
    tool: ToolName
    success: bool
    result: Any
    error: Optional[str] = None
    formatted: str = ""

    def __str__(self) -> str:
        if not self.success:
            return f"[{self.tool.value}] ERROR: {self.error}"
        return f"[{self.tool.value}] {self.formatted}"


class Tool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> ToolName:
        """Tool identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description for the agent."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, str]:
        """Parameter descriptions."""
        pass

    @abstractmethod
    def execute(self, params: dict[str, Any], context: 'ToolContext') -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def to_prompt_description(self) -> str:
        """Format tool for inclusion in system prompt."""
        param_lines = [f"    - {k}: {v}" for k, v in self.parameters.items()]
        params_str = "\n".join(param_lines) if param_lines else "    (none)"
        return f"""**{self.name.value}**: {self.description}
  Parameters:
{params_str}"""


@dataclass
class ToolContext:
    """
    Context available to tools during execution.

    Provides access to game state, beliefs, and other shared state.
    """
    game_state: Optional[GameState] = None
    belief_state: Optional[BeliefState] = None
    hero_cards: Optional[HoleCards] = None
    board: Optional[Board] = None


# ============== Concrete Tools ==============

class HandEvalTool(Tool):
    """Evaluate the current hand strength."""

    @property
    def name(self) -> ToolName:
        return ToolName.HAND_EVAL

    @property
    def description(self) -> str:
        return "Evaluate hand strength. Returns hand rank (pair, flush, etc.) and description."

    @property
    def parameters(self) -> dict[str, str]:
        return {}  # Uses context

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.hero_cards is None:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error="No hole cards available"
            )

        board = context.board or Board()

        if len(board) < 3:
            return ToolResult(
                tool=self.name,
                success=True,
                result={"rank": "preflop", "cards": context.hero_cards.notation},
                formatted=f"Preflop: {context.hero_cards.notation}"
            )

        try:
            evaluation = evaluate_hand(context.hero_cards, board)
            return ToolResult(
                tool=self.name,
                success=True,
                result=evaluation,
                formatted=f"{evaluation.rank.name}: {evaluation.description}"
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error=str(e)
            )


class EquityCalcTool(Tool):
    """Calculate hand equity."""

    @property
    def name(self) -> ToolName:
        return ToolName.EQUITY_CALC

    @property
    def description(self) -> str:
        return "Calculate equity vs random hands or a specific range. Returns win/tie/lose percentages."

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "vs_range": "(optional) Opponent range like 'AA,KK,QQ' or 'JJ+'. Omit for vs random.",
            "num_opponents": "(optional) Number of opponents (default 1)"
        }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.hero_cards is None:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error="No hole cards available"
            )

        board = context.board or Board()
        vs_range = params.get("vs_range")
        num_opponents = params.get("num_opponents", 1)

        try:
            if vs_range:
                opponent_range = HandRange.from_string(vs_range)
                result = calculate_equity_vs_range(
                    context.hero_cards, board, opponent_range, simulations=5000
                )
                formatted = f"Equity vs {vs_range}: {result.equity:.1%} (W:{result.win_pct:.1%} T:{result.tie_pct:.1%} L:{result.lose_pct:.1%})"
            else:
                result = calculate_equity(
                    context.hero_cards, board, num_opponents, simulations=5000
                )
                formatted = f"Equity vs {num_opponents} random: {result.equity:.1%} (W:{result.win_pct:.1%} T:{result.tie_pct:.1%} L:{result.lose_pct:.1%})"

            return ToolResult(
                tool=self.name,
                success=True,
                result=result,
                formatted=formatted
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error=str(e)
            )


class PotOddsTool(Tool):
    """Calculate pot odds."""

    @property
    def name(self) -> ToolName:
        return ToolName.POT_ODDS

    @property
    def parameters(self) -> dict[str, str]:
        return {}  # Uses context

    @property
    def description(self) -> str:
        return "Calculate current pot odds and required equity to call."

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.game_state is None:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error="No game state available"
            )

        gs = context.game_state
        result = pot_odds(gs.pot.total, gs.to_call)

        formatted = f"Pot: {gs.pot.total:.0f}, To call: {gs.to_call:.0f}\n"
        formatted += f"Pot odds: {result.pot_odds:.1%} ({result.pot_odds_ratio})\n"
        formatted += f"Need {result.break_even_equity:.1%} equity to call profitably"

        return ToolResult(
            tool=self.name,
            success=True,
            result=result,
            formatted=formatted
        )


class EVCalcTool(Tool):
    """Calculate expected value of actions."""

    @property
    def name(self) -> ToolName:
        return ToolName.EV_CALC

    @property
    def description(self) -> str:
        return "Calculate EV of calling or betting given equity estimate."

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "action": "'call' or 'bet'",
            "equity": "Estimated equity (0.0 to 1.0)",
            "bet_size": "(for bet) Bet amount",
            "fold_equity": "(for bet) Estimated fold equity (0.0 to 1.0)"
        }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.game_state is None:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error="No game state available"
            )

        action = params.get("action", "call")
        equity = params.get("equity", 0.5)
        gs = context.game_state

        try:
            if action == "call":
                result = expected_value_call(
                    gs.pot.total, gs.to_call, equity, gs.table.big_blind
                )
                return ToolResult(
                    tool=self.name,
                    success=True,
                    result=result,
                    formatted=str(result)
                )
            else:
                # For betting, need more complex calculation
                from ..tools.pot_odds import PotOddsCalculator
                calc = PotOddsCalculator()
                bet_size = params.get("bet_size", gs.pot.total * 0.67)
                fold_equity = params.get("fold_equity", 0.3)
                result = calc.expected_value_bet(
                    gs.pot.total, bet_size, fold_equity, equity, gs.table.big_blind
                )
                return ToolResult(
                    tool=self.name,
                    success=True,
                    result=result,
                    formatted=str(result)
                )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error=str(e)
            )


class GTOAdvisorTool(Tool):
    """Get GTO-based recommendations."""

    @property
    def name(self) -> ToolName:
        return ToolName.GTO_ADVISOR

    @property
    def description(self) -> str:
        return "Get GTO recommendation for current situation (open, 3-bet, etc.)."

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "situation": "'open', '3bet', '4bet', or 'cbet'",
            "villain_position": "(for 3bet/4bet) Villain's position"
        }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.hero_cards is None or context.game_state is None:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error="No hole cards or game state available"
            )

        situation = params.get("situation", "open")
        hero_pos = context.game_state.hero.position

        try:
            if situation == "open":
                result = should_open(context.hero_cards, hero_pos)
            elif situation == "3bet":
                villain_pos_str = params.get("villain_position", "UTG")
                villain_pos = Position[villain_pos_str]
                result = should_3bet(context.hero_cards, hero_pos, villain_pos)
            elif situation == "4bet":
                villain_pos_str = params.get("villain_position", "BTN")
                villain_pos = Position[villain_pos_str]
                result = should_4bet(context.hero_cards, villain_pos)
            else:
                result = GTORecommendation(
                    primary_action=None,
                    reasoning="Unknown situation"
                )

            return ToolResult(
                tool=self.name,
                success=True,
                result=result,
                formatted=str(result)
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error=str(e)
            )


class BeliefQueryTool(Tool):
    """Query current beliefs about opponents."""

    @property
    def name(self) -> ToolName:
        return ToolName.BELIEF_QUERY

    @property
    def description(self) -> str:
        return "Query beliefs about a specific opponent. Returns SL opinions sorted by knowledge."

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "player_id": "(optional) Specific player to query, or 'all'"
        }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.belief_state is None:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error="No belief state available"
            )

        player_id = params.get("player_id", "all")

        try:
            if player_id == "all":
                formatted = context.belief_state.to_agent_format()
            else:
                villain_beliefs = context.belief_state.villain_beliefs.get(player_id)
                if villain_beliefs:
                    formatted = villain_beliefs.to_agent_format()
                else:
                    formatted = f"No beliefs recorded for {player_id}"

            return ToolResult(
                tool=self.name,
                success=True,
                result=context.belief_state,
                formatted=formatted
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error=str(e)
            )


class BoardTextureTool(Tool):
    """Analyze the board texture."""

    @property
    def name(self) -> ToolName:
        return ToolName.BOARD_TEXTURE

    @property
    def description(self) -> str:
        return "Analyze board texture (dry/wet, paired, connected, flush draws). Helps determine optimal bet sizing and strategy."

    @property
    def parameters(self) -> dict[str, str]:
        return {}  # Uses context

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.board is None or len(context.board) < 3:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error="No board or insufficient cards (need at least flop)"
            )

        try:
            texture = analyze_board(context.board)

            # Build detailed formatted output
            lines = [
                f"Board: {context.board}",
                f"Texture: {texture}",
                f"",
                f"Characteristics:",
                f"  - Wetness: {texture.wetness.name.replace('_', ' ')}",
                f"  - Pairedness: {texture.pairedness.name}",
                f"  - High card: {texture.high_card.name if texture.high_card else 'N/A'}",
                f"",
                f"Draws:",
                f"  - Flush possible: {'Yes' if texture.flush_possible else 'No'}",
                f"  - Flush draw: {'Yes' if texture.flush_draw_possible else 'No'}",
                f"  - Straight possible: {'Yes' if texture.straight_possible else 'No'}",
                f"  - Straight draw: {'Yes' if texture.straight_draw_possible else 'No'}",
                f"",
                f"Strategic implications:",
            ]

            if texture.is_static:
                lines.append("  - Static board favors preflop aggressor")
                lines.append("  - Smaller bet sizes work well")
            if texture.is_draw_heavy:
                lines.append("  - Draw-heavy board - need protection bets")
                lines.append("  - Larger sizing recommended for value")
            if texture.favors_aggressor:
                lines.append("  - Board favors preflop aggressor's range")

            formatted = "\n".join(lines)

            return ToolResult(
                tool=self.name,
                success=True,
                result=texture,
                formatted=formatted
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error=str(e)
            )


class BetSizingTool(Tool):
    """Get bet sizing recommendations."""

    @property
    def name(self) -> ToolName:
        return ToolName.BET_SIZING

    @property
    def description(self) -> str:
        return "Get recommended bet sizing based on hand strength, board texture, and game situation."

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "hand_strength": "Category: 'trash', 'weak_draw', 'strong_draw', 'marginal', 'medium', 'strong', 'very_strong', 'monster'",
            "is_aggressor": "(optional) Whether we were preflop aggressor (default false)",
            "bet_previous": "(optional) Whether we bet the previous street (default false)"
        }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.game_state is None:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error="No game state available"
            )

        # Parse hand strength
        strength_map = {
            'trash': HandStrengthCategory.TRASH,
            'weak_draw': HandStrengthCategory.WEAK_DRAW,
            'strong_draw': HandStrengthCategory.STRONG_DRAW,
            'marginal': HandStrengthCategory.MARGINAL,
            'medium': HandStrengthCategory.MEDIUM,
            'strong': HandStrengthCategory.STRONG,
            'very_strong': HandStrengthCategory.VERY_STRONG,
            'monster': HandStrengthCategory.MONSTER,
        }

        strength_str = params.get("hand_strength", "medium").lower()
        hand_strength = strength_map.get(strength_str, HandStrengthCategory.MEDIUM)

        is_aggressor = params.get("is_aggressor", False)
        bet_previous = params.get("bet_previous", False)

        try:
            gs = context.game_state
            pot = gs.pot.total
            bb = gs.table.big_blind

            sizing = get_postflop_sizing(
                gs, hand_strength, is_aggressor, bet_previous
            )

            amount = sizing.get_amount(pot, bb)

            lines = [
                f"Recommended sizing: {sizing}",
                f"",
                f"Bet amount: {amount:.1f} ({sizing.size_pot_fraction*100:.0f}% of pot)",
                f"Purpose: {sizing.purpose.name}",
                f"",
                f"Context:",
                f"  - Pot: {pot:.0f}",
                f"  - Street: {gs.street.name}",
                f"  - Hand strength: {hand_strength.name}",
            ]

            if sizing.alternative_size:
                alt_amount = pot * sizing.alternative_size
                lines.append(f"  - Alternative: {alt_amount:.1f} ({sizing.alternative_size*100:.0f}% of pot)")

            formatted = "\n".join(lines)

            return ToolResult(
                tool=self.name,
                success=True,
                result=sizing,
                formatted=formatted
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                result=None,
                error=str(e)
            )


class ToolRegistry:
    """
    Registry of available tools.

    Manages tool lookup and provides tool descriptions for prompts.
    """

    def __init__(self):
        self.tools: dict[ToolName, Tool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default tools."""
        self.register(HandEvalTool())
        self.register(EquityCalcTool())
        self.register(PotOddsTool())
        self.register(EVCalcTool())
        self.register(GTOAdvisorTool())
        self.register(BeliefQueryTool())
        self.register(BoardTextureTool())
        self.register(BetSizingTool())

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: ToolName) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute a tool call."""
        tool = self.get(call.tool)
        if tool is None:
            return ToolResult(
                tool=call.tool,
                success=False,
                result=None,
                error=f"Unknown tool: {call.tool}"
            )
        return tool.execute(call.params, context)

    def get_tools_prompt(self) -> str:
        """Generate tools description for system prompt."""
        lines = ["## Available Tools\n"]
        for tool in self.tools.values():
            lines.append(tool.to_prompt_description())
            lines.append("")
        return "\n".join(lines)
