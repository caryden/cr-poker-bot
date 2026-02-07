"""
ReAct (Reasoning + Acting) agent for poker decisions.

Implements the ReAct pattern where the agent:
1. Observes the current situation
2. Thinks about what to do
3. Acts by calling tools or making decisions
4. Repeats until a decision is reached

Integrated with belief state conditioning and TRT self-verification.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto
import json

from ..core.primitives import Action, ActionType, Position
from ..core.game_state import GameState
from ..core.beliefs import BeliefState
from .tools import ToolRegistry, ToolContext, ToolCall, ToolResult, ToolName
from ..logging_config import get_logger

logger = get_logger(__name__)


class AgentPhase(Enum):
    """Current phase of agent reasoning."""
    OBSERVE = auto()      # Gathering information
    REASON = auto()       # Analyzing situation
    VERIFY = auto()       # Self-verification (TRT)
    DECIDE = auto()       # Making final decision
    COMPLETE = auto()     # Decision made


@dataclass
class ReasoningStep:
    """
    A single step in the agent's reasoning process.

    Attributes:
        phase: Current reasoning phase
        thought: The agent's thought/reasoning
        tool_call: Tool call if any
        tool_result: Result from tool if called
        action_proposed: Action being considered
    """
    phase: AgentPhase
    thought: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    action_proposed: Optional[Action] = None


@dataclass
class AgentDecision:
    """
    Final decision from the agent.

    Attributes:
        action: The chosen action
        reasoning: Full reasoning chain
        confidence: Confidence in decision (0-1)
        verification_passed: Whether TRT verification passed
        steps: All reasoning steps taken
    """
    action: Action
    reasoning: str
    confidence: float
    verification_passed: bool
    steps: list[ReasoningStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for logging/display."""
        return {
            'action': str(self.action),
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'verification_passed': self.verification_passed,
            'num_steps': len(self.steps)
        }


class ReActAgent:
    """
    ReAct-style agent for poker decisions.

    Uses a think-act loop with:
    - Tool calls for information gathering
    - Belief state conditioning
    - TRT self-verification
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        llm_call: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize the agent.

        Args:
            tool_registry: Registry of available tools
            max_steps: Maximum reasoning steps before forcing decision
            llm_call: Function to call LLM (prompt -> response)
        """
        self.tools = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        if llm_call is None:
            import sys
            print("[ReActAgent] WARNING: No LLM client provided, using mock", file=sys.stderr)
        self.llm_call = llm_call or self._mock_llm_call
        self.steps: list[ReasoningStep] = []

    def decide(
        self,
        game_state: GameState,
        belief_state: Optional[BeliefState] = None
    ) -> AgentDecision:
        """
        Make a decision for the current game state.

        Args:
            game_state: Current game state
            belief_state: Current beliefs about opponents

        Returns:
            AgentDecision with chosen action and reasoning
        """
        self.steps = []

        # Create tool context
        context = ToolContext(
            game_state=game_state,
            belief_state=belief_state,
            hero_cards=game_state.hero.hole_cards,
            board=game_state.board
        )

        # Build initial prompt
        prompt = self._build_initial_prompt(game_state, belief_state)
        logger.info("Agent deciding: %s, to_call=%.0f, pot=%.0f",
                     game_state.street.value, game_state.to_call, game_state.pot.total)
        logger.debug("Agent prompt:\n%s", prompt)

        # ReAct loop
        for step_num in range(self.max_steps):
            # Get LLM response
            response = self.llm_call(prompt)
            logger.debug("LLM response (step %d):\n%s", step_num + 1, response)

            # Parse response
            step = self._parse_response(response, context)
            self.steps.append(step)

            if step.tool_call:
                logger.info("  Tool call: %s -> %s",
                             step.tool_call, step.tool_result.result if step.tool_result else "?")

            # Check if decision made
            if step.phase == AgentPhase.DECIDE and step.action_proposed:
                # Run TRT verification
                verified, verify_reasoning = self._verify_decision(
                    step.action_proposed, game_state, context
                )

                decision = AgentDecision(
                    action=step.action_proposed,
                    reasoning=self._compile_reasoning(),
                    confidence=self._estimate_confidence(),
                    verification_passed=verified,
                    steps=self.steps
                )
                logger.info("  Decision: %s (confidence=%.2f, verified=%s)",
                             decision.action, decision.confidence, verified)
                return decision

            # If tool was called, add result to prompt
            if step.tool_result:
                prompt = self._build_continuation_prompt(
                    prompt, step, game_state
                )

        # Max steps reached - force decision
        logger.warning("Agent reached max steps (%d), forcing decision", self.max_steps)
        return self._force_decision(game_state, context)

    def _build_initial_prompt(
        self,
        game_state: GameState,
        belief_state: Optional[BeliefState]
    ) -> str:
        """Build the initial prompt for the agent."""
        parts = []

        # System context
        parts.append("You are a poker agent making decisions in 6-max NLHE.")
        parts.append("Use the ReAct pattern: Think, then Act (call tools or decide).")
        parts.append("")

        # Tools available
        parts.append(self.tools.get_tools_prompt())

        # Game state
        parts.append("## Current Situation")
        parts.append(self._format_game_state(game_state))
        parts.append("")

        # Belief state (sorted by knowledge)
        if belief_state:
            parts.append("## Belief State")
            parts.append("(Sorted by knowledge: b+d. High d = strong evidence AGAINST)")
            parts.append(belief_state.to_agent_format())
            parts.append("")

        # Legal actions
        parts.append("## Legal Actions")
        for action in game_state.get_legal_actions():
            parts.append(f"  - {action}")
        parts.append("")

        # Instructions
        parts.append("## Instructions")
        parts.append("1. THINK: Analyze the situation")
        parts.append("2. ACT: Call a tool OR make a decision")
        parts.append("3. If calling a tool, format as: TOOL: tool_name(params)")
        parts.append("4. If deciding, format as: DECISION: action")
        parts.append("")
        parts.append("Begin your analysis:")

        return "\n".join(parts)

    def _format_game_state(self, gs: GameState) -> str:
        """Format game state for prompt."""
        lines = []
        lines.append(f"Street: {gs.street}")
        lines.append(f"Board: {gs.board}")
        lines.append(f"Pot: {gs.pot.total:.0f}")
        lines.append(f"Hero: {gs.hero.position} with {gs.hero.hole_cards or '??'}")
        lines.append(f"Hero stack: {gs.hero.stack:.0f}")
        lines.append(f"To call: {gs.to_call:.0f}")
        lines.append(f"Pot odds: {gs.pot_odds:.1%}")
        lines.append(f"SPR: {gs.spr:.1f}")

        if gs.villains:
            lines.append("Villains in hand:")
            for v in gs.villains:
                if v.is_in_hand:
                    lines.append(f"  {v.position}: stack {v.stack:.0f}, invested {v.total_invested:.0f}")

        return "\n".join(lines)

    def _parse_response(self, response: str, context: ToolContext) -> ReasoningStep:
        """Parse LLM response into a reasoning step."""
        response = response.strip()

        # Check for tool call
        if "TOOL:" in response:
            return self._parse_tool_call(response, context)

        # Check for decision
        if "DECISION:" in response:
            return self._parse_decision(response)

        # Just reasoning
        return ReasoningStep(
            phase=AgentPhase.REASON,
            thought=response
        )

    def _parse_tool_call(self, response: str, context: ToolContext) -> ReasoningStep:
        """Parse a tool call from response."""
        # Extract thought and tool call
        parts = response.split("TOOL:")
        thought = parts[0].strip()
        tool_str = parts[1].strip() if len(parts) > 1 else ""

        # Parse tool name and params
        # Format: tool_name(param1=value1, param2=value2)
        try:
            if "(" in tool_str:
                tool_name = tool_str[:tool_str.index("(")].strip()
                params_str = tool_str[tool_str.index("(")+1:tool_str.rindex(")")]
                params = self._parse_params(params_str)
            else:
                tool_name = tool_str.strip()
                params = {}

            tool_enum = ToolName(tool_name)
            tool_call = ToolCall(tool=tool_enum, params=params, reasoning=thought)

            # Execute tool
            result = self.tools.execute(tool_call, context)

            return ReasoningStep(
                phase=AgentPhase.OBSERVE,
                thought=thought,
                tool_call=tool_call,
                tool_result=result
            )
        except Exception as e:
            import sys
            print(f"[ReActAgent] Tool parse error: {e}", file=sys.stderr)
            return ReasoningStep(
                phase=AgentPhase.REASON,
                thought=f"{thought}\n[Tool parse error: {e}]"
            )

    def _parse_params(self, params_str: str) -> dict:
        """Parse parameter string into dict, respecting quoted strings."""
        params = {}
        if not params_str.strip():
            return params

        # Split on commas that are NOT inside quotes
        parts = []
        current = []
        in_quote = None
        for ch in params_str:
            if ch in ('"', "'") and in_quote is None:
                in_quote = ch
                current.append(ch)
            elif ch == in_quote:
                in_quote = None
                current.append(ch)
            elif ch == ',' and in_quote is None:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                # Try to parse as number
                try:
                    value = float(value)
                except ValueError:
                    pass
                params[key] = value

        return params

    def _parse_decision(self, response: str) -> ReasoningStep:
        """Parse a decision from response."""
        parts = response.split("DECISION:")
        thought = parts[0].strip()
        decision_str = parts[1].strip().lower() if len(parts) > 1 else ""

        # Parse action
        action = None
        if "fold" in decision_str:
            action = Action.fold()
        elif "check" in decision_str:
            action = Action.check()
        elif "call" in decision_str:
            # Extract amount if present
            amount = self._extract_amount(decision_str) or 0
            action = Action.call(amount)
        elif "raise" in decision_str or "bet" in decision_str:
            amount = self._extract_amount(decision_str) or 0
            if "raise" in decision_str:
                action = Action.raise_to(amount)
            else:
                action = Action.bet(amount)
        elif "all" in decision_str and "in" in decision_str:
            amount = self._extract_amount(decision_str) or 0
            action = Action.all_in(amount)

        return ReasoningStep(
            phase=AgentPhase.DECIDE,
            thought=thought,
            action_proposed=action
        )

    def _extract_amount(self, s: str) -> Optional[float]:
        """Extract numeric amount from string."""
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', s)
        if match:
            return float(match.group(1))
        return None

    def _build_continuation_prompt(
        self,
        prev_prompt: str,
        step: ReasoningStep,
        game_state: GameState
    ) -> str:
        """Build continuation prompt after tool call."""
        parts = [prev_prompt, "", "---", ""]
        parts.append(f"Your thought: {step.thought}")

        if step.tool_result:
            parts.append(f"Tool result: {step.tool_result}")

        parts.append("")
        parts.append("Continue your analysis:")

        return "\n".join(parts)

    def _verify_decision(
        self,
        action: Action,
        game_state: GameState,
        context: ToolContext
    ) -> tuple[bool, str]:
        """
        TRT self-verification of the decision.

        Checks:
        1. GTO consistency (is this in range?)
        2. EV sanity check
        3. Belief consistency
        """
        verifications = []
        passed = True

        # 1. Check if action makes basic sense
        if action.action_type == ActionType.FOLD and game_state.to_call == 0:
            verifications.append("WARNING: Folding when can check for free")
            passed = False

        # 2. If we have equity info, verify EV
        # This would use cached tool results in a full implementation

        # 3. Check belief consistency
        # Strong disbelief in villain bluffing + we're folding = might be ok
        # Strong belief in villain bluffing + we're folding = inconsistent

        return passed, "\n".join(verifications) if verifications else "Verification passed"

    def _compile_reasoning(self) -> str:
        """Compile all reasoning steps into summary."""
        parts = []
        for i, step in enumerate(self.steps):
            parts.append(f"Step {i+1} ({step.phase.name}): {step.thought[:100]}...")
        return "\n".join(parts)

    def _estimate_confidence(self) -> float:
        """Estimate confidence based on reasoning process."""
        # Simple heuristic: more tool calls = more informed = higher confidence
        tool_calls = sum(1 for s in self.steps if s.tool_result)
        base_confidence = 0.5
        confidence = min(0.95, base_confidence + tool_calls * 0.1)
        return confidence

    def _force_decision(
        self,
        game_state: GameState,
        context: ToolContext
    ) -> AgentDecision:
        """Force a decision when max steps reached."""
        import sys
        print(f"[ReActAgent] WARNING: Max steps ({self.max_steps}) reached, forcing fallback decision", file=sys.stderr)
        # Default to check/call if cheap, fold otherwise
        if game_state.to_call == 0:
            action = Action.check()
        elif game_state.to_call <= game_state.table.big_blind:
            action = Action.call(game_state.to_call)
        else:
            action = Action.fold()

        return AgentDecision(
            action=action,
            reasoning="Max reasoning steps reached, defaulting to safe action",
            confidence=0.3,
            verification_passed=False,
            steps=self.steps
        )

    def _mock_llm_call(self, prompt: str) -> str:
        """
        Mock LLM call for testing.

        In production, this would call the actual LLM API.
        """
        # Simple rule-based mock for testing
        if "pot_odds" not in prompt.lower():
            return "Let me check the pot odds first.\n\nTOOL: pot_odds()"

        if "equity" not in prompt.lower():
            return "Now let me calculate my equity.\n\nTOOL: equity_calc()"

        if "gto" not in prompt.lower():
            return "Let me check GTO recommendation.\n\nTOOL: gto_advisor(situation='open')"

        # Make decision based on simple heuristics
        return """Based on my analysis:
- Pot odds require ~33% equity
- My hand has good equity
- GTO says this is a call

DECISION: call"""
