"""
Prompt templates for the poker agent.

Provides structured prompts for different situations including:
- System prompt with tool descriptions
- Situation-specific prompts
- Belief state formatting
"""

from typing import Optional
from string import Template

from ..core.primitives import Position, Street
from ..core.game_state import GameState
from ..core.beliefs import BeliefState, sort_beliefs_by_knowledge
from ..agent.tools import ToolRegistry


# ============== System Prompt ==============

SYSTEM_PROMPT = """You are an expert poker agent playing 6-max No-Limit Texas Hold'em (NLHE).

## Your Capabilities
- Analyze game situations mathematically
- Use tools to gather information (equity, pot odds, GTO advice)
- Track beliefs about opponents using Subjective Logic
- Make decisions using the ReAct pattern (Reasoning + Acting)

## Decision Framework
1. OBSERVE: Gather information using tools
2. REASON: Analyze the situation considering:
   - Pot odds and equity
   - Position and stack depths
   - Opponent tendencies (from beliefs)
   - GTO baseline and deviations
3. VERIFY: Check your reasoning for consistency
4. DECIDE: Choose an action with clear justification

## Understanding Beliefs (Subjective Logic)
Beliefs are expressed as (b, d, u) tuples:
- b = belief (evidence FOR)
- d = disbelief (evidence AGAINST)
- u = uncertainty (lack of evidence)
- Constraint: b + d + u = 1

**Key insight**: High disbelief (d) is highly valuable:
- It represents strong evidence that something is NOT true
- Use it to prune your search space
- "Villain does NOT bluff often" (high d) means don't call light

Beliefs are sorted by **knowledge** (b + d). Higher knowledge = more actionable.

## Output Format
Think step by step, then either:
1. Call a tool: `TOOL: tool_name(param1=value1, param2=value2)`
2. Make decision: `DECISION: action [amount]`

{tools_section}
"""


# ============== Situation Templates ==============

PREFLOP_TEMPLATE = Template("""## Preflop Situation

**Position**: ${hero_position}
**Hand**: ${hero_hand}
**Stack**: ${hero_stack} BB
**Pot**: ${pot_size} BB
**Action**: ${action_description}

**Villains**:
${villains_info}

${belief_section}

**Your options**:
${legal_actions}

Analyze this preflop spot. Consider your position, hand strength, and opponent tendencies.""")


POSTFLOP_TEMPLATE = Template("""## Postflop Situation

**Street**: ${street}
**Board**: ${board}
**Position**: ${hero_position}
**Hand**: ${hero_hand} (${hand_description})
**Stack**: ${hero_stack} BB
**Pot**: ${pot_size} BB
**To Call**: ${to_call} BB
**SPR**: ${spr}

**Villains**:
${villains_info}

${belief_section}

**Your options**:
${legal_actions}

Analyze this postflop spot. Consider board texture, position, and opponent tendencies.""")


DECISION_VERIFICATION_TEMPLATE = Template("""## Verify Your Decision

You are about to: **${proposed_action}**

Before committing, verify:

1. **Math Check**:
   - Pot odds: ${pot_odds}
   - Required equity: ${required_equity}
   - Your estimated equity: ${equity_estimate}
   - Is the math favorable? ${math_check}

2. **GTO Check**:
   - GTO recommendation: ${gto_recommendation}
   - Your action ${gto_deviation}

3. **Belief Check**:
   - Relevant beliefs: ${relevant_beliefs}
   - Consistent with beliefs? ${belief_check}

4. **Risk Check**:
   - Stack commitment: ${stack_commitment}
   - Risk level: ${risk_level}

If all checks pass, confirm with: `CONFIRM: ${proposed_action}`
Otherwise, reconsider with: `RECONSIDER: [reason]`""")


# ============== Helper Functions ==============

def format_villains(game_state: GameState, bb: float) -> str:
    """Format villain info for prompts."""
    lines = []
    for villain in game_state.villains:
        if villain.is_in_hand:
            stack_bb = villain.stack / bb
            invested_bb = villain.total_invested / bb
            lines.append(f"  {villain.position}: {stack_bb:.0f} BB, invested {invested_bb:.1f} BB")
    return "\n".join(lines) if lines else "  (none active)"


def format_beliefs(belief_state: Optional[BeliefState]) -> str:
    """Format beliefs for prompts."""
    if belief_state is None:
        return "(No belief state available)"

    lines = ["**Beliefs (sorted by knowledge)**:"]

    for villain_id, villain_beliefs in belief_state.villain_beliefs.items():
        lines.append(f"\n{villain_id}:")
        sorted_beliefs = villain_beliefs.sorted_by_knowledge()[:5]  # Top 5
        for belief in sorted_beliefs:
            op = belief.opinion
            lines.append(f"  {belief.label}: (b={op.belief:.2f}, d={op.disbelief:.2f}, u={op.uncertainty:.2f})")

    return "\n".join(lines)


def format_legal_actions(game_state: GameState) -> str:
    """Format legal actions."""
    actions = game_state.get_legal_actions()
    return "\n".join(f"  - {action}" for action in actions)


class PromptBuilder:
    """
    Builds prompts for different poker situations.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.tools = tool_registry or ToolRegistry()

    def build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        tools_section = self.tools.get_tools_prompt()
        return SYSTEM_PROMPT.format(tools_section=tools_section)

    def build_situation_prompt(
        self,
        game_state: GameState,
        belief_state: Optional[BeliefState] = None
    ) -> str:
        """Build prompt for current situation."""
        bb = game_state.table.big_blind
        hero = game_state.hero

        # Common values
        values = {
            'hero_position': str(hero.position),
            'hero_hand': hero.hole_cards.notation if hero.hole_cards else '??',
            'hero_stack': f"{hero.stack / bb:.0f}",
            'pot_size': f"{game_state.pot.total / bb:.1f}",
            'villains_info': format_villains(game_state, bb),
            'belief_section': format_beliefs(belief_state),
            'legal_actions': format_legal_actions(game_state),
        }

        if game_state.street == Street.PREFLOP:
            # Describe preflop action
            if game_state.to_call <= bb:
                action_desc = "You can open or complete"
            elif game_state.to_call <= bb * 3:
                action_desc = "Facing a raise"
            else:
                action_desc = "Facing a 3-bet or larger"

            values['action_description'] = action_desc
            return PREFLOP_TEMPLATE.substitute(values)
        else:
            # Postflop
            values['street'] = str(game_state.street)
            values['board'] = str(game_state.board)
            values['to_call'] = f"{game_state.to_call / bb:.1f}"
            values['spr'] = f"{game_state.spr:.1f}"
            values['hand_description'] = self._describe_hand(game_state)
            return POSTFLOP_TEMPLATE.substitute(values)

    def build_verification_prompt(
        self,
        game_state: GameState,
        proposed_action: str,
        equity_estimate: float = 0.5
    ) -> str:
        """Build verification prompt."""
        bb = game_state.table.big_blind
        pot_odds = game_state.pot_odds
        required_eq = pot_odds

        values = {
            'proposed_action': proposed_action,
            'pot_odds': f"{pot_odds:.1%}",
            'required_equity': f"{required_eq:.1%}",
            'equity_estimate': f"{equity_estimate:.1%}",
            'math_check': "YES" if equity_estimate > required_eq else "NO - RECONSIDER",
            'gto_recommendation': "[use GTO tool]",
            'gto_deviation': "matches GTO" if True else "deviates from GTO",
            'relevant_beliefs': "[query beliefs]",
            'belief_check': "consistent",
            'stack_commitment': f"{game_state.to_call / game_state.hero.stack:.0%}" if game_state.hero.stack > 0 else "N/A",
            'risk_level': self._assess_risk(game_state),
        }

        return DECISION_VERIFICATION_TEMPLATE.substitute(values)

    def _describe_hand(self, game_state: GameState) -> str:
        """Describe hero's hand strength."""
        if game_state.hero.hole_cards is None or len(game_state.board) < 3:
            return "unknown"

        from ..tools.hand_eval import evaluate_hand
        try:
            eval_ = evaluate_hand(game_state.hero.hole_cards, game_state.board)
            return eval_.description
        except:
            return "evaluation failed"

    def _summarize_situation(self, game_state: GameState) -> str:
        """Brief situation summary."""
        bb = game_state.table.big_blind
        return (f"{game_state.street}, {game_state.hero.position}, "
                f"pot {game_state.pot.total/bb:.0f} BB, "
                f"to call {game_state.to_call/bb:.0f} BB")

    def _assess_risk(self, game_state: GameState) -> str:
        """Assess risk level of current decision."""
        if game_state.hero.stack == 0:
            return "ALL-IN"
        commitment = game_state.to_call / game_state.hero.stack
        if commitment > 0.5:
            return "HIGH"
        elif commitment > 0.2:
            return "MEDIUM"
        elif commitment > 0:
            return "LOW"
        return "NONE"
