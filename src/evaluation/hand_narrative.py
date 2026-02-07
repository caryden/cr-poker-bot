"""
Interspersed hand narrative that combines game actions with LLM agent reasoning.

Produces a human-readable timeline of a poker hand where each game action
(PHH-style) is shown alongside the hero agent's thought process when it is
the hero's turn to act.

Example output:

    === Hand abc123 | Hero: BTN with AhKd ===

    -- PREFLOP (pot 15) --
    UTG  folds
    HJ   folds
    CO   raises to 25

    >>> HERO TURN (BTN) | pot 40 | to_call 25 | stack 990 <<<
    [Prompt] Position: BTN, Hand: AKo, Pot: 40, To Call: 25 ...
    [Think]  AKo is a premium hand on the BTN facing a CO open...
    [Tool]   equity_calc -> 62.3% equity vs CO range
    [Tool]   pot_odds -> need 38.5% equity, we have 62.3%
    [Think]  Clear 3-bet for value...
    [Decision] RAISE to 75 (confidence: 0.85, verified: True)

    CO   calls 75
    SB   folds
    BB   folds

    -- FLOP: Ah 7c 2d (pot 165) --
    ...
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from pathlib import Path

from ..core.primitives import Action, ActionType, Position, Street, HoleCards
from ..core.game_state import GameState, ActionRecord
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HeroThought:
    """A single thought/step in the hero's reasoning."""
    kind: str  # "prompt", "think", "tool", "decision"
    content: str


@dataclass
class HeroDecisionNarrative:
    """The hero's full reasoning for one decision point."""
    street: Street
    position: Position
    pot: float
    to_call: float
    stack: float
    hole_cards: str
    board: str
    thoughts: list[HeroThought] = field(default_factory=list)
    action: Optional[Action] = None
    confidence: float = 0.0
    verified: bool = False

    def add_prompt_summary(self, prompt: str) -> None:
        """Record a summary of what the hero was shown."""
        # Truncate prompt to first meaningful lines
        lines = prompt.strip().split('\n')
        summary = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                summary.append(line)
            if len(summary) >= 3:
                break
        self.thoughts.append(HeroThought("prompt", " | ".join(summary)))

    def add_thought(self, text: str) -> None:
        self.thoughts.append(HeroThought("think", text))

    def add_tool_call(self, tool_name: str, result_summary: str) -> None:
        self.thoughts.append(HeroThought("tool", f"{tool_name} -> {result_summary}"))

    def set_decision(self, action: Action, confidence: float, verified: bool) -> None:
        self.action = action
        self.confidence = confidence
        self.verified = verified
        self.thoughts.append(HeroThought(
            "decision",
            f"{action} (confidence: {confidence:.2f}, verified: {verified})"
        ))

    def render(self, indent: str = "    ") -> str:
        """Render the hero decision narrative as text."""
        lines = []
        board_str = f" | board {self.board}" if self.board else ""
        lines.append(
            f">>> HERO TURN ({self.position.value}) | "
            f"pot {self.pot:.0f} | to_call {self.to_call:.0f} | "
            f"stack {self.stack:.0f}{board_str} <<<"
        )
        for t in self.thoughts:
            tag = {
                "prompt": "Prompt",
                "think": "Think",
                "tool": "Tool",
                "decision": "Decision",
            }.get(t.kind, t.kind)
            lines.append(f"{indent}[{tag:>8s}] {t.content}")
        return "\n".join(lines)


@dataclass
class NarrativeEvent:
    """A single event in the hand timeline (either game action or hero reasoning)."""
    kind: str  # "action", "hero_decision", "street", "deal", "result"
    text: str
    order: int = 0


class HandNarrative:
    """
    Builds and renders a human-readable hand narrative.

    Interleaves game actions with hero's LLM reasoning trace.
    """

    def __init__(self, hand_id: str, hero_id: str, hero_position: Position,
                 hero_cards: HoleCards):
        self.hand_id = hand_id
        self.hero_id = hero_id
        self.hero_position = hero_position
        self.hero_cards = hero_cards
        self.events: list[NarrativeEvent] = []
        self._event_counter = 0
        self._current_hero_decision: Optional[HeroDecisionNarrative] = None
        self.profit_bb: Optional[float] = None
        self.timestamp = datetime.now()

    def _next_order(self) -> int:
        self._event_counter += 1
        return self._event_counter

    # -- Street transitions --

    def add_street(self, street: Street, board_str: str, pot: float) -> None:
        """Record a new street being dealt."""
        if street == Street.PREFLOP:
            text = f"-- PREFLOP (pot {pot:.0f}) --"
        else:
            text = f"-- {street.value.upper()}: {board_str} (pot {pot:.0f}) --"
        self.events.append(NarrativeEvent("street", text, self._next_order()))

    # -- Villain/generic actions --

    def add_action(self, player_id: str, position: Position, action: Action,
                   pot_after: float) -> None:
        """Record any player's action."""
        amount_str = ""
        if action.amount and action.action_type not in (ActionType.FOLD, ActionType.CHECK):
            amount_str = f" {action.amount:.0f}"
        text = f"{position.value:<4s} {action.action_type.value}{amount_str}"
        self.events.append(NarrativeEvent("action", text, self._next_order()))

    # -- Hero reasoning --

    def begin_hero_decision(self, game_state: GameState) -> HeroDecisionNarrative:
        """Start capturing the hero's decision-making process."""
        hero = game_state.hero
        self._current_hero_decision = HeroDecisionNarrative(
            street=game_state.street,
            position=hero.position,
            pot=game_state.pot.total,
            to_call=game_state.to_call,
            stack=hero.stack,
            hole_cards=hero.hole_cards.notation if hero.hole_cards else "??",
            board=str(game_state.board) if game_state.board.cards else "",
        )
        return self._current_hero_decision

    def end_hero_decision(self) -> None:
        """Finalize and embed the hero decision into the timeline."""
        if self._current_hero_decision:
            rendered = self._current_hero_decision.render()
            self.events.append(NarrativeEvent("hero_decision", rendered, self._next_order()))
            self._current_hero_decision = None

    # -- Hand result --

    def set_result(self, profit_bb: float, went_to_showdown: bool) -> None:
        self.profit_bb = profit_bb
        outcome = "WON" if profit_bb > 0 else ("LOST" if profit_bb < 0 else "SPLIT")
        sd_str = " (showdown)" if went_to_showdown else ""
        text = f"=== Result: {outcome} {profit_bb:+.1f} BB{sd_str} ==="
        self.events.append(NarrativeEvent("result", text, self._next_order()))

    # -- Rendering --

    def render(self) -> str:
        """Render the full hand narrative as human-readable text."""
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(
            f"Hand {self.hand_id} | Hero: {self.hero_position.value} "
            f"with {self.hero_cards} | {self.timestamp:%Y-%m-%d %H:%M:%S}"
        )
        lines.append(f"{'=' * 60}")
        lines.append("")

        for event in sorted(self.events, key=lambda e: e.order):
            if event.kind == "street":
                lines.append("")
                lines.append(event.text)
            elif event.kind == "hero_decision":
                lines.append("")
                lines.append(event.text)
                lines.append("")
            elif event.kind == "result":
                lines.append("")
                lines.append(event.text)
            else:
                lines.append(f"    {event.text}")

        lines.append("")
        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        """Save narrative to a text file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())
        logger.info("Hand narrative saved to %s", filepath)


class NarrativeTracer:
    """
    Session-level narrative tracer that produces interspersed hand narratives.

    Integrates with the HandRunner to capture game actions + hero reasoning
    in a unified timeline.
    """

    def __init__(self, output_dir: str = "traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.narratives: list[HandNarrative] = []
        self._current: Optional[HandNarrative] = None

    def start_hand(self, hand_id: str, hero_id: str, hero_position: Position,
                   hero_cards: HoleCards) -> HandNarrative:
        """Begin tracing a new hand."""
        narrative = HandNarrative(hand_id, hero_id, hero_position, hero_cards)
        narrative.add_street(Street.PREFLOP, "", 0)
        self._current = narrative
        self.narratives.append(narrative)
        return narrative

    @property
    def current(self) -> Optional[HandNarrative]:
        return self._current

    def end_hand(self, profit_bb: float, went_to_showdown: bool) -> None:
        """Finalize current hand narrative."""
        if self._current:
            self._current.set_result(profit_bb, went_to_showdown)
            # Auto-save individual hand narrative
            safe_id = self._current.hand_id.replace("/", "_")
            filepath = self.output_dir / f"hand_{safe_id}.txt"
            self._current.save(str(filepath))
            self._current = None

    def save_session_summary(self, session_id: str) -> str:
        """Save all hand narratives into a single session file."""
        filepath = self.output_dir / f"session_{session_id}.txt"
        parts = []
        total_profit = 0.0
        for n in self.narratives:
            parts.append(n.render())
            if n.profit_bb is not None:
                total_profit += n.profit_bb

        header = (
            f"Session {session_id} | {len(self.narratives)} hands | "
            f"Total: {total_profit:+.1f} BB\n"
            f"{'=' * 60}\n\n"
        )
        filepath.write_text(header + "\n".join(parts))
        logger.info("Session narrative saved to %s", filepath)
        return str(filepath)
