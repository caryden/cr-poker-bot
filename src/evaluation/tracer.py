"""
Trace capture for poker bot decisions.

Records the agent's reasoning process for analysis:
- Thoughts and observations
- Tool calls and results
- Final decisions with reasoning
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import json
import sqlite3
from pathlib import Path

from ..core.primitives import Position, Street, ActionType, Action, HoleCards
from ..core.game_state import GameState
from ..agent.tools import ToolCall, ToolResult, ToolName


@dataclass
class DecisionTrace:
    """
    Trace of a single decision point.

    Captures the full reasoning process from observation to action.
    """
    decision_id: str
    hand_id: str
    street: Street
    position: Position
    pot: float
    to_call: float
    stack: float

    # Agent reasoning
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)  # {tool, params, result}

    # Decision
    action: Optional[Action] = None
    reasoning: str = ""
    confidence: float = 0.0

    # Context
    hole_cards: Optional[str] = None
    board: Optional[str] = None
    villain_actions: list[str] = field(default_factory=list)

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def add_thought(self, thought: str) -> None:
        """Record a thought."""
        self.thoughts.append(thought)

    def add_tool_call(self, call: ToolCall, result: ToolResult) -> None:
        """Record a tool call and its result."""
        self.tool_calls.append({
            'tool': call.tool.value,
            'params': call.params,
            'reasoning': call.reasoning,
            'success': result.success,
            'result': result.formatted,
            'error': result.error
        })

    def set_decision(self, action: Action, reasoning: str, confidence: float = 0.0) -> None:
        """Record the final decision."""
        self.action = action
        self.reasoning = reasoning
        self.confidence = confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'decision_id': self.decision_id,
            'hand_id': self.hand_id,
            'street': self.street.name,
            'position': self.position.value,
            'pot': self.pot,
            'to_call': self.to_call,
            'stack': self.stack,
            'thoughts': self.thoughts,
            'tool_calls': self.tool_calls,
            'action': self.action.action_type.value if self.action else None,
            'action_amount': self.action.amount if self.action else None,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'hole_cards': self.hole_cards,
            'board': self.board,
            'villain_actions': self.villain_actions,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"=== Decision {self.decision_id} ===",
            f"Street: {self.street.name} | Position: {self.position.value}",
            f"Pot: {self.pot:.0f} | To call: {self.to_call:.0f} | Stack: {self.stack:.0f}",
            f"Cards: {self.hole_cards} | Board: {self.board or 'preflop'}",
            "",
            "Thoughts:"
        ]
        for t in self.thoughts:
            lines.append(f"  - {t}")

        if self.tool_calls:
            lines.append("")
            lines.append("Tool calls:")
            for tc in self.tool_calls:
                status = "OK" if tc['success'] else "FAIL"
                lines.append(f"  [{status}] {tc['tool']}: {tc['result'][:100]}...")

        lines.append("")
        lines.append(f"Decision: {self.action.action_type.value if self.action else 'None'}")
        lines.append(f"Reasoning: {self.reasoning}")

        return "\n".join(lines)


@dataclass
class HandTrace:
    """
    Trace of a complete hand.

    Contains all decisions made during the hand and the outcome.
    """
    hand_id: str
    hero_position: Position
    hero_cards: str

    decisions: list[DecisionTrace] = field(default_factory=list)

    # Outcome
    profit_bb: float = 0.0
    went_to_showdown: bool = False
    won_at_showdown: bool = False
    final_board: str = ""

    # Summary stats
    vpip: bool = False  # Voluntarily put money in pot
    pfr: bool = False   # Preflop raise

    timestamp: datetime = field(default_factory=datetime.now)

    def add_decision(self, decision: DecisionTrace) -> None:
        """Add a decision trace."""
        self.decisions.append(decision)

        # Track VPIP/PFR
        if decision.street == Street.PREFLOP and decision.action:
            if decision.action.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.BET):
                self.vpip = True
            if decision.action.action_type in (ActionType.RAISE, ActionType.BET):
                self.pfr = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'hand_id': self.hand_id,
            'hero_position': self.hero_position.value,
            'hero_cards': self.hero_cards,
            'decisions': [d.to_dict() for d in self.decisions],
            'profit_bb': self.profit_bb,
            'went_to_showdown': self.went_to_showdown,
            'won_at_showdown': self.won_at_showdown,
            'final_board': self.final_board,
            'vpip': self.vpip,
            'pfr': self.pfr,
            'timestamp': self.timestamp.isoformat()
        }

    def summary(self) -> str:
        """Get hand summary."""
        result = "WON" if self.profit_bb > 0 else "LOST" if self.profit_bb < 0 else "PUSH"
        lines = [
            f"=== Hand {self.hand_id} ===",
            f"Position: {self.hero_position.value} | Cards: {self.hero_cards}",
            f"Result: {result} {self.profit_bb:+.1f} BB",
            f"Board: {self.final_board or 'N/A'}",
            f"Showdown: {'Yes' if self.went_to_showdown else 'No'}",
            f"VPIP: {'Yes' if self.vpip else 'No'} | PFR: {'Yes' if self.pfr else 'No'}",
            f"Decisions: {len(self.decisions)}"
        ]
        return "\n".join(lines)


@dataclass
class SessionTrace:
    """
    Trace of a complete session.
    """
    session_id: str
    opponent_types: list[str]

    hands: list[HandTrace] = field(default_factory=list)

    # Aggregate stats
    total_hands: int = 0
    total_profit_bb: float = 0.0

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_hand(self, hand: HandTrace) -> None:
        """Add a hand trace."""
        self.hands.append(hand)
        self.total_hands += 1
        self.total_profit_bb += hand.profit_bb

    def finalize(self) -> None:
        """Mark session complete."""
        self.end_time = datetime.now()

    @property
    def bb_per_100(self) -> float:
        """Calculate win rate."""
        if self.total_hands == 0:
            return 0.0
        return (self.total_profit_bb / self.total_hands) * 100

    @property
    def vpip_pct(self) -> float:
        """Calculate VPIP percentage."""
        if not self.hands:
            return 0.0
        return sum(1 for h in self.hands if h.vpip) / len(self.hands)

    @property
    def pfr_pct(self) -> float:
        """Calculate PFR percentage."""
        if not self.hands:
            return 0.0
        return sum(1 for h in self.hands if h.pfr) / len(self.hands)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'opponent_types': self.opponent_types,
            'hands': [h.to_dict() for h in self.hands],
            'total_hands': self.total_hands,
            'total_profit_bb': self.total_profit_bb,
            'bb_per_100': self.bb_per_100,
            'vpip_pct': self.vpip_pct,
            'pfr_pct': self.pfr_pct,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

    def summary(self) -> str:
        """Get session summary."""
        duration = ""
        if self.end_time:
            dur = (self.end_time - self.start_time).total_seconds()
            duration = f" ({dur:.1f}s)"

        lines = [
            f"=== Session {self.session_id} ===",
            f"Opponents: {', '.join(self.opponent_types)}",
            f"Hands: {self.total_hands}{duration}",
            f"Profit: {self.total_profit_bb:+.1f} BB",
            f"Win rate: {self.bb_per_100:+.1f} BB/100",
            f"VPIP: {self.vpip_pct:.1%} | PFR: {self.pfr_pct:.1%}"
        ]
        return "\n".join(lines)


class Tracer:
    """
    Active tracer for recording decisions.

    Use as context manager during hand/session execution.
    """

    def __init__(self, session_id: str, opponent_types: list[str]):
        self.session = SessionTrace(
            session_id=session_id,
            opponent_types=opponent_types
        )
        self._current_hand: Optional[HandTrace] = None
        self._current_decision: Optional[DecisionTrace] = None
        self._decision_counter = 0

    def start_hand(self, hand_id: str, position: Position, cards: HoleCards) -> HandTrace:
        """Start tracing a new hand."""
        self._current_hand = HandTrace(
            hand_id=hand_id,
            hero_position=position,
            hero_cards=cards.notation
        )
        return self._current_hand

    def start_decision(self, game_state: GameState) -> DecisionTrace:
        """Start tracing a decision point."""
        self._decision_counter += 1
        self._current_decision = DecisionTrace(
            decision_id=f"{self._current_hand.hand_id}_{self._decision_counter}",
            hand_id=self._current_hand.hand_id,
            street=game_state.street,
            position=game_state.hero.position,
            pot=game_state.pot.total,
            to_call=game_state.to_call,
            stack=game_state.hero.stack,
            hole_cards=game_state.hero.hole_cards.notation if game_state.hero.hole_cards else None,
            board=str(game_state.board) if game_state.board.cards else None
        )
        return self._current_decision

    def record_thought(self, thought: str) -> None:
        """Record a thought during decision."""
        if self._current_decision:
            self._current_decision.add_thought(thought)

    def record_tool_call(self, call: ToolCall, result: ToolResult) -> None:
        """Record a tool call."""
        if self._current_decision:
            self._current_decision.add_tool_call(call, result)

    def record_decision(self, action: Action, reasoning: str, confidence: float = 0.0) -> None:
        """Record the final decision."""
        if self._current_decision:
            self._current_decision.set_decision(action, reasoning, confidence)
            self._current_hand.add_decision(self._current_decision)
            self._current_decision = None

    def end_hand(self, profit_bb: float, went_to_sd: bool, won_sd: bool, board: str) -> None:
        """Finish tracing a hand."""
        if self._current_hand:
            self._current_hand.profit_bb = profit_bb
            self._current_hand.went_to_showdown = went_to_sd
            self._current_hand.won_at_showdown = won_sd
            self._current_hand.final_board = board
            self.session.add_hand(self._current_hand)
            self._current_hand = None

    def finalize(self) -> SessionTrace:
        """Finalize and return the session trace."""
        self.session.finalize()
        return self.session


class TraceStore:
    """
    Persistent storage for traces.

    Supports both SQLite and JSON export.
    """

    def __init__(self, db_path: str = "traces.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    opponent_types TEXT,
                    total_hands INTEGER,
                    total_profit_bb REAL,
                    bb_per_100 REAL,
                    vpip_pct REAL,
                    pfr_pct REAL,
                    start_time TEXT,
                    end_time TEXT,
                    data JSON
                );

                CREATE TABLE IF NOT EXISTS hands (
                    hand_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    hero_position TEXT,
                    hero_cards TEXT,
                    profit_bb REAL,
                    went_to_showdown INTEGER,
                    won_at_showdown INTEGER,
                    vpip INTEGER,
                    pfr INTEGER,
                    num_decisions INTEGER,
                    timestamp TEXT,
                    data JSON,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    hand_id TEXT,
                    street TEXT,
                    position TEXT,
                    pot REAL,
                    to_call REAL,
                    action TEXT,
                    action_amount REAL,
                    num_tool_calls INTEGER,
                    reasoning TEXT,
                    data JSON,
                    FOREIGN KEY (hand_id) REFERENCES hands(hand_id)
                );

                CREATE INDEX IF NOT EXISTS idx_hands_session ON hands(session_id);
                CREATE INDEX IF NOT EXISTS idx_decisions_hand ON decisions(hand_id);
                CREATE INDEX IF NOT EXISTS idx_hands_profit ON hands(profit_bb);
            """)

    def save_session(self, session: SessionTrace) -> None:
        """Save a session trace to the database."""
        with sqlite3.connect(self.db_path) as conn:
            # Save session
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, opponent_types, total_hands, total_profit_bb,
                 bb_per_100, vpip_pct, pfr_pct, start_time, end_time, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                json.dumps(session.opponent_types),
                session.total_hands,
                session.total_profit_bb,
                session.bb_per_100,
                session.vpip_pct,
                session.pfr_pct,
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                json.dumps(session.to_dict())
            ))

            # Save hands
            for hand in session.hands:
                conn.execute("""
                    INSERT OR REPLACE INTO hands
                    (hand_id, session_id, hero_position, hero_cards, profit_bb,
                     went_to_showdown, won_at_showdown, vpip, pfr, num_decisions,
                     timestamp, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hand.hand_id,
                    session.session_id,
                    hand.hero_position.value,
                    hand.hero_cards,
                    hand.profit_bb,
                    int(hand.went_to_showdown),
                    int(hand.won_at_showdown),
                    int(hand.vpip),
                    int(hand.pfr),
                    len(hand.decisions),
                    hand.timestamp.isoformat(),
                    json.dumps(hand.to_dict())
                ))

                # Save decisions
                for decision in hand.decisions:
                    conn.execute("""
                        INSERT OR REPLACE INTO decisions
                        (decision_id, hand_id, street, position, pot, to_call,
                         action, action_amount, num_tool_calls, reasoning, data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        decision.decision_id,
                        decision.hand_id,
                        decision.street.name,
                        decision.position.value,
                        decision.pot,
                        decision.to_call,
                        decision.action.action_type.value if decision.action else None,
                        decision.action.amount if decision.action else None,
                        len(decision.tool_calls),
                        decision.reasoning,
                        json.dumps(decision.to_dict())
                    ))

    def load_session(self, session_id: str) -> Optional[dict]:
        """Load a session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def list_sessions(self) -> list[dict]:
        """List all sessions with summary stats."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT session_id, opponent_types, total_hands,
                       total_profit_bb, bb_per_100, start_time
                FROM sessions
                ORDER BY start_time DESC
            """).fetchall()

            return [
                {
                    'session_id': r[0],
                    'opponent_types': json.loads(r[1]),
                    'total_hands': r[2],
                    'total_profit_bb': r[3],
                    'bb_per_100': r[4],
                    'start_time': r[5]
                }
                for r in rows
            ]

    def get_losing_hands(self, session_id: str, limit: int = 10) -> list[dict]:
        """Get biggest losing hands for review."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT data FROM hands
                WHERE session_id = ? AND profit_bb < 0
                ORDER BY profit_bb ASC
                LIMIT ?
            """, (session_id, limit)).fetchall()

            return [json.loads(r[0]) for r in rows]

    def get_decisions_by_street(self, session_id: str, street: str) -> list[dict]:
        """Get all decisions for a specific street."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT d.data FROM decisions d
                JOIN hands h ON d.hand_id = h.hand_id
                WHERE h.session_id = ? AND d.street = ?
            """, (session_id, street)).fetchall()

            return [json.loads(r[0]) for r in rows]

    def export_json(self, session_id: str, filepath: str) -> None:
        """Export session to JSON file."""
        session_data = self.load_session(session_id)
        if session_data:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
