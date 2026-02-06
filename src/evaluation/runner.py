"""
Evaluation runner for poker bot testing.

Provides a high-level interface for running evaluations with trace capture.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
import uuid

from ..core.primitives import Position
from ..game.runner import SimplePokerEngine, HandRunner, SessionRunner, SessionStats
from ..game.opponents import create_opponent
from .tracer import Tracer, TraceStore, SessionTrace
from .analyzer import SessionAnalyzer, PerformanceMetrics


@dataclass
class EvaluationConfig:
    """
    Configuration for an evaluation run.
    """
    # Session parameters
    num_hands: int = 1000
    opponent_types: list[str] = field(default_factory=lambda: ["fish", "tag"])

    # Stakes
    small_blind: float = 0.5
    big_blind: float = 1.0
    starting_stack: float = 100.0  # In BB

    # Trace settings
    capture_traces: bool = True
    db_path: str = "evaluation_traces.db"

    # Verbosity
    verbose: bool = False
    progress_interval: int = 100  # Print progress every N hands


@dataclass
class EvaluationResult:
    """
    Results from an evaluation run.
    """
    session_id: str
    config: EvaluationConfig
    metrics: PerformanceMetrics
    session_trace: Optional[SessionTrace] = None

    # Summary
    total_hands: int = 0
    total_profit_bb: float = 0.0
    duration_seconds: float = 0.0

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "EVALUATION RESULT",
            "=" * 60,
            f"Session: {self.session_id}",
            f"Hands: {self.total_hands}",
            f"Duration: {self.duration_seconds:.1f}s ({self.total_hands / max(self.duration_seconds, 1):.1f} hands/sec)",
            "",
            f"Opponents: {', '.join(self.config.opponent_types)}",
            f"Stakes: {self.config.small_blind}/{self.config.big_blind}",
            "",
            str(self.metrics)
        ]
        return "\n".join(lines)


class EvaluationRunner:
    """
    Runs poker bot evaluations with full trace capture.

    Example usage:
        config = EvaluationConfig(num_hands=500, opponent_types=["fish", "lag"])
        runner = EvaluationRunner(hero_agent, config)
        result = runner.run()
        print(result)
        print(runner.analyzer.generate_report())
    """

    def __init__(
        self,
        hero,  # PlayerProtocol
        config: EvaluationConfig,
        on_hand_complete: Optional[Callable] = None
    ):
        self.hero = hero
        self.config = config
        self.on_hand_complete = on_hand_complete

        self.session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Set up tracing
        self.tracer = Tracer(self.session_id, config.opponent_types) if config.capture_traces else None
        self.trace_store = TraceStore(config.db_path) if config.capture_traces else None

        # Results
        self.result: Optional[EvaluationResult] = None
        self.analyzer: Optional[SessionAnalyzer] = None

    def run(self) -> EvaluationResult:
        """Run the full evaluation."""
        start_time = datetime.now()

        # Create opponents
        villains = {}
        for i, opp_type in enumerate(self.config.opponent_types):
            player_id = f"villain_{i+1}"
            villains[player_id] = create_opponent(opp_type, player_id)

        # Run session
        session_runner = SessionRunner(
            self.hero,
            villains,
            big_blind=self.config.big_blind,
            starting_stack=self.config.starting_stack * self.config.big_blind
        )
        stats = session_runner.run_session(
            num_hands=self.config.num_hands,
            verbose=self.config.verbose
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Create basic trace from session runner data if tracing enabled
        if self.tracer:
            self._build_trace_from_history(session_runner)
            session_trace = self.tracer.finalize()

            # Save to store
            if self.trace_store:
                self.trace_store.save_session(session_trace)

            # Set up analyzer
            self.analyzer = SessionAnalyzer(session_trace)
            metrics = self.analyzer.compute_metrics()
        else:
            session_trace = None
            metrics = self._stats_to_metrics(stats)

        # Build result
        self.result = EvaluationResult(
            session_id=self.session_id,
            config=self.config,
            metrics=metrics,
            session_trace=session_trace,
            total_hands=stats.hands_played,
            total_profit_bb=stats.total_profit,
            duration_seconds=duration
        )

        return self.result

    def _build_trace_from_history(self, session_runner: SessionRunner) -> None:
        """Build trace from session runner hand history."""
        # Note: Full tracing would require instrumenting the hand runner
        # This is a simplified version using available data
        for i, result in enumerate(session_runner.hand_history):
            # Create hand trace
            hand_trace = self.tracer.start_hand(
                hand_id=result.hand_id,
                position=result.hero_position,
                cards=result.hero_cards
            )

            # End hand with result
            self.tracer.end_hand(
                profit_bb=result.hero_profit,
                went_to_sd=result.went_to_showdown,
                won_sd=result.hero_profit > 0 and result.went_to_showdown,
                board=""  # Board info not available in HandResult
            )

    def _stats_to_metrics(self, stats: SessionStats) -> PerformanceMetrics:
        """Convert SessionStats to PerformanceMetrics."""
        return PerformanceMetrics(
            total_hands=stats.hands_played,
            total_profit_bb=stats.total_profit,
            bb_per_100=stats.bb_per_100,
            vpip=stats.vpip,
            pfr=stats.pfr,
            wtsd=stats.wtsd,
            won_at_sd=stats.won_at_sd
        )

    def get_report(self) -> str:
        """Get detailed analysis report."""
        if self.analyzer:
            return self.analyzer.generate_report()
        elif self.result:
            return str(self.result)
        return "No evaluation run yet"

    def export_traces(self, filepath: str) -> None:
        """Export session traces to JSON file."""
        if self.trace_store and self.session_id:
            self.trace_store.export_json(self.session_id, filepath)


def quick_eval(
    hero,
    opponent_type: str = "fish",
    num_hands: int = 100,
    verbose: bool = True
) -> EvaluationResult:
    """
    Quick evaluation helper for testing.

    Args:
        hero: The agent to test
        opponent_type: Single opponent type
        num_hands: Number of hands to play
        verbose: Print progress

    Returns:
        EvaluationResult
    """
    config = EvaluationConfig(
        num_hands=num_hands,
        opponent_types=[opponent_type],
        capture_traces=False,  # Skip traces for quick eval
        verbose=verbose
    )
    runner = EvaluationRunner(hero, config)
    return runner.run()


def full_eval(
    hero,
    opponent_types: list[str] = None,
    num_hands: int = 1000,
    db_path: str = "evaluation_traces.db"
) -> EvaluationResult:
    """
    Full evaluation with trace capture.

    Args:
        hero: The agent to test
        opponent_types: List of opponent types
        num_hands: Number of hands to play
        db_path: Path to trace database

    Returns:
        EvaluationResult with full traces
    """
    if opponent_types is None:
        opponent_types = ["fish", "nit", "lag", "tag"]

    config = EvaluationConfig(
        num_hands=num_hands,
        opponent_types=opponent_types,
        capture_traces=True,
        db_path=db_path,
        verbose=True,
        progress_interval=100
    )
    runner = EvaluationRunner(hero, config)
    result = runner.run()

    # Print report
    print(runner.get_report())

    return result
