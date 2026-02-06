"""
Analysis tools for poker bot evaluation.

Provides metrics calculation and decision analysis.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from ..core.primitives import Position, Street, ActionType
from .tracer import SessionTrace, HandTrace, DecisionTrace


@dataclass
class PerformanceMetrics:
    """
    Aggregate performance metrics for a session or set of sessions.
    """
    # Sample size
    total_hands: int = 0
    total_decisions: int = 0

    # Win rate
    total_profit_bb: float = 0.0
    bb_per_100: float = 0.0
    std_dev_bb: float = 0.0

    # Preflop stats
    vpip: float = 0.0       # Voluntarily put money in pot
    pfr: float = 0.0        # Preflop raise
    three_bet: float = 0.0  # 3-bet percentage

    # Postflop stats
    cbet_flop: float = 0.0     # C-bet on flop
    cbet_turn: float = 0.0     # C-bet on turn
    fold_to_cbet: float = 0.0  # Fold to c-bet

    # Showdown stats
    wtsd: float = 0.0       # Went to showdown
    won_at_sd: float = 0.0  # Won money at showdown

    # By position
    profit_by_position: dict[str, float] = field(default_factory=dict)

    # By opponent type
    profit_by_opponent: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=== Performance Metrics ===",
            f"Hands: {self.total_hands} | Decisions: {self.total_decisions}",
            "",
            f"Win rate: {self.bb_per_100:+.1f} BB/100 (±{self.std_dev_bb:.1f})",
            f"Total profit: {self.total_profit_bb:+.1f} BB",
            "",
            "Preflop:",
            f"  VPIP: {self.vpip:.1%} | PFR: {self.pfr:.1%} | 3-Bet: {self.three_bet:.1%}",
            "",
            "Postflop:",
            f"  C-bet flop: {self.cbet_flop:.1%} | C-bet turn: {self.cbet_turn:.1%}",
            f"  Fold to c-bet: {self.fold_to_cbet:.1%}",
            "",
            "Showdown:",
            f"  WTSD: {self.wtsd:.1%} | W$SD: {self.won_at_sd:.1%}",
        ]

        if self.profit_by_position:
            lines.append("")
            lines.append("By position:")
            for pos, profit in sorted(self.profit_by_position.items()):
                lines.append(f"  {pos}: {profit:+.1f} BB")

        if self.profit_by_opponent:
            lines.append("")
            lines.append("By opponent type:")
            for opp, profit in sorted(self.profit_by_opponent.items()):
                lines.append(f"  {opp}: {profit:+.1f} BB")

        return "\n".join(lines)


@dataclass
class DecisionAnalysis:
    """
    Analysis of a specific decision.
    """
    decision: DecisionTrace
    classification: str = ""  # e.g., "good_value_bet", "missed_bluff", "bad_call"
    ev_estimate: float = 0.0
    suggested_action: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Decision: {self.decision.action.action_type.value if self.decision.action else 'None'}",
            f"Classification: {self.classification}",
            f"EV estimate: {self.ev_estimate:+.2f} BB",
        ]
        if self.suggested_action:
            lines.append(f"Suggested: {self.suggested_action}")
        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


class SessionAnalyzer:
    """
    Analyzes session traces to compute metrics and find leaks.
    """

    def __init__(self, session: SessionTrace):
        self.session = session

    def compute_metrics(self) -> PerformanceMetrics:
        """Compute aggregate performance metrics."""
        metrics = PerformanceMetrics()

        if not self.session.hands:
            return metrics

        metrics.total_hands = len(self.session.hands)
        metrics.total_profit_bb = self.session.total_profit_bb
        metrics.bb_per_100 = self.session.bb_per_100

        # Calculate standard deviation
        profits = [h.profit_bb for h in self.session.hands]
        if profits:
            mean = sum(profits) / len(profits)
            variance = sum((p - mean) ** 2 for p in profits) / len(profits)
            metrics.std_dev_bb = variance ** 0.5

        # Preflop stats
        vpip_count = sum(1 for h in self.session.hands if h.vpip)
        pfr_count = sum(1 for h in self.session.hands if h.pfr)
        metrics.vpip = vpip_count / metrics.total_hands
        metrics.pfr = pfr_count / metrics.total_hands

        # Count decisions and showdowns
        showdown_count = 0
        won_showdown_count = 0

        for hand in self.session.hands:
            metrics.total_decisions += len(hand.decisions)

            if hand.went_to_showdown:
                showdown_count += 1
                if hand.won_at_showdown:
                    won_showdown_count += 1

            # Track by position
            pos = hand.hero_position.value
            metrics.profit_by_position[pos] = \
                metrics.profit_by_position.get(pos, 0) + hand.profit_bb

        # Showdown stats
        if vpip_count > 0:
            metrics.wtsd = showdown_count / vpip_count
        if showdown_count > 0:
            metrics.won_at_sd = won_showdown_count / showdown_count

        # Track by opponent
        for opp_type in self.session.opponent_types:
            metrics.profit_by_opponent[opp_type] = metrics.total_profit_bb / len(self.session.opponent_types)

        return metrics

    def find_biggest_pots(self, n: int = 10, won: bool = True) -> list[HandTrace]:
        """Find the biggest winning or losing hands."""
        hands = sorted(
            self.session.hands,
            key=lambda h: h.profit_bb,
            reverse=won
        )
        return hands[:n]

    def find_leak_candidates(self) -> list[DecisionAnalysis]:
        """
        Find decisions that might be leaks.

        Looks for patterns like:
        - Large calls that lost
        - Folds in small pots
        - No-tool decisions in complex spots
        """
        leaks = []

        for hand in self.session.hands:
            for decision in hand.decisions:
                analysis = self._analyze_decision(decision, hand)
                if analysis.classification.startswith("potential_leak"):
                    leaks.append(analysis)

        return leaks

    def _analyze_decision(self, decision: DecisionTrace, hand: HandTrace) -> DecisionAnalysis:
        """Analyze a single decision for quality."""
        analysis = DecisionAnalysis(decision=decision)

        if decision.action is None:
            analysis.classification = "no_action"
            return analysis

        action_type = decision.action.action_type

        # Check for no-tool complex decisions
        is_complex = (
            decision.pot > 20 and
            decision.to_call > 5 and
            decision.street != Street.PREFLOP
        )
        if is_complex and len(decision.tool_calls) == 0:
            analysis.classification = "potential_leak_no_analysis"
            analysis.notes.append("Complex decision made without tool consultation")
            return analysis

        # Check for large losing calls
        if action_type == ActionType.CALL and hand.profit_bb < -10:
            if decision.to_call > decision.pot * 0.5:
                analysis.classification = "potential_leak_large_call"
                analysis.notes.append(f"Large call ({decision.to_call:.0f} into {decision.pot:.0f}) lost {hand.profit_bb:.1f} BB")
                return analysis

        # Check for folds in small pots with odds
        if action_type == ActionType.FOLD and decision.to_call < decision.pot * 0.25:
            analysis.classification = "potential_leak_tight_fold"
            analysis.notes.append(f"Folded for {decision.to_call:.0f} into {decision.pot:.0f} pot")
            return analysis

        analysis.classification = "standard"
        return analysis

    def get_action_distribution(self, street: Optional[Street] = None) -> dict[str, int]:
        """Get distribution of actions taken."""
        distribution = defaultdict(int)

        for hand in self.session.hands:
            for decision in hand.decisions:
                if street and decision.street != street:
                    continue
                if decision.action:
                    distribution[decision.action.action_type.value] += 1

        return dict(distribution)

    def get_tool_usage_stats(self) -> dict[str, int]:
        """Get statistics on tool usage."""
        usage = defaultdict(int)

        for hand in self.session.hands:
            for decision in hand.decisions:
                for tc in decision.tool_calls:
                    usage[tc['tool']] += 1

        return dict(usage)

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        metrics = self.compute_metrics()
        leaks = self.find_leak_candidates()
        biggest_wins = self.find_biggest_pots(5, won=True)
        biggest_losses = self.find_biggest_pots(5, won=False)
        tool_usage = self.get_tool_usage_stats()

        lines = [
            "=" * 60,
            "POKER BOT EVALUATION REPORT",
            "=" * 60,
            "",
            str(metrics),
            "",
            "=" * 60,
            "TOOL USAGE",
            "=" * 60,
        ]

        for tool, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
            lines.append(f"  {tool}: {count}")

        lines.extend([
            "",
            "=" * 60,
            "BIGGEST WINS",
            "=" * 60,
        ])
        for hand in biggest_wins:
            lines.append(f"  {hand.hand_id}: {hand.profit_bb:+.1f} BB ({hand.hero_cards} from {hand.hero_position.value})")

        lines.extend([
            "",
            "=" * 60,
            "BIGGEST LOSSES",
            "=" * 60,
        ])
        for hand in biggest_losses:
            lines.append(f"  {hand.hand_id}: {hand.profit_bb:+.1f} BB ({hand.hero_cards} from {hand.hero_position.value})")

        if leaks:
            lines.extend([
                "",
                "=" * 60,
                f"POTENTIAL LEAKS ({len(leaks)} found)",
                "=" * 60,
            ])
            for leak in leaks[:10]:
                lines.append(f"  [{leak.classification}] {leak.decision.hand_id}: {leak.notes[0] if leak.notes else ''}")

        return "\n".join(lines)


def compare_sessions(sessions: list[SessionTrace]) -> str:
    """Compare metrics across multiple sessions."""
    if not sessions:
        return "No sessions to compare"

    lines = [
        "=" * 60,
        "SESSION COMPARISON",
        "=" * 60,
        "",
        f"{'Session ID':<20} {'Hands':>8} {'Profit':>10} {'BB/100':>10} {'VPIP':>8} {'PFR':>8}",
        "-" * 60,
    ]

    for session in sessions:
        analyzer = SessionAnalyzer(session)
        metrics = analyzer.compute_metrics()
        lines.append(
            f"{session.session_id[:20]:<20} "
            f"{metrics.total_hands:>8} "
            f"{metrics.total_profit_bb:>+10.1f} "
            f"{metrics.bb_per_100:>+10.1f} "
            f"{metrics.vpip:>8.1%} "
            f"{metrics.pfr:>8.1%}"
        )

    return "\n".join(lines)
