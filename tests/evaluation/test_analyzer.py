"""Tests for analyzer module."""

import pytest

from src.evaluation.analyzer import (
    PerformanceMetrics, DecisionAnalysis, SessionAnalyzer, compare_sessions
)
from src.evaluation.tracer import DecisionTrace, HandTrace, SessionTrace
from src.core import Position, Street, ActionType, Action


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_default_metrics(self):
        metrics = PerformanceMetrics()

        assert metrics.total_hands == 0
        assert metrics.total_profit_bb == 0.0
        assert metrics.vpip == 0.0
        assert metrics.pfr == 0.0

    def test_metrics_with_values(self):
        metrics = PerformanceMetrics(
            total_hands=1000,
            total_profit_bb=50.0,
            bb_per_100=5.0,
            vpip=0.22,
            pfr=0.18,
            wtsd=0.35,
            won_at_sd=0.55
        )

        assert metrics.total_hands == 1000
        assert metrics.bb_per_100 == 5.0
        assert metrics.vpip == 0.22

    def test_metrics_str_format(self):
        metrics = PerformanceMetrics(
            total_hands=100,
            total_profit_bb=25.0,
            bb_per_100=25.0,
            vpip=0.25,
            pfr=0.20
        )

        output = str(metrics)
        assert "Performance Metrics" in output
        assert "100" in output
        assert "25.0" in output

    def test_metrics_with_position_breakdown(self):
        metrics = PerformanceMetrics(
            total_hands=100,
            profit_by_position={
                "btn": 30.0,
                "co": 15.0,
                "bb": -10.0,
                "sb": -5.0
            }
        )

        assert metrics.profit_by_position["btn"] == 30.0
        assert "By position" in str(metrics)


class TestDecisionAnalysis:
    """Tests for DecisionAnalysis dataclass."""

    def test_create_analysis(self):
        decision = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.FLOP,
            position=Position.BTN,
            pot=100.0,
            to_call=50.0,
            stack=950.0,
            action=Action.call(50.0)
        )

        analysis = DecisionAnalysis(
            decision=decision,
            classification="standard",
            ev_estimate=5.0
        )

        assert analysis.classification == "standard"
        assert analysis.ev_estimate == 5.0

    def test_analysis_with_notes(self):
        decision = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.TURN,
            position=Position.CO,
            pot=200.0,
            to_call=150.0,
            stack=700.0,
            action=Action.call(150.0)
        )

        analysis = DecisionAnalysis(
            decision=decision,
            classification="potential_leak_large_call",
            notes=["Large call with marginal holding", "No equity calculation used"]
        )

        assert len(analysis.notes) == 2
        assert "Notes:" in str(analysis)


class TestSessionAnalyzer:
    """Tests for SessionAnalyzer class."""

    @pytest.fixture
    def sample_session(self) -> SessionTrace:
        """Create a sample session for testing."""
        session = SessionTrace(
            session_id="test_session",
            opponent_types=["fish", "tag"],
            total_profit_bb=30.0
        )

        # Hand 1: Win with VPIP and PFR
        hand1 = HandTrace(
            hand_id="hand_001",
            hero_position=Position.BTN,
            hero_cards="AhKh",
            profit_bb=50.0,
            went_to_showdown=True,
            won_at_showdown=True,
            vpip=True,
            pfr=True
        )
        hand1.decisions.append(
            DecisionTrace(
                decision_id="d001",
                hand_id="hand_001",
                street=Street.PREFLOP,
                position=Position.BTN,
                pot=15.0,
                to_call=10.0,
                stack=990.0,
                action=Action.raise_to(30.0),
                tool_calls=[{"tool": "hand_eval", "result": "ak_suited"}]
            )
        )
        hand1.decisions.append(
            DecisionTrace(
                decision_id="d002",
                hand_id="hand_001",
                street=Street.FLOP,
                position=Position.BTN,
                pot=60.0,
                to_call=0.0,
                stack=960.0,
                action=Action.bet(45.0),
                tool_calls=[{"tool": "equity_calc", "result": "72%"}]
            )
        )
        session.hands.append(hand1)

        # Hand 2: Loss with VPIP
        hand2 = HandTrace(
            hand_id="hand_002",
            hero_position=Position.CO,
            hero_cards="7d8d",
            profit_bb=-20.0,
            went_to_showdown=True,
            won_at_showdown=False,
            vpip=True,
            pfr=False
        )
        hand2.decisions.append(
            DecisionTrace(
                decision_id="d003",
                hand_id="hand_002",
                street=Street.PREFLOP,
                position=Position.CO,
                pot=15.0,
                to_call=10.0,
                stack=990.0,
                action=Action.call(10.0)
            )
        )
        session.hands.append(hand2)

        # Hand 3: Fold
        hand3 = HandTrace(
            hand_id="hand_003",
            hero_position=Position.UTG,
            hero_cards="2h7c",
            profit_bb=0.0,
            vpip=False,
            pfr=False
        )
        hand3.decisions.append(
            DecisionTrace(
                decision_id="d004",
                hand_id="hand_003",
                street=Street.PREFLOP,
                position=Position.UTG,
                pot=15.0,
                to_call=10.0,
                stack=990.0,
                action=Action.fold()
            )
        )
        session.hands.append(hand3)

        return session

    def test_compute_metrics_total_hands(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        metrics = analyzer.compute_metrics()

        assert metrics.total_hands == 3

    def test_compute_metrics_profit(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        metrics = analyzer.compute_metrics()

        assert metrics.total_profit_bb == 30.0

    def test_compute_metrics_vpip(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        metrics = analyzer.compute_metrics()

        # 2 out of 3 hands had VPIP
        assert abs(metrics.vpip - 0.667) < 0.01

    def test_compute_metrics_pfr(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        metrics = analyzer.compute_metrics()

        # 1 out of 3 hands had PFR
        assert abs(metrics.pfr - 0.333) < 0.01

    def test_compute_metrics_showdown(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        metrics = analyzer.compute_metrics()

        # 2 showdowns, 1 won
        assert metrics.won_at_sd == 0.5

    def test_find_biggest_pots_winners(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        biggest_wins = analyzer.find_biggest_pots(n=3, won=True)

        assert biggest_wins[0].profit_bb == 50.0
        assert biggest_wins[0].hand_id == "hand_001"

    def test_find_biggest_pots_losers(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        biggest_losses = analyzer.find_biggest_pots(n=3, won=False)

        assert biggest_losses[0].profit_bb == -20.0
        assert biggest_losses[0].hand_id == "hand_002"

    def test_get_action_distribution(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        distribution = analyzer.get_action_distribution()

        # Check that actions are counted
        total_actions = sum(distribution.values())
        assert total_actions == 4  # raise, bet, call, fold

    def test_get_action_distribution_by_street(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        preflop_actions = analyzer.get_action_distribution(street=Street.PREFLOP)

        # 3 preflop actions (raise, call, fold)
        total_actions = sum(preflop_actions.values())
        assert total_actions == 3

    def test_get_tool_usage_stats(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        usage = analyzer.get_tool_usage_stats()

        assert usage.get("hand_eval") == 1
        assert usage.get("equity_calc") == 1

    def test_generate_report(self, sample_session):
        analyzer = SessionAnalyzer(sample_session)
        report = analyzer.generate_report()

        assert "POKER BOT EVALUATION REPORT" in report
        assert "TOOL USAGE" in report
        assert "BIGGEST WINS" in report
        assert "BIGGEST LOSSES" in report

    def test_find_leak_candidates_large_call(self):
        """Test detection of large losing calls."""
        session = SessionTrace(
            session_id="leak_test",
            opponent_types=["tag"],
            total_profit_bb=-50.0
        )

        hand = HandTrace(
            hand_id="leak_hand",
            hero_position=Position.BB,
            hero_cards="9h9s",
            profit_bb=-50.0,
            vpip=True
        )
        # Large call that lost (add tool_calls to avoid triggering "no_analysis" check first)
        hand.decisions.append(
            DecisionTrace(
                decision_id="d001",
                hand_id="leak_hand",
                street=Street.RIVER,
                position=Position.BB,
                pot=100.0,
                to_call=75.0,  # 75% pot call
                stack=925.0,
                action=Action.call(75.0),
                tool_calls=[{"tool": "equity_calc", "result": "30%"}]  # Used analysis but still bad call
            )
        )
        session.hands.append(hand)

        analyzer = SessionAnalyzer(session)
        leaks = analyzer.find_leak_candidates()

        assert len(leaks) > 0
        assert any("large_call" in leak.classification for leak in leaks)


class TestCompareSessionsFunction:
    """Tests for compare_sessions function."""

    def test_compare_empty_list(self):
        result = compare_sessions([])
        assert "No sessions to compare" in result

    def test_compare_multiple_sessions(self):
        sessions = []

        for i in range(3):
            session = SessionTrace(
                session_id=f"session_{i}",
                opponent_types=["random"],
                total_profit_bb=i * 25.0
            )
            # Add some hands
            for j in range(100):
                session.hands.append(
                    HandTrace(
                        hand_id=f"hand_{i}_{j}",
                        hero_position=Position.BTN,
                        hero_cards="AhKs",
                        profit_bb=i * 0.25,
                        vpip=(j % 4 == 0),
                        pfr=(j % 5 == 0)
                    )
                )
            sessions.append(session)

        result = compare_sessions(sessions)

        assert "SESSION COMPARISON" in result
        assert "session_0" in result
        assert "session_1" in result
        assert "session_2" in result
        assert "Hands" in result
        assert "Profit" in result


class TestAnalyzerEdgeCases:
    """Edge case tests for analyzer."""

    def test_empty_session(self):
        session = SessionTrace(
            session_id="empty",
            opponent_types=["random"]
        )

        analyzer = SessionAnalyzer(session)
        metrics = analyzer.compute_metrics()

        assert metrics.total_hands == 0
        assert metrics.vpip == 0.0
        assert metrics.won_at_sd == 0.0

    def test_no_showdowns(self):
        session = SessionTrace(
            session_id="no_sd",
            opponent_types=["fish"],
            total_profit_bb=10.0
        )

        # All hands folded before showdown
        for i in range(10):
            session.hands.append(
                HandTrace(
                    hand_id=f"hand_{i}",
                    hero_position=Position.CO,
                    hero_cards="KhQs",
                    profit_bb=1.0,
                    went_to_showdown=False,
                    vpip=True
                )
            )

        analyzer = SessionAnalyzer(session)
        metrics = analyzer.compute_metrics()

        assert metrics.wtsd == 0.0
        assert metrics.won_at_sd == 0.0

    def test_all_folds(self):
        session = SessionTrace(
            session_id="all_folds",
            opponent_types=["lag"],
            total_profit_bb=-15.0  # Lost blinds
        )

        for i in range(10):
            session.hands.append(
                HandTrace(
                    hand_id=f"hand_{i}",
                    hero_position=Position.UTG,
                    hero_cards="2d7c",
                    profit_bb=-1.5,
                    vpip=False,
                    pfr=False
                )
            )

        analyzer = SessionAnalyzer(session)
        metrics = analyzer.compute_metrics()

        assert metrics.vpip == 0.0
        assert metrics.pfr == 0.0
