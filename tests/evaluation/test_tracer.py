"""Tests for tracer module."""

import pytest
import tempfile
import os

from src.evaluation.tracer import (
    DecisionTrace, HandTrace, SessionTrace,
    Tracer, TraceStore
)
from src.core import Position, Street, ActionType, Action, HoleCards, Card
from src.core.game_state import create_6max_game


class TestDecisionTrace:
    """Tests for DecisionTrace dataclass."""

    def test_create_decision_trace(self):
        trace = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.FLOP,
            position=Position.BTN,
            pot=100.0,
            to_call=30.0,
            stack=970.0
        )

        assert trace.decision_id == "d001"
        assert trace.hand_id == "hand_001"
        assert trace.street == Street.FLOP
        assert trace.pot == 100.0
        assert trace.to_call == 30.0
        assert trace.stack == 970.0

    def test_decision_trace_with_action(self):
        action = Action.call(30.0)
        trace = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.FLOP,
            position=Position.CO,
            pot=100.0,
            to_call=30.0,
            stack=970.0,
            action=action
        )

        assert trace.action.action_type == ActionType.CALL
        assert trace.action.amount == 30.0

    def test_decision_trace_add_thought(self):
        trace = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.TURN,
            position=Position.BTN,
            pot=200.0,
            to_call=50.0,
            stack=800.0
        )

        trace.add_thought("Evaluating pot odds")
        trace.add_thought("Considering equity")

        assert len(trace.thoughts) == 2
        assert "pot odds" in trace.thoughts[0]

    def test_decision_trace_set_decision(self):
        trace = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.RIVER,
            position=Position.SB,
            pot=300.0,
            to_call=100.0,
            stack=500.0
        )

        action = Action.fold()
        trace.set_decision(action, "Opponent shows strength", confidence=0.85)

        assert trace.action.action_type == ActionType.FOLD
        assert trace.reasoning == "Opponent shows strength"
        assert trace.confidence == 0.85

    def test_decision_trace_to_dict(self):
        trace = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.PREFLOP,
            position=Position.UTG,
            pot=15.0,
            to_call=10.0,
            stack=990.0
        )

        d = trace.to_dict()
        assert d["decision_id"] == "d001"
        assert d["hand_id"] == "hand_001"
        assert d["street"] == "PREFLOP"
        assert d["pot"] == 15.0


class TestHandTrace:
    """Tests for HandTrace dataclass."""

    def test_create_hand_trace(self):
        trace = HandTrace(
            hand_id="hand_001",
            hero_position=Position.BTN,
            hero_cards="AhKh"
        )

        assert trace.hand_id == "hand_001"
        assert trace.hero_position == Position.BTN
        assert trace.hero_cards == "AhKh"

    def test_hand_trace_add_decision(self):
        hand = HandTrace(
            hand_id="hand_001",
            hero_position=Position.CO,
            hero_cards="QdQs"
        )

        decision = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.PREFLOP,
            position=Position.CO,
            pot=15.0,
            to_call=10.0,
            stack=990.0,
            action=Action.raise_to(30.0)
        )

        hand.add_decision(decision)
        assert len(hand.decisions) == 1
        assert hand.vpip is True  # Raised preflop
        assert hand.pfr is True

    def test_hand_trace_complete_hand(self):
        hand = HandTrace(
            hand_id="hand_001",
            hero_position=Position.SB,
            hero_cards="7d7s",
            profit_bb=25.0,
            went_to_showdown=True,
            won_at_showdown=True,
            final_board="Kd7c2h5s3c",
            vpip=True,
            pfr=True
        )

        assert hand.profit_bb == 25.0
        assert hand.went_to_showdown is True
        assert hand.won_at_showdown is True
        assert hand.vpip is True
        assert hand.pfr is True

    def test_hand_trace_vpip_on_call(self):
        """Test that VPIP is tracked on calls."""
        hand = HandTrace(
            hand_id="hand_001",
            hero_position=Position.CO,
            hero_cards="JdTs"
        )

        decision = DecisionTrace(
            decision_id="d001",
            hand_id="hand_001",
            street=Street.PREFLOP,
            position=Position.CO,
            pot=25.0,
            to_call=20.0,
            stack=980.0,
            action=Action.call(20.0)
        )

        hand.add_decision(decision)
        assert hand.vpip is True
        assert hand.pfr is False  # Called, didn't raise

    def test_hand_trace_to_dict(self):
        hand = HandTrace(
            hand_id="hand_002",
            hero_position=Position.BB,
            hero_cards="AsAc",
            profit_bb=100.0
        )

        d = hand.to_dict()
        assert d["hand_id"] == "hand_002"
        assert d["hero_position"].lower() == "bb"  # Position value may be uppercase
        assert d["profit_bb"] == 100.0


class TestSessionTrace:
    """Tests for SessionTrace dataclass."""

    def test_create_session_trace(self):
        session = SessionTrace(
            session_id="session_001",
            opponent_types=["fish", "tag"]
        )

        assert session.session_id == "session_001"
        assert len(session.opponent_types) == 2

    def test_session_bb_per_100_empty(self):
        session = SessionTrace(
            session_id="session_001",
            opponent_types=["random"]
        )

        assert session.bb_per_100 == 0.0

    def test_session_bb_per_100_with_hands(self):
        session = SessionTrace(
            session_id="session_001",
            opponent_types=["fish"]
        )

        # Add 100 hands with 50 BB profit total
        for i in range(100):
            hand = HandTrace(
                hand_id=f"hand_{i}",
                hero_position=Position.BTN,
                hero_cards="AhKs",
                profit_bb=0.5  # 0.5 BB per hand = 50 BB total
            )
            session.add_hand(hand)

        assert session.total_hands == 100
        assert session.total_profit_bb == 50.0
        assert session.bb_per_100 == 50.0

    def test_session_vpip_pct(self):
        session = SessionTrace(
            session_id="session_001",
            opponent_types=["fish"]
        )

        # Add hands with varying VPIP
        for i in range(10):
            hand = HandTrace(
                hand_id=f"hand_{i}",
                hero_position=Position.BTN,
                hero_cards="AhKs",
                vpip=(i % 2 == 0)  # 50% VPIP
            )
            session.add_hand(hand)

        assert abs(session.vpip_pct - 0.5) < 0.01

    def test_session_trace_to_dict(self):
        session = SessionTrace(
            session_id="session_002",
            opponent_types=["nit", "lag"],
            total_profit_bb=25.0
        )

        hand = HandTrace(
            hand_id="hand_001",
            hero_position=Position.CO,
            hero_cards="JdTd",
            profit_bb=25.0
        )
        session.add_hand(hand)

        d = session.to_dict()
        assert d["session_id"] == "session_002"
        assert len(d["hands"]) == 1


class TestTracer:
    """Tests for Tracer class."""

    def test_tracer_creation(self):
        tracer = Tracer(
            session_id="test_session",
            opponent_types=["fish", "tag"]
        )

        assert tracer.session.session_id == "test_session"
        assert tracer.session.opponent_types == ["fish", "tag"]

    def test_start_hand(self):
        tracer = Tracer("session_001", ["random"])

        cards = HoleCards(Card.from_str("As"), Card.from_str("Kh"))
        hand = tracer.start_hand("hand_001", Position.BTN, cards)

        assert hand.hand_id == "hand_001"
        assert hand.hero_position == Position.BTN
        assert hand.hero_cards == "AKo"

    def test_start_decision_with_game_state(self):
        tracer = Tracer("session_001", ["random"])

        cards = HoleCards(Card.from_str("As"), Card.from_str("Kh"))
        tracer.start_hand("hand_001", Position.BTN, cards)

        # Create a game state with all required arguments
        stacks = {pos: 1000.0 for pos in Position}
        game = create_6max_game(
            hand_id="hand_001",
            hero_id="hero",
            hero_position=Position.BTN,
            hero_cards=HoleCards.from_str("AsKh"),
            stacks=stacks
        )

        decision = tracer.start_decision(game)

        assert decision.hand_id == "hand_001"
        assert decision.street == Street.PREFLOP

    def test_end_hand(self):
        tracer = Tracer("session_001", ["random"])

        cards = HoleCards(Card.from_str("Qd"), Card.from_str("Qs"))
        tracer.start_hand("hand_001", Position.CO, cards)

        tracer.end_hand(
            profit_bb=50.0,
            went_to_sd=True,
            won_sd=True,
            board="Ah7d2c9sKc"
        )

        assert len(tracer.session.hands) == 1
        assert tracer.session.hands[0].profit_bb == 50.0

    def test_finalize_session(self):
        tracer = Tracer("session_001", ["fish"])

        # Play two hands
        cards1 = HoleCards(Card.from_str("As"), Card.from_str("Kh"))
        tracer.start_hand("hand_001", Position.BTN, cards1)
        tracer.end_hand(profit_bb=20.0, went_to_sd=False, won_sd=False, board="")

        cards2 = HoleCards(Card.from_str("7d"), Card.from_str("2c"))
        tracer.start_hand("hand_002", Position.SB, cards2)
        tracer.end_hand(profit_bb=-15.0, went_to_sd=False, won_sd=False, board="")

        session = tracer.finalize()

        assert session.total_profit_bb == 5.0
        assert len(session.hands) == 2
        assert session.end_time is not None


class TestTraceStore:
    """Tests for TraceStore database operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_store_creation(self, temp_db):
        store = TraceStore(temp_db)
        assert store is not None

    def test_save_and_load_session(self, temp_db):
        store = TraceStore(temp_db)

        # Create a session
        session = SessionTrace(
            session_id="test_session",
            opponent_types=["fish", "tag"],
            total_profit_bb=50.0
        )

        hand = HandTrace(
            hand_id="hand_001",
            hero_position=Position.BTN,
            hero_cards="AhKh",
            profit_bb=50.0,
            went_to_showdown=True,
            won_at_showdown=True
        )
        session.add_hand(hand)
        session.finalize()

        # Save
        store.save_session(session)

        # Load
        loaded = store.load_session("test_session")

        assert loaded is not None
        assert loaded["session_id"] == "test_session"
        assert loaded["total_profit_bb"] == 100.0  # 50 from init + 50 from hand
        assert len(loaded["hands"]) == 1

    def test_list_sessions(self, temp_db):
        store = TraceStore(temp_db)

        # Create multiple sessions
        for i in range(3):
            session = SessionTrace(
                session_id=f"session_{i}",
                opponent_types=["random"],
                total_profit_bb=i * 10.0
            )
            session.finalize()
            store.save_session(session)

        sessions = store.list_sessions()
        assert len(sessions) == 3

    def test_load_nonexistent_session(self, temp_db):
        store = TraceStore(temp_db)
        loaded = store.load_session("nonexistent")
        assert loaded is None

    def test_get_losing_hands(self, temp_db):
        store = TraceStore(temp_db)

        session = SessionTrace(
            session_id="query_test",
            opponent_types=["fish"]
        )

        # Add winning and losing hands
        for i in range(5):
            hand = HandTrace(
                hand_id=f"hand_{i}",
                hero_position=Position.BTN,
                hero_cards="AhKs",
                profit_bb=20.0 if i % 2 == 0 else -10.0
            )
            session.add_hand(hand)

        session.finalize()
        store.save_session(session)

        # Query losing hands
        losers = store.get_losing_hands(session_id="query_test", limit=10)
        assert len(losers) == 2  # hands 1 and 3


class TestTracerIntegration:
    """Integration tests for full tracing workflow."""

    def test_full_session_trace(self):
        """Test complete session with multiple hands."""
        tracer = Tracer("integration_test", ["fish", "nit"])

        # Hand 1: Win
        cards1 = HoleCards(Card.from_str("As"), Card.from_str("Kh"))
        tracer.start_hand("hand_001", Position.BTN, cards1)
        tracer.end_hand(
            profit_bb=55.0,
            went_to_sd=False,
            won_sd=False,
            board="Ah7d2c"
        )

        # Hand 2: Loss
        cards2 = HoleCards(Card.from_str("7d"), Card.from_str("8d"))
        tracer.start_hand("hand_002", Position.CO, cards2)
        tracer.end_hand(profit_bb=-10.0, went_to_sd=False, won_sd=False, board="")

        # Finalize
        session = tracer.finalize()

        assert session.total_profit_bb == 45.0  # 55 - 10
        assert len(session.hands) == 2
        assert session.hands[0].profit_bb == 55.0
        assert session.hands[1].profit_bb == -10.0
