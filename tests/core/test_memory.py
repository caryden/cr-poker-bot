"""Tests for opponent memory system."""

import pytest
import math
from datetime import datetime

from src.core.memory import (
    HandSummary, OpponentStats, OpponentMemory, TableMemory
)
from src.core.primitives import ActionType, Position, Street
from src.core.beliefs import PLAYER_TYPES


class TestOpponentStats:
    """Tests for OpponentStats class."""

    def test_stats_creation(self):
        stats = OpponentStats()
        assert stats.vpip_hands == 0
        assert stats.total_hands == 0

    def test_vpip_calculation(self):
        stats = OpponentStats()
        stats.vpip_opportunities = 100
        stats.vpip_hands = 25

        assert stats.vpip == 0.25

    def test_vpip_zero_hands(self):
        stats = OpponentStats()
        assert stats.vpip == 0.0

    def test_pfr_calculation(self):
        stats = OpponentStats()
        stats.pfr_opportunities = 100
        stats.pfr_hands = 20

        assert stats.pfr == 0.20

    def test_three_bet_pct(self):
        stats = OpponentStats()
        stats.three_bet_opportunities = 20
        stats.three_bet_made = 4

        assert stats.three_bet_pct == 0.20

    def test_fold_to_3bet_pct(self):
        stats = OpponentStats()
        stats.faced_3bet = 10
        stats.fold_to_3bet = 6

        assert stats.fold_to_3bet_pct == 0.60

    def test_cbet_pct(self):
        stats = OpponentStats()
        stats.cbet_opportunities = 50
        stats.cbet_made = 35

        assert stats.cbet_pct == 0.70

    def test_aggression_factor(self):
        stats = OpponentStats()
        stats.postflop_bets = 30
        stats.postflop_raises = 20
        stats.postflop_calls = 25

        # AF = (30 + 20) / 25 = 2.0
        assert stats.aggression_factor == 2.0

    def test_aggression_factor_no_calls(self):
        stats = OpponentStats()
        stats.postflop_bets = 10
        stats.postflop_raises = 5
        stats.postflop_calls = 0

        assert stats.aggression_factor == float('inf')

    def test_aggression_factor_no_actions(self):
        stats = OpponentStats()
        assert stats.aggression_factor == 0.0

    def test_wtsd(self):
        stats = OpponentStats()
        stats.could_showdown = 50
        stats.went_to_showdown = 20

        assert stats.wtsd == 0.40

    def test_won_at_sd(self):
        stats = OpponentStats()
        stats.went_to_showdown = 20
        stats.won_at_showdown = 12

        assert stats.won_at_sd == 0.60

    def test_sample_quality_tiny(self):
        stats = OpponentStats()
        stats.vpip_opportunities = 10
        assert stats.get_sample_quality() == "tiny"

    def test_sample_quality_large(self):
        stats = OpponentStats()
        stats.vpip_opportunities = 500
        assert stats.get_sample_quality() == "large"


class TestOpponentMemory:
    """Tests for OpponentMemory class."""

    def test_memory_creation(self):
        memory = OpponentMemory("villain1")
        assert memory.opponent_id == "villain1"
        assert memory.stats.total_hands == 0

    def test_record_preflop_call(self):
        memory = OpponentMemory("villain1")

        memory.record_preflop_action(ActionType.CALL, Position.BTN)

        assert memory.stats.vpip_opportunities == 1
        assert memory.stats.vpip_hands == 1
        assert memory.stats.pfr_hands == 0

    def test_record_preflop_raise(self):
        memory = OpponentMemory("villain1")

        memory.record_preflop_action(ActionType.RAISE, Position.CO)

        assert memory.stats.vpip_hands == 1
        assert memory.stats.pfr_hands == 1

    def test_record_preflop_fold(self):
        memory = OpponentMemory("villain1")

        memory.record_preflop_action(ActionType.FOLD, Position.UTG)

        assert memory.stats.vpip_opportunities == 1
        assert memory.stats.vpip_hands == 0

    def test_record_3bet(self):
        memory = OpponentMemory("villain1")

        memory.record_preflop_action(ActionType.RAISE, Position.BTN, facing_raise=True)

        assert memory.stats.three_bet_opportunities == 1
        assert memory.stats.three_bet_made == 1

    def test_record_fold_to_3bet(self):
        memory = OpponentMemory("villain1")

        memory.record_preflop_action(ActionType.FOLD, Position.CO, facing_raise=True)

        assert memory.stats.faced_3bet == 1
        assert memory.stats.fold_to_3bet == 1

    def test_record_cbet(self):
        memory = OpponentMemory("villain1")

        memory.record_cbet_opportunity(made_cbet=True)
        memory.record_cbet_opportunity(made_cbet=False)

        assert memory.stats.cbet_opportunities == 2
        assert memory.stats.cbet_made == 1

    def test_record_faced_cbet(self):
        memory = OpponentMemory("villain1")

        memory.record_faced_cbet(folded=True)
        memory.record_faced_cbet(folded=False)

        assert memory.stats.faced_cbet == 2
        assert memory.stats.fold_to_cbet == 1

    def test_record_postflop_actions(self):
        memory = OpponentMemory("villain1")

        memory.record_postflop_action(ActionType.BET)
        memory.record_postflop_action(ActionType.RAISE)
        memory.record_postflop_action(ActionType.CALL)
        memory.record_postflop_action(ActionType.FOLD)

        assert memory.stats.postflop_bets == 1
        assert memory.stats.postflop_raises == 1
        assert memory.stats.postflop_calls == 1
        assert memory.stats.postflop_folds == 1

    def test_record_showdown(self):
        memory = OpponentMemory("villain1")

        memory.record_showdown(won=True, showed_strong=True)
        memory.record_showdown(won=False, showed_strong=False)

        assert memory.stats.went_to_showdown == 2
        assert memory.stats.won_at_showdown == 1
        assert memory.stats.showed_value == 1
        assert memory.stats.showed_bluff == 1

    def test_timestamps_set(self):
        memory = OpponentMemory("villain1")
        assert memory.first_seen is None

        memory.record_preflop_action(ActionType.CALL, Position.BTN)

        assert memory.first_seen is not None
        assert memory.last_seen is not None

    def test_to_beliefs(self):
        memory = OpponentMemory("villain1")

        # Record some actions
        for _ in range(10):
            memory.record_preflop_action(ActionType.CALL, Position.BTN)
            memory.record_postflop_action(ActionType.CALL)

        beliefs = memory.to_beliefs()

        assert beliefs.player_id == "villain1"
        assert beliefs.player_type is not None

    def test_to_beliefs_infers_fish(self):
        memory = OpponentMemory("villain1")

        # Fish pattern: high VPIP, low PFR, passive
        for _ in range(50):
            memory.record_preflop_action(ActionType.CALL, Position.BTN)
            memory.record_postflop_action(ActionType.CALL)

        beliefs = memory.to_beliefs()
        best_type, _ = beliefs.get_most_likely_player_type()

        assert best_type == "Fish"

    def test_to_beliefs_infers_nit(self):
        memory = OpponentMemory("villain1")

        # Nit pattern: low VPIP (lots of folds)
        for _ in range(50):
            memory.record_preflop_action(ActionType.FOLD, Position.UTG)

        beliefs = memory.to_beliefs()
        best_type, _ = beliefs.get_most_likely_player_type()

        assert best_type == "NIT"

    def test_get_summary(self):
        memory = OpponentMemory("villain1")

        for _ in range(20):
            memory.record_preflop_action(ActionType.RAISE, Position.BTN)

        summary = memory.get_summary()

        assert "villain1" in summary
        assert "20 hands" in summary
        assert "VPIP" in summary

    def test_to_dict_and_from_dict(self):
        memory = OpponentMemory("villain1")
        memory.record_preflop_action(ActionType.RAISE, Position.BTN)
        memory.record_postflop_action(ActionType.BET)

        data = memory.to_dict()
        restored = OpponentMemory.from_dict(data)

        assert restored.opponent_id == "villain1"
        assert restored.stats.vpip_hands == 1
        assert restored.stats.postflop_bets == 1


class TestTableMemory:
    """Tests for TableMemory class."""

    def test_table_memory_creation(self):
        table = TableMemory()
        assert len(table.opponents) == 0

    def test_get_opponent_creates_new(self):
        table = TableMemory()

        memory = table.get_opponent("villain1")

        assert memory.opponent_id == "villain1"
        assert "villain1" in table.opponents

    def test_get_opponent_returns_existing(self):
        table = TableMemory()

        memory1 = table.get_opponent("villain1")
        memory1.record_preflop_action(ActionType.CALL, Position.BTN)

        memory2 = table.get_opponent("villain1")

        assert memory1 is memory2
        assert memory2.stats.vpip_hands == 1

    def test_get_all_summaries(self):
        table = TableMemory()

        table.get_opponent("villain1").record_preflop_action(ActionType.CALL, Position.BTN)
        table.get_opponent("villain2").record_preflop_action(ActionType.RAISE, Position.CO)

        summaries = table.get_all_summaries()

        assert "villain1" in summaries
        assert "villain2" in summaries

    def test_to_dict_and_from_dict(self):
        table = TableMemory()
        table.get_opponent("v1").record_preflop_action(ActionType.CALL, Position.BTN)
        table.get_opponent("v2").record_preflop_action(ActionType.RAISE, Position.CO)

        data = table.to_dict()
        restored = TableMemory.from_dict(data)

        assert "v1" in restored.opponents
        assert "v2" in restored.opponents
        assert restored.opponents["v1"].stats.vpip_hands == 1


class TestHandSummary:
    """Tests for HandSummary class."""

    def test_hand_summary_creation(self):
        summary = HandSummary(
            hand_id="hand1",
            timestamp=datetime.now(),
            position=Position.BTN,
            hole_cards=None
        )
        assert summary.hand_id == "hand1"

    def test_notable_big_pot(self):
        summary = HandSummary(
            hand_id="hand1",
            timestamp=datetime.now(),
            position=Position.BTN,
            hole_cards=None,
            pot_size=100
        )
        assert summary.is_notable()

    def test_notable_showdown(self):
        summary = HandSummary(
            hand_id="hand1",
            timestamp=datetime.now(),
            position=Position.BTN,
            hole_cards=None,
            went_to_showdown=True
        )
        assert summary.is_notable()

    def test_notable_aggressive_line(self):
        summary = HandSummary(
            hand_id="hand1",
            timestamp=datetime.now(),
            position=Position.BTN,
            hole_cards=None,
            postflop_aggression=4
        )
        assert summary.is_notable()

    def test_not_notable_small_pot(self):
        summary = HandSummary(
            hand_id="hand1",
            timestamp=datetime.now(),
            position=Position.BTN,
            hole_cards=None,
            pot_size=10
        )
        assert not summary.is_notable()
