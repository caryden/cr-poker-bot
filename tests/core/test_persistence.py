"""Tests for persistence layer."""

import pytest
import tempfile
from pathlib import Path

from src.core.persistence import (
    MemoryStore, BeliefStore, PersistentMemoryManager
)
from src.core.memory import OpponentMemory, TableMemory
from src.core.beliefs import VillainBeliefs, PLAYER_TYPES
from src.core.subjective_logic import Opinion, MultinomialOpinion
from src.core.primitives import ActionType, Position


class TestMemoryStore:
    """Tests for MemoryStore class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def store(self, temp_db):
        """Create a MemoryStore with temp database."""
        return MemoryStore(temp_db)

    def test_store_creation(self, store):
        assert store.db_path.exists()

    def test_save_and_load_opponent(self, store):
        memory = OpponentMemory("villain1")
        memory.record_preflop_action(ActionType.RAISE, Position.BTN)
        memory.record_postflop_action(ActionType.BET)

        store.save_opponent(memory)
        loaded = store.load_opponent("villain1")

        assert loaded is not None
        assert loaded.opponent_id == "villain1"
        assert loaded.stats.vpip_hands == 1
        assert loaded.stats.postflop_bets == 1

    def test_load_nonexistent_opponent(self, store):
        loaded = store.load_opponent("nonexistent")
        assert loaded is None

    def test_update_opponent(self, store):
        memory = OpponentMemory("villain1")
        memory.record_preflop_action(ActionType.CALL, Position.BTN)
        store.save_opponent(memory)

        # Load, modify, save again
        loaded = store.load_opponent("villain1")
        loaded.record_preflop_action(ActionType.RAISE, Position.CO)
        store.save_opponent(loaded)

        # Load again and verify
        reloaded = store.load_opponent("villain1")
        assert reloaded.stats.vpip_opportunities == 2
        assert reloaded.stats.pfr_hands == 1

    def test_load_all_opponents(self, store):
        for i in range(3):
            memory = OpponentMemory(f"villain{i}")
            memory.record_preflop_action(ActionType.CALL, Position.BTN)
            store.save_opponent(memory)

        all_opponents = store.load_all_opponents()

        assert len(all_opponents) == 3
        assert "villain0" in all_opponents
        assert "villain1" in all_opponents
        assert "villain2" in all_opponents

    def test_save_and_load_table_memory(self, store):
        table = TableMemory()
        table.get_opponent("v1").record_preflop_action(ActionType.CALL, Position.BTN)
        table.get_opponent("v2").record_preflop_action(ActionType.RAISE, Position.CO)

        store.save_table_memory(table)
        loaded = store.load_table_memory()

        assert "v1" in loaded.opponents
        assert "v2" in loaded.opponents

    def test_delete_opponent(self, store):
        memory = OpponentMemory("villain1")
        store.save_opponent(memory)

        deleted = store.delete_opponent("villain1")
        assert deleted

        loaded = store.load_opponent("villain1")
        assert loaded is None

    def test_delete_nonexistent(self, store):
        deleted = store.delete_opponent("nonexistent")
        assert not deleted

    def test_get_stats(self, store):
        memory = OpponentMemory("villain1")
        memory.record_preflop_action(ActionType.CALL, Position.BTN)
        store.save_opponent(memory)

        stats = store.get_stats()

        assert stats["total_opponents"] == 1
        assert stats["total_hands_observed"] == 1

    def test_session_management(self, store):
        session_id = store.start_session("Test session")
        assert session_id > 0

        store.record_session_opponent(session_id, "villain1", hands=10, profit=50.0)
        store.end_session(session_id, hands_played=10)


class TestBeliefStore:
    """Tests for BeliefStore class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def store(self, temp_db):
        """Create a BeliefStore with temp database."""
        return BeliefStore(temp_db)

    def test_save_and_load_beliefs(self, store):
        beliefs = VillainBeliefs("villain1")
        beliefs.update_player_type("TAG", strength=0.3)
        beliefs.add_belief("is aggressive", Opinion(0.6, 0.2, 0.2, 0.5))

        store.save_beliefs(beliefs)
        loaded = store.load_beliefs("villain1")

        assert loaded is not None
        assert loaded.player_id == "villain1"

        # Check player type preserved
        best_type, _ = loaded.get_most_likely_player_type()
        assert loaded.player_type.beliefs["TAG"] > 0

        # Check binomial belief preserved
        agg_belief = loaded.get_belief("is aggressive")
        assert agg_belief is not None
        assert agg_belief.opinion.belief == 0.6

    def test_load_nonexistent_beliefs(self, store):
        loaded = store.load_beliefs("nonexistent")
        assert loaded is None

    def test_update_beliefs(self, store):
        beliefs = VillainBeliefs("villain1")
        store.save_beliefs(beliefs)

        # Modify and save again
        beliefs.update_player_type("Fish", strength=0.4)
        store.save_beliefs(beliefs)

        loaded = store.load_beliefs("villain1")
        assert loaded.player_type.beliefs["Fish"] > 0

    def test_load_all_beliefs(self, store):
        for i in range(3):
            beliefs = VillainBeliefs(f"villain{i}")
            store.save_beliefs(beliefs)

        all_beliefs = store.load_all_beliefs()

        assert len(all_beliefs) == 3

    def test_delete_beliefs(self, store):
        beliefs = VillainBeliefs("villain1")
        store.save_beliefs(beliefs)

        deleted = store.delete_beliefs("villain1")
        assert deleted

        loaded = store.load_beliefs("villain1")
        assert loaded is None


class TestPersistentMemoryManager:
    """Tests for PersistentMemoryManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def manager(self, temp_db):
        """Create a manager with temp database."""
        return PersistentMemoryManager(temp_db)

    def test_start_and_end_session(self, manager):
        manager.start_session("Test session")

        assert manager.session_id is not None
        assert manager.table_memory is not None

        manager.end_session(hands_played=100)

    def test_get_opponent_memory(self, manager):
        manager.start_session()

        memory = manager.get_opponent_memory("villain1")
        memory.record_preflop_action(ActionType.CALL, Position.BTN)

        # Same opponent returns same memory
        memory2 = manager.get_opponent_memory("villain1")
        assert memory is memory2

    def test_get_opponent_beliefs(self, manager):
        manager.start_session()

        # Record some actions
        memory = manager.get_opponent_memory("villain1")
        for _ in range(15):
            memory.record_preflop_action(ActionType.CALL, Position.BTN)

        beliefs = manager.get_opponent_beliefs("villain1")

        assert beliefs is not None
        assert beliefs.player_id == "villain1"

    def test_save_and_reload(self, temp_db):
        # First session
        manager1 = PersistentMemoryManager(temp_db)
        manager1.start_session()

        memory = manager1.get_opponent_memory("villain1")
        memory.record_preflop_action(ActionType.RAISE, Position.BTN)
        memory.record_preflop_action(ActionType.RAISE, Position.CO)

        manager1.save_all()
        manager1.end_session(hands_played=2)

        # Second session - should load previous data
        manager2 = PersistentMemoryManager(temp_db)
        manager2.start_session()

        loaded_memory = manager2.get_opponent_memory("villain1")

        assert loaded_memory.stats.vpip_hands == 2
        assert loaded_memory.stats.pfr_hands == 2

    def test_save_opponent(self, manager):
        manager.start_session()

        memory = manager.get_opponent_memory("villain1")
        memory.record_preflop_action(ActionType.CALL, Position.BTN)

        manager.save_opponent("villain1")

        # Verify it's saved
        loaded = manager.memory_store.load_opponent("villain1")
        assert loaded is not None

    def test_get_table_summary(self, manager):
        manager.start_session()

        manager.get_opponent_memory("v1").record_preflop_action(ActionType.CALL, Position.BTN)
        manager.get_opponent_memory("v2").record_preflop_action(ActionType.RAISE, Position.CO)

        summary = manager.get_table_summary()

        assert "v1" in summary
        assert "v2" in summary

    def test_get_database_stats(self, manager):
        manager.start_session()

        memory = manager.get_opponent_memory("villain1")
        memory.record_preflop_action(ActionType.CALL, Position.BTN)
        manager.save_opponent("villain1")

        stats = manager.get_database_stats()

        assert stats["total_opponents"] >= 1
