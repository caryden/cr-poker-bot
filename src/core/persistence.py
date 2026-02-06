"""
SQLite persistence layer for opponent memory and beliefs.

Stores:
- Opponent statistics across sessions
- Belief states with uncertainty
- Session metadata
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

from .memory import OpponentMemory, TableMemory, OpponentStats
from .beliefs import VillainBeliefs, BeliefState, PLAYER_TYPES, DEFAULT_PLAYER_TYPE_BASE_RATES
from .subjective_logic import Opinion, MultinomialOpinion, Belief


# Default database path
DEFAULT_DB_PATH = Path.home() / ".poker_agent" / "memory.db"


class MemoryStore:
    """
    SQLite-backed persistent storage for opponent memory.

    Handles:
    - Storing and retrieving opponent statistics
    - Session tracking
    - Automatic schema migrations
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            cursor = conn.cursor()

            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # Check current version
            cursor.execute("SELECT MAX(version) FROM schema_version")
            row = cursor.fetchone()
            current_version = row[0] if row[0] else 0

            if current_version < self.SCHEMA_VERSION:
                self._migrate(conn, current_version)

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run schema migrations."""
        cursor = conn.cursor()

        if from_version < 1:
            # Initial schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS opponents (
                    opponent_id TEXT PRIMARY KEY,
                    first_seen TEXT,
                    last_seen TEXT,
                    stats_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    hands_played INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_opponents (
                    session_id INTEGER,
                    opponent_id TEXT,
                    hands_together INTEGER DEFAULT 0,
                    profit_vs REAL DEFAULT 0,
                    PRIMARY KEY (session_id, opponent_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (opponent_id) REFERENCES opponents(opponent_id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_opponents_last_seen
                ON opponents(last_seen)
            """)

            cursor.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (1,)
            )

    def save_opponent(self, memory: OpponentMemory) -> None:
        """Save or update opponent memory."""
        with self._connection() as conn:
            cursor = conn.cursor()

            stats_json = json.dumps(memory.to_dict()["stats"])
            first_seen = memory.first_seen.isoformat() if memory.first_seen else None
            last_seen = memory.last_seen.isoformat() if memory.last_seen else None

            cursor.execute("""
                INSERT INTO opponents (opponent_id, first_seen, last_seen, stats_json, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(opponent_id) DO UPDATE SET
                    first_seen = COALESCE(opponents.first_seen, excluded.first_seen),
                    last_seen = excluded.last_seen,
                    stats_json = excluded.stats_json,
                    updated_at = CURRENT_TIMESTAMP
            """, (memory.opponent_id, first_seen, last_seen, stats_json))

    def load_opponent(self, opponent_id: str) -> Optional[OpponentMemory]:
        """Load opponent memory from database."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM opponents WHERE opponent_id = ?",
                (opponent_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_memory(row)

    def load_all_opponents(self) -> dict[str, OpponentMemory]:
        """Load all opponent memories."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM opponents")

            result = {}
            for row in cursor.fetchall():
                memory = self._row_to_memory(row)
                result[memory.opponent_id] = memory

            return result

    def _row_to_memory(self, row: sqlite3.Row) -> OpponentMemory:
        """Convert database row to OpponentMemory."""
        data = {
            "opponent_id": row["opponent_id"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "stats": json.loads(row["stats_json"])
        }
        return OpponentMemory.from_dict(data)

    def save_table_memory(self, table_memory: TableMemory) -> None:
        """Save all opponents in table memory."""
        for memory in table_memory.opponents.values():
            self.save_opponent(memory)

    def load_table_memory(self) -> TableMemory:
        """Load all opponents into table memory."""
        memory = TableMemory()
        memory.opponents = self.load_all_opponents()
        return memory

    def get_recent_opponents(self, days: int = 30) -> list[OpponentMemory]:
        """Get opponents seen in the last N days."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM opponents
                WHERE last_seen >= datetime('now', ?)
                ORDER BY last_seen DESC
            """, (f'-{days} days',))

            return [self._row_to_memory(row) for row in cursor.fetchall()]

    def delete_opponent(self, opponent_id: str) -> bool:
        """Delete an opponent's data."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM opponents WHERE opponent_id = ?",
                (opponent_id,)
            )
            return cursor.rowcount > 0

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM opponents")
            total_opponents = cursor.fetchone()[0]

            cursor.execute("SELECT SUM(json_extract(stats_json, '$.vpip_opportunities')) FROM opponents")
            total_hands = cursor.fetchone()[0] or 0

            return {
                "total_opponents": total_opponents,
                "total_hands_observed": total_hands,
                "db_path": str(self.db_path),
            }

    # Session management

    def start_session(self, notes: Optional[str] = None) -> int:
        """Start a new session, return session ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (started_at, notes) VALUES (?, ?)",
                (datetime.now().isoformat(), notes)
            )
            return cursor.lastrowid

    def end_session(self, session_id: int, hands_played: int = 0) -> None:
        """End a session."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET ended_at = ?, hands_played = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), hands_played, session_id))

    def record_session_opponent(self, session_id: int, opponent_id: str,
                                hands: int = 0, profit: float = 0) -> None:
        """Record opponent interaction in a session."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_opponents (session_id, opponent_id, hands_together, profit_vs)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, opponent_id) DO UPDATE SET
                    hands_together = hands_together + excluded.hands_together,
                    profit_vs = profit_vs + excluded.profit_vs
            """, (session_id, opponent_id, hands, profit))


class BeliefStore:
    """
    Persistent storage for belief states.

    Separate from MemoryStore to allow different update frequencies.
    Beliefs are derived from memory but may also include manual adjustments.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize beliefs table."""
        with self._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    opponent_id TEXT PRIMARY KEY,
                    player_type_json TEXT NOT NULL,
                    binomial_beliefs_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_beliefs(self, beliefs: VillainBeliefs) -> None:
        """Save villain beliefs."""
        with self._connection() as conn:
            cursor = conn.cursor()

            # Serialize player type (MultinomialOpinion)
            player_type_data = {
                "beliefs": beliefs.player_type.beliefs,
                "uncertainty": beliefs.player_type.uncertainty,
                "base_rates": beliefs.player_type.base_rates,
            }

            # Serialize binomial beliefs
            binomial_data = {}
            for label, belief in beliefs.beliefs.items():
                binomial_data[label] = {
                    "b": belief.opinion.belief,
                    "d": belief.opinion.disbelief,
                    "u": belief.opinion.uncertainty,
                    "a": belief.opinion.base_rate,
                    "category": belief.category,
                }

            cursor.execute("""
                INSERT INTO beliefs (opponent_id, player_type_json, binomial_beliefs_json, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(opponent_id) DO UPDATE SET
                    player_type_json = excluded.player_type_json,
                    binomial_beliefs_json = excluded.binomial_beliefs_json,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                beliefs.player_id,
                json.dumps(player_type_data),
                json.dumps(binomial_data)
            ))

    def load_beliefs(self, opponent_id: str) -> Optional[VillainBeliefs]:
        """Load villain beliefs."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM beliefs WHERE opponent_id = ?",
                (opponent_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_beliefs(row)

    def _row_to_beliefs(self, row: sqlite3.Row) -> VillainBeliefs:
        """Convert database row to VillainBeliefs."""
        opponent_id = row["opponent_id"]
        player_type_data = json.loads(row["player_type_json"])
        binomial_data = json.loads(row["binomial_beliefs_json"])

        # Reconstruct player type
        player_type = MultinomialOpinion(
            beliefs=player_type_data["beliefs"],
            uncertainty=player_type_data["uncertainty"],
            base_rates=player_type_data["base_rates"]
        )

        # Create VillainBeliefs
        beliefs = VillainBeliefs(opponent_id)
        beliefs.player_type = player_type

        # Reconstruct binomial beliefs
        for label, data in binomial_data.items():
            opinion = Opinion(
                data["b"], data["d"], data["u"], data["a"]
            )
            beliefs.add_belief(label, opinion, data.get("category", "general"))

        return beliefs

    def load_all_beliefs(self) -> dict[str, VillainBeliefs]:
        """Load all saved beliefs."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM beliefs")

            result = {}
            for row in cursor.fetchall():
                beliefs = self._row_to_beliefs(row)
                result[beliefs.player_id] = beliefs

            return result

    def delete_beliefs(self, opponent_id: str) -> bool:
        """Delete beliefs for an opponent."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM beliefs WHERE opponent_id = ?",
                (opponent_id,)
            )
            return cursor.rowcount > 0


class PersistentMemoryManager:
    """
    High-level manager combining memory and belief persistence.

    Provides a unified interface for the agent to:
    - Load opponent data at session start
    - Save data periodically and at session end
    - Merge memory-derived beliefs with stored beliefs
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.memory_store = MemoryStore(db_path)
        self.belief_store = BeliefStore(db_path)
        self.table_memory: Optional[TableMemory] = None
        self.session_id: Optional[int] = None

    def start_session(self, notes: Optional[str] = None) -> None:
        """Start a new session and load existing data."""
        self.session_id = self.memory_store.start_session(notes)
        self.table_memory = self.memory_store.load_table_memory()

    def end_session(self, hands_played: int = 0) -> None:
        """End session and persist all data."""
        if self.table_memory:
            self.memory_store.save_table_memory(self.table_memory)

        if self.session_id:
            self.memory_store.end_session(self.session_id, hands_played)

    def get_opponent_memory(self, opponent_id: str) -> OpponentMemory:
        """Get memory for an opponent (creates if not exists)."""
        if self.table_memory is None:
            self.table_memory = TableMemory()
        return self.table_memory.get_opponent(opponent_id)

    def get_opponent_beliefs(self, opponent_id: str) -> VillainBeliefs:
        """
        Get beliefs for an opponent.

        Merges stored beliefs with memory-derived beliefs, preferring
        more recent observations.
        """
        # Try to get stored beliefs
        stored_beliefs = self.belief_store.load_beliefs(opponent_id)

        # Get memory-derived beliefs
        memory = self.get_opponent_memory(opponent_id)
        memory_beliefs = memory.to_beliefs()

        if stored_beliefs is None:
            return memory_beliefs

        # Merge: use memory-derived if we have recent observations
        # Otherwise use stored beliefs
        if memory.stats.total_hands > 10:
            # Enough new data - prefer memory-derived
            return memory_beliefs
        else:
            # Not much new data - blend with stored
            # For now, just return stored; could implement fusion later
            return stored_beliefs

    def save_opponent(self, opponent_id: str) -> None:
        """Save a specific opponent's data."""
        memory = self.get_opponent_memory(opponent_id)
        self.memory_store.save_opponent(memory)

        beliefs = memory.to_beliefs()
        self.belief_store.save_beliefs(beliefs)

    def save_all(self) -> None:
        """Save all opponent data."""
        if self.table_memory:
            for opponent_id in self.table_memory.opponents:
                self.save_opponent(opponent_id)

    def get_table_summary(self) -> str:
        """Get summary of all opponents at table."""
        if self.table_memory:
            return self.table_memory.get_all_summaries()
        return "No opponent data available."

    def get_database_stats(self) -> dict:
        """Get database statistics."""
        return self.memory_store.get_stats()
