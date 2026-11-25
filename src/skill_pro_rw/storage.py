"""Thread-safe SQLite persistence for Skills, Experiences, Baselines, and Audit Trails."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from .models import BackendEvent, EvolutionAuditEntry, Experience, Skill, utc_now_iso


class SkillStore:
    """Persistent storage layer backed by SQLite with thread-safe locking."""

    def __init__(self, db_path: str = "data/skill_pro.db") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._event_listeners: list[Callable[[BackendEvent], None]] = []

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and enable WAL mode for file-backed SQLite concurrency."""
        with self._lock:
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiences (
                    episode_id TEXT PRIMARY KEY,
                    seq INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS baselines (
                    env_name TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id TEXT PRIMARY KEY,
                    seq INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection safely."""
        with self._lock:
            self._conn.close()

    def register_listener(self, callback: Callable[[BackendEvent], None]) -> None:
        """Register a listener invoked whenever a new BackendEvent is persisted."""
        with self._lock:
            self._event_listeners.append(callback)

    def unregister_listener(self, callback: Callable[[BackendEvent], None]) -> None:
        """Remove a previously registered event listener."""
        with self._lock:
            if callback in self._event_listeners:
                self._event_listeners.remove(callback)

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def upsert_skill(self, skill: Skill) -> None:
        """Insert or update a Skill record."""
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO skills (id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                (skill.id, skill.model_dump_json(), now),
            )
            self._conn.commit()

    def upsert_skills(self, skills: list[Skill]) -> None:
        """Batch upsert multiple skills atomically."""
        now = utc_now_iso()
        rows = [(s.id, s.model_dump_json(), now) for s in skills]
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO skills (id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                rows,
            )
            self._conn.commit()

    def delete_skill(self, skill_id: str) -> None:
        """Remove a skill by ID."""
        with self._lock:
            self._conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            self._conn.commit()

    def replace_all_skills(self, skills: list[Skill]) -> None:
        """Atomically replace the entire skills table after pool maintenance."""
        now = utc_now_iso()
        rows = [(s.id, s.model_dump_json(), now) for s in skills]
        with self._lock:
            self._conn.execute("DELETE FROM skills")
            self._conn.executemany(
                "INSERT INTO skills (id, payload_json, updated_at) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def list_skills(self) -> list[Skill]:
        """Load all persisted Skills."""
        with self._lock:
            cur = self._conn.execute("SELECT payload_json FROM skills ORDER BY rowid ASC")
            return [Skill.model_validate_json(row["payload_json"]) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Experiences & Baselines
    # ------------------------------------------------------------------

    def save_experience(self, exp: Experience) -> None:
        """Persist a completed episode experience."""
        with self._lock:
            cur = self._conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM experiences")
            next_seq = int(cur.fetchone()["next_seq"])
            self._conn.execute(
                """
                INSERT INTO experiences (episode_id, seq, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET payload_json=excluded.payload_json
                """,
                (exp.episode_id, next_seq, exp.model_dump_json(), exp.created_at),
            )
            self._conn.commit()

    def list_experiences(self, limit: int | None = None) -> list[Experience]:
        """Return persisted experiences ordered from oldest to newest (or latest N)."""
        with self._lock:
            if limit is None:
                cur = self._conn.execute("SELECT payload_json FROM experiences ORDER BY seq ASC")
                return [Experience.model_validate_json(row["payload_json"]) for row in cur.fetchall()]
            cur = self._conn.execute(
                """
                SELECT payload_json FROM (
                    SELECT seq, payload_json FROM experiences ORDER BY seq DESC LIMIT ?
                ) ORDER BY seq ASC
                """,
                (limit,),
            )
            return [Experience.model_validate_json(row["payload_json"]) for row in cur.fetchall()]

    def get_baselines(self) -> dict[str, float]:
        """Return all per-category EMA reward baselines."""
        with self._lock:
            cur = self._conn.execute("SELECT env_name, value FROM baselines")
            return {row["env_name"]: float(row["value"]) for row in cur.fetchall()}

    def set_baseline(self, env_name: str, value: float) -> None:
        """Persist updated EMA baseline for an incident category."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO baselines (env_name, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(env_name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (env_name, value, utc_now_iso()),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # PPO Gate Audit Logs & Events
    # ------------------------------------------------------------------

    def save_audit_log(self, entry: EvolutionAuditEntry) -> None:
        """Persist a Non-Parametric PPO candidate verification record."""
        with self._lock:
            cur = self._conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM audit_logs")
            next_seq = int(cur.fetchone()["next_seq"])
            self._conn.execute(
                """
                INSERT INTO audit_logs (audit_id, seq, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (entry.audit_id, next_seq, entry.model_dump_json(), entry.created_at),
            )
            self._conn.commit()

    def list_audit_logs(self, limit: int = 50) -> list[EvolutionAuditEntry]:
        """Return latest evolution audit logs ordered newest first."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT payload_json FROM audit_logs ORDER BY seq DESC LIMIT ?",
                (limit,),
            )
            return [EvolutionAuditEntry.model_validate_json(row["payload_json"]) for row in cur.fetchall()]

    def append_event(self, event_type: str, payload: dict[str, object]) -> BackendEvent:
        """Persist and broadcast a structured live event for the observability UI."""
        created_at = utc_now_iso()
        serialized = json.dumps(payload, default=str)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO run_events (event_type, payload_json, created_at) VALUES (?, ?, ?)",
                (event_type, serialized, created_at),
            )
            seq = int(cur.lastrowid or 0)
            self._conn.commit()
            event = BackendEvent(seq=seq, event_type=event_type, payload=dict(payload), created_at=created_at)
            listeners = list(self._event_listeners)

        # Notify SSE subscribers outside DB lock to prevent deadlocks
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # Listener teardown must never break backend execution
                continue
        return event

    def list_events(self, limit: int = 100) -> list[BackendEvent]:
        """Return recent backend events ordered newest first."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT seq, event_type, payload_json, created_at FROM run_events ORDER BY seq DESC LIMIT ?",
                (limit,),
            )
            events: list[BackendEvent] = []
            for row in cur.fetchall():
                events.append(
                    BackendEvent(
                        seq=int(row["seq"]),
                        event_type=str(row["event_type"]),
                        payload=json.loads(row["payload_json"]),
                        created_at=str(row["created_at"]),
                    )
                )
            return events

    def get_iteration(self) -> int:
        """Return the current outer evolution iteration index."""
        with self._lock:
            cur = self._conn.execute("SELECT value FROM meta WHERE key = 'iteration'")
            row = cur.fetchone()
            return int(row["value"]) if row else 0

    def set_iteration(self, iteration: int) -> None:
        """Persist the outer evolution iteration index."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO meta (key, value) VALUES ('iteration', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(iteration),),
            )
            self._conn.commit()

    def reset_all(self) -> None:
        """Clear all persisted state (used for sandbox reset)."""
        with self._lock:
            self._conn.executescript(
                """
                DELETE FROM skills;
                DELETE FROM experiences;
                DELETE FROM baselines;
                DELETE FROM audit_logs;
                DELETE FROM run_events;
                DELETE FROM meta;
                """
            )
            self._conn.commit()
