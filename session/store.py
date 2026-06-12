import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                messages_json TEXT NOT NULL DEFAULT '[]',
                updated_at  TEXT NOT NULL
            )
        """)
        conn.commit()


def load_history(db_path: str, session_id: str) -> list[dict]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT messages_json FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return []
    return json.loads(row["messages_json"])


def save_history(db_path: str, session_id: str, messages: list[dict]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, messages_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                messages_json = excluded.messages_json,
                updated_at    = excluded.updated_at
            """,
            (session_id, json.dumps(messages, ensure_ascii=False), now),
        )
        conn.commit()


def clear_session(db_path: str, session_id: str) -> None:
    save_history(db_path, session_id, [])
