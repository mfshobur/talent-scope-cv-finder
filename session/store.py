import json
import sqlite3
from datetime import datetime, timezone, timedelta
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_keys (
                key_hash   TEXT PRIMARY KEY,
                label      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                is_revoked INTEGER NOT NULL DEFAULT 0
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


# ── Access key management ─────────────────────────────────────────────────────

def generate_access_key(db_path: str, label: str = "") -> str:
    import hashlib, secrets, string
    alphabet = string.ascii_uppercase + string.digits
    raw = "-".join("".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(4))
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    now = datetime.now(timezone.utc)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO access_keys (key_hash, label, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (key_hash, label, now.isoformat(), (now + timedelta(hours=24)).isoformat()),
        )
        conn.commit()
    return raw


def validate_access_key(db_path: str, raw_key: str) -> bool:
    import hashlib
    key_hash = hashlib.sha256(raw_key.strip().upper().encode()).hexdigest()
    return _validate_hash(db_path, key_hash)


def _validate_hash(db_path: str, key_hash: str) -> bool:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT expires_at, is_revoked FROM access_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
    if not row or row["is_revoked"]:
        return False
    return datetime.now(timezone.utc) < datetime.fromisoformat(row["expires_at"])


def revoke_access_key(db_path: str, key_hash: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("UPDATE access_keys SET is_revoked = 1 WHERE key_hash = ?", (key_hash,))
        conn.commit()


def list_access_keys(db_path: str) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT key_hash, label, created_at, expires_at, is_revoked "
            "FROM access_keys ORDER BY created_at DESC"
        ).fetchall()
    now = datetime.now(timezone.utc)
    result = []
    for row in rows:
        if row["is_revoked"]:
            status = "revoked"
        elif now >= datetime.fromisoformat(row["expires_at"]):
            status = "expired"
        else:
            status = "valid"
        result.append({
            "key_hash": row["key_hash"],
            "label": row["label"],
            "expires_at": row["expires_at"],
            "status": status,
        })
    return result
