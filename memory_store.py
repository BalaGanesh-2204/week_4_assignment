"""
Persistent cross-run memory backed by SQLite.

Stores:
- facts   : durable key-value notes about a customer
- sessions: registry of chat sessions and turn counts
"""

import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import DATA_DIR


DB_PATH = DATA_DIR / "memory.db"


def _connect() -> sqlite3.Connection:
    """
    Open a short-lived connection with the schema applied.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            customer_key TEXT NOT NULL,
            key          TEXT NOT NULL,
            value        TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            PRIMARY KEY (customer_key, key)
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            turns      INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remember_fact(customer_key: str, key: str, value: str):
    """
    Insert or update a fact for a customer.
    """

    with _connect() as connection:

        connection.execute(
            """
            INSERT INTO facts (customer_key, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(customer_key, key)
            DO UPDATE SET value = excluded.value,
                          updated_at = excluded.updated_at
            """,
            (customer_key.strip(), key.strip(), value.strip(), _now()),
        )


def recall_facts(
    customer_key: str,
    key: Optional[str] = None,
) -> List[Dict]:
    """
    Recall all facts for a customer, or one specific fact.
    """

    query = (
        "SELECT key, value, updated_at FROM facts "
        "WHERE customer_key = ?"
    )
    params = [customer_key.strip()]

    if key:
        query += " AND key = ?"
        params.append(key.strip())

    query += " ORDER BY updated_at DESC"

    with _connect() as connection:

        rows = connection.execute(query, params).fetchall()

    return [
        {"key": row[0], "value": row[1], "updated_at": row[2]}
        for row in rows
    ]


def forget_customer(customer_key: str) -> int:
    """
    Delete all facts for a customer. Returns deleted count.
    """

    with _connect() as connection:

        cursor = connection.execute(
            "DELETE FROM facts WHERE customer_key = ?",
            (customer_key.strip(),),
        )

        return cursor.rowcount


def register_session(session_id: str):
    """
    Register a new session id (idempotent).
    """

    with _connect() as connection:

        connection.execute(
            """
            INSERT OR IGNORE INTO sessions (session_id, started_at, turns)
            VALUES (?, ?, 0)
            """,
            (session_id, _now()),
        )


def increment_session_turn(session_id: str) -> int:
    """
    Increment the stored turn counter for a session.
    Returns the new count.
    """

    register_session(session_id)

    with _connect() as connection:

        connection.execute(
            "UPDATE sessions SET turns = turns + 1 WHERE session_id = ?",
            (session_id,),
        )

        row = connection.execute(
            "SELECT turns FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    return int(row[0]) if row else 0


def list_sessions(limit: int = 50) -> List[Dict]:
    """
    Most recent sessions first.
    """

    with _connect() as connection:

        rows = connection.execute(
            """
            SELECT session_id, started_at, turns
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {"session_id": r[0], "started_at": r[1], "turns": r[2]}
        for r in rows
    ]
