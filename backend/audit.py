
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "approval_audit.db"


def log_audit_event(
    request_id,
    stage,
    reviewer,
    decision,
    comments="",
):
    """Persist an approval event in SQLite."""

    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                timestamp TEXT,
                stage TEXT,
                reviewer TEXT,
                decision TEXT,
                comments TEXT
            )
        """)

        conn.execute("""
            INSERT INTO audit_log (
                request_id, timestamp, stage,
                reviewer, decision, comments
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            datetime.now(timezone.utc).isoformat(),
            stage,
            reviewer,
            decision,
            comments,
        ))


def get_audit_trail(request_id):
    """Retrieve audit history for a request."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM audit_log
            WHERE request_id = ?
            ORDER BY id
        """, (request_id,)).fetchall()

    return [dict(row) for row in rows]
