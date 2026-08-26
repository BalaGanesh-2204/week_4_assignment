"""
Per-session, per-turn logging.

Every turn writes one JSON file under logs/{session_id}/turn_NNN.json
containing the user input, guardrail verdicts, every agent step with
tool calls and latencies, token usage and the final structured answer.

These files are the audit trail used to debug misbehaving turns.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from config import LOGS_DIR


def new_session_id() -> str:
    """
    Generate a fresh session id.
    """

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import uuid

    suffix = uuid.uuid4().hex[:6]

    return f"s_{stamp}_{suffix}"


def session_dir(session_id: str) -> Path:
    return LOGS_DIR / session_id


def next_turn_number(session_id: str) -> int:
    """
    Turn numbers are derived from existing log files.
    """

    directory = session_dir(session_id)

    if not directory.exists():
        return 1

    existing = list(directory.glob("turn_*.json"))

    return len(existing) + 1


def write_turn_log(session_id: str, payload: Dict) -> Path:
    """
    Write one turn's full trace to its own JSON file.
    """

    turn_number = payload.get("turn", next_turn_number(session_id))

    payload.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
    payload["session_id"] = session_id
    payload["turn"] = turn_number

    directory = session_dir(session_id)

    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"turn_{turn_number:03d}.json"

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return path


def get_sessions() -> List[str]:
    """
    Session ids that have at least one logged turn, newest first.
    """

    if not LOGS_DIR.exists():
        return []

    sessions = [
        d.name
        for d in LOGS_DIR.iterdir()
        if d.is_dir() and any(d.glob("turn_*.json"))
    ]

    sessions.sort(reverse=True)

    return sessions


def get_turn_logs(session_id: str) -> List[Path]:
    """
    All turn files of a session in order.
    """

    directory = session_dir(session_id)

    if not directory.exists():
        return []

    return sorted(directory.glob("turn_*.json"))


def read_log(path: Path) -> Dict:
    """
    Load one turn log file.
    """

    return json.loads(path.read_text(encoding="utf-8"))
