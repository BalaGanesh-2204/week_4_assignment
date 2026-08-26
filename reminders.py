"""
Reminder and escalation queue for the human support lead.

Everything lands in data/reminders.json. Escalations are
high-urgency entries raised by tools or guardrail events;
ordinary reminders are notes the agent leaves for review.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import DATA_DIR, SUPPORT_LEAD


REMINDERS_FILE = DATA_DIR / "reminders.json"

VALID_URGENCIES = ("low", "medium", "high", "critical")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> List[Dict]:
    """
    Read all reminder entries (empty list if none yet).
    """

    if not REMINDERS_FILE.exists():
        return []

    try:
        return json.loads(
            REMINDERS_FILE.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        # Corrupt file should never silently lose data - keep a backup.
        backup = REMINDERS_FILE.with_suffix(".corrupt.bak")
        REMINDERS_FILE.replace(backup)
        return []


def _save(entries: List[Dict]):
    """
    Persist reminder entries.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    REMINDERS_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def add_entry(
    entry_type: str,
    message: str,
    urgency: str = "medium",
    meta: Optional[Dict] = None,
) -> Dict:
    """
    Add a reminder or escalation entry.
    """

    if urgency not in VALID_URGENCIES:
        urgency = "medium"

    entry = {
        "id": uuid.uuid4().hex[:10],
        "created_at": _now(),
        "type": entry_type,
        "message": message.strip(),
        "urgency": urgency,
        "status": "open",
        "meta": meta or {},
    }

    entries = _load()
    entries.append(entry)
    _save(entries)

    return entry


def add_reminder(message: str, urgency: str = "medium") -> Dict:
    """
    Ordinary note for the support lead.
    """

    return add_entry("reminder", message, urgency)


def raise_escalation(
    summary: str,
    reason: str = "",
    meta: Optional[Dict] = None,
) -> Dict:
    """
    Human handoff: always critical urgency so it surfaces first.
    """

    message = summary
    if reason:
        message = f"{summary} | Reason: {reason}"

    return add_entry(
        "escalation",
        f"[For {SUPPORT_LEAD}] {message}",
        urgency="critical",
        meta=meta,
    )


def resolve_entry(entry_id: str) -> bool:
    """
    Mark an entry resolved. Returns True when found.
    """

    entries = _load()

    for entry in entries:

        if entry["id"] == entry_id:

            entry["status"] = "resolved"
            entry["resolved_at"] = _now()
            _save(entries)
            return True

    return False


def open_entries() -> List[Dict]:
    """
    Open entries, most urgent and newest first.
    """

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    open_items = [
        e for e in _load() if e.get("status") == "open"
    ]

    open_items.sort(
        key=lambda e: (
            order.get(e.get("urgency", "medium"), 2),
            e.get("created_at", ""),
        ),
        reverse=False,
    )

    # Newest within same urgency: reverse created_at inside urgency groups
    open_items.sort(
        key=lambda e: order.get(e.get("urgency", "medium"), 2),
        reverse=False,
    )

    return open_items


def counts() -> Dict[str, int]:
    """
    Quick counters for UI badges.
    """

    entries = [e for e in _load() if e.get("status") == "open"]

    return {
        "total": len(entries),
        "escalations": sum(
            1 for e in entries if e.get("type") == "escalation"
        ),
        "reminders": sum(
            1 for e in entries if e.get("type") == "reminder"
        ),
        "critical": sum(
            1 for e in entries if e.get("urgency") == "critical"
        ),
    }
