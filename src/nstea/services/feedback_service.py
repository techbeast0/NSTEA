"""Feedback service — stores and retrieves clinician feedback.

Uses a JSON-file backend for now; can be swapped for PostgreSQL later.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from threading import Lock

from nstea.config import settings
from nstea.models.feedback import FeedbackInput, FeedbackRecord

_FEEDBACK_FILE = settings.project_root / "data" / "feedback.json"
_lock = Lock()


def _load_all() -> list[dict]:
    if not _FEEDBACK_FILE.exists():
        return []
    with open(_FEEDBACK_FILE) as f:
        return json.load(f)


def _save_all(records: list[dict]) -> None:
    _FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_FEEDBACK_FILE, "w") as f:
        json.dump(records, f, indent=2, default=str)


def submit_feedback(feedback: FeedbackInput) -> FeedbackRecord:
    """Store clinician feedback and return the persisted record."""
    record = FeedbackRecord(
        feedback_id=str(uuid.uuid4())[:8],
        **feedback.model_dump(),
    )
    with _lock:
        all_records = _load_all()
        all_records.append(record.model_dump(mode="json"))
        _save_all(all_records)
    return record


def get_feedback_for_analysis(analysis_id: str) -> list[FeedbackRecord]:
    """Retrieve all feedback for a given analysis."""
    all_records = _load_all()
    return [
        FeedbackRecord(**r)
        for r in all_records
        if r["analysis_id"] == analysis_id
    ]


def get_feedback_summary() -> dict:
    """Return aggregate feedback statistics."""
    all_records = _load_all()
    total = len(all_records)
    if total == 0:
        return {"total": 0, "accept": 0, "modify": 0, "reject": 0, "accept_rate": 0.0}

    verdicts = {"accept": 0, "modify": 0, "reject": 0}
    for r in all_records:
        v = r.get("verdict", "reject")
        verdicts[v] = verdicts.get(v, 0) + 1

    return {
        "total": total,
        **verdicts,
        "accept_rate": round(verdicts["accept"] / total, 2),
    }
