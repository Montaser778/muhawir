"""Session store.

Right now this is an in-memory dictionary plus files on disk. That is
deliberate: every read and write goes through this one module, so swapping
it for Postgres later means rewriting this file and nothing else.

The interface is the important part, not the implementation:

    create(session_id, role, language)
    set_status(session_id, status)
    save(session_id, report)
    get(session_id) -> SessionRecord | None
    list_recent(limit) -> list[SessionRecord]
"""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Status = Literal["running", "scoring", "ready", "empty", "error"]

REPORTS_DIR = Path("reports")


@dataclass
class SessionRecord:
    session_id: str
    role: str
    language: str
    status: Status = "running"
    created_at: float = field(default_factory=time.time)
    report: dict | None = None
    error: str | None = None

    def public(self) -> dict:
        """What the browser is allowed to see."""
        return {
            "session_id": self.session_id,
            "role": self.role,
            "language": self.language,
            "status": self.status,
            "created_at": self.created_at,
            "report": self.report,
            "error": self.error,
        }

    def summary(self) -> dict:
        """Lightweight row for listings."""
        return {
            "session_id": self.session_id,
            "role": self.role,
            "language": self.language,
            "status": self.status,
            "created_at": self.created_at,
            "overall_score": (self.report or {}).get("overall_score"),
            "questions_answered": (self.report or {}).get("questions_answered"),
        }


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionRecord] = {}
        self._load_from_disk()

    # --- writes ------------------------------------------------------
    def create(self, session_id: str, role: str, language: str) -> SessionRecord:
        with self._lock:
            record = SessionRecord(session_id=session_id, role=role, language=language)
            self._sessions[session_id] = record
            return record

    def set_status(self, session_id: str, status: Status, error: str | None = None) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record:
                record.status = status
                if error:
                    record.error = error

    def save(self, session_id: str, report: dict) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                record = SessionRecord(
                    session_id=session_id,
                    role=report.get("role", ""),
                    language=report.get("language", ""),
                )
                self._sessions[session_id] = record
            record.report = report
            record.status = "ready"

    # --- reads -------------------------------------------------------
    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_recent(self, limit: int = 50) -> list[SessionRecord]:
        with self._lock:
            records = sorted(
                self._sessions.values(), key=lambda r: r.created_at, reverse=True
            )
            return records[:limit]

    # --- durability --------------------------------------------------
    def _load_from_disk(self) -> None:
        """Rehydrate finished sessions so a restart doesn't lose reports."""
        if not REPORTS_DIR.exists():
            return
        for path in REPORTS_DIR.glob("*.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            session_id = report.get("session_id") or path.stem
            self._sessions[session_id] = SessionRecord(
                session_id=session_id,
                role=report.get("role", ""),
                language=report.get("language", ""),
                status="ready",
                created_at=path.stat().st_mtime,
                report=report,
            )


store = SessionStore()
