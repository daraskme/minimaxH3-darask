from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    phase TEXT NOT NULL DEFAULT '', request_json TEXT NOT NULL,
                    resolved_json TEXT NOT NULL, output_path TEXT, metadata_path TEXT,
                    error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
                )"""
            )
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(jobs)").fetchall()}
            if "kind" not in columns:
                self._db.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'generate'")

    def create(
        self, job_id: str, request: dict[str, Any], resolved: dict[str, Any], kind: str = "generate",
    ) -> dict[str, Any]:
        now = utcnow()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO jobs (id,status,request_json,resolved_json,created_at,updated_at,kind) VALUES (?,?,?,?,?,?,?)",
                (job_id, "queued", json.dumps(request, ensure_ascii=False), json.dumps(resolved, ensure_ascii=False), now, now, kind),
            )
        return self.get(job_id)

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {"status", "progress", "phase", "output_path", "metadata_path", "error", "cancel_requested", "started_at", "finished_at"}
        fields = {key: value for key, value in fields.items() if key in allowed}
        fields["updated_at"] = utcnow()
        query = ", ".join(f"{key}=?" for key in fields)
        with self._lock, self._db:
            self._db.execute(f"UPDATE jobs SET {query} WHERE id=?", (*fields.values(), job_id))
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._decode(row)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._decode(row) for row in rows]

    def recover_interrupted(self) -> None:
        now = utcnow()
        with self._lock, self._db:
            self._db.execute(
                "UPDATE jobs SET status='failed', error='アプリの再起動により生成が中断されました', finished_at=?, updated_at=? WHERE status IN ('running','cancelling')",
                (now, now),
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["request"] = json.loads(data.pop("request_json"))
        data["resolved"] = json.loads(data.pop("resolved_json"))
        data["cancel_requested"] = bool(data["cancel_requested"])
        return data
