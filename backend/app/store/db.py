"""SQLite persistence for analyzed cases.

A thin, dependency-free store built on stdlib ``sqlite3`` + ``json``. Each
analyzed :class:`CaseResult` is upserted as a row whose ``payload`` column holds
the full Pydantic JSON, so reads round-trip losslessly via
``CaseResult.model_validate_json``.

Robust to concurrent access from the (threaded) web server:
  * every operation opens a short-lived connection (``check_same_thread=False``),
  * WAL journaling + a busy timeout reduce ``database is locked`` errors,
  * writes are wrapped in a transaction context manager.

Only stdlib is imported here, so this module always loads.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator, List, Optional

from app.api.schemas import CaseResult
from app.config import DB_PATH

__all__ = ["init_db", "save_case", "list_cases", "get_case", "count"]

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS cases (
    id      TEXT PRIMARY KEY,
    created TEXT NOT NULL,
    risk    INTEGER NOT NULL,
    payload TEXT NOT NULL
)
"""
_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created DESC)"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Yield a short-lived, concurrency-tolerant connection (auto-closed)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
        timeout=30.0,
    )
    try:
        conn.row_factory = sqlite3.Row
        # WAL lets readers and a writer coexist without blocking.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the ``cases`` table (and index) if they do not yet exist."""
    with _connect() as conn:
        conn.execute(_TABLE_DDL)
        conn.execute(_INDEX_DDL)
        conn.commit()


def _created_of(case: CaseResult, meta) -> str:
    """Pick a sortable 'created' key: prefer upload date, fall back as needed."""
    created = (getattr(case, "uploadDate", "") or "").strip()
    if created:
        return created
    # meta is opaque (AnalysisMeta or dict or None); never let it break a save.
    try:
        if isinstance(meta, dict):
            cand = meta.get("created") or meta.get("uploadDate")
            if cand:
                return str(cand)
    except Exception:  # noqa: BLE001
        pass
    return ""


def save_case(case: CaseResult, meta=None) -> None:
    """Upsert ``case`` keyed by ``case.id``. ``meta`` is optional/ignored-safe.

    ``payload`` stores the full ``case.model_dump_json()`` so reads round-trip.
    """
    payload = case.model_dump_json()
    created = _created_of(case, meta)
    risk = int(getattr(case, "riskScore", 0) or 0)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO cases (id, created, risk, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                created = excluded.created,
                risk    = excluded.risk,
                payload = excluded.payload
            """,
            (case.id, created, risk, payload),
        )
        conn.commit()


def list_cases(limit: int = 200) -> List[CaseResult]:
    """Return up to ``limit`` cases, newest first."""
    if limit <= 0:
        return []
    out: List[CaseResult] = []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT payload FROM cases
            ORDER BY created DESC, rowid DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    for row in rows:
        try:
            out.append(CaseResult.model_validate_json(row["payload"]))
        except Exception:  # noqa: BLE001 - skip a corrupt row, never crash listing
            continue
    return out


def get_case(case_id: str) -> Optional[CaseResult]:
    """Fetch a single case by id, or None if absent/corrupt."""
    if not case_id:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload FROM cases WHERE id = ?",
            (case_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        return CaseResult.model_validate_json(row["payload"])
    except Exception:  # noqa: BLE001
        return None


def count() -> int:
    """Total number of stored cases."""
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM cases").fetchone()
    return int(row["n"]) if row is not None else 0
