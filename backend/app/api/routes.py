"""HTTP API surface — a single ``APIRouter`` mounted under ``/api``.

Endpoints:

  GET  /api/health        -> engine status + capabilities snapshot
  POST /api/analyze       -> multipart upload (heavy multimodal cascade)
  POST /api/analyze/url   -> analyze-by-reference (lightweight, no download)
  GET  /api/cases         -> recently analyzed cases (newest first)
  GET  /api/cases/{id}    -> a single stored case

Uploads are size-checked against ``settings.max_upload_mb`` and streamed to
``settings.UPLOAD_DIR`` before analysis. The orchestrator never raises, so the
routes stay thin.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import app as app_pkg
from app.api.schemas import (
    AnalysisResponse,
    AnalyzeUrlRequest,
    CaseResult,
    HealthResponse,
)
from app.config import UPLOAD_DIR, settings
from app.models import registry
from app.pipeline import orchestrator
from app.store import db

router = APIRouter(prefix="/api")

# Read uploads in bounded chunks so a large file never balloons memory.
_CHUNK = 1024 * 1024  # 1 MiB


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + capability probe (which ML lanes can run right now)."""
    return HealthResponse(
        status="ok",
        version=app_pkg.__version__,
        engineMode=registry.engine_mode(),
        capabilities=registry.capabilities(),
    )


# --------------------------------------------------------------------------- #
# Analyze an uploaded video (full cascade)
# --------------------------------------------------------------------------- #
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    platform: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    hashtags: Optional[str] = Form(None),
) -> AnalysisResponse:
    """Accept a multipart upload, persist it, and run the full analysis cascade."""
    dest = UPLOAD_DIR / f"upload_{uuid.uuid4().hex}{_suffix(file.filename)}"
    size = await _save_upload(file, dest)

    media = orchestrator.make_media_from_upload(
        path=str(dest),
        filename=file.filename or dest.name,
        size=size,
        title=title or "",
        platform=platform,
        description=description or "",
        hashtags=_split_hashtags(hashtags),
    )
    return orchestrator.analyze(media)


# --------------------------------------------------------------------------- #
# Analyze by reference (no heavy download)
# --------------------------------------------------------------------------- #
@router.post("/analyze/url", response_model=AnalysisResponse)
def analyze_url(req: AnalyzeUrlRequest) -> AnalysisResponse:
    """Analyze using only supplied metadata / captions — never downloads media."""
    media = orchestrator.make_media_from_url(req)
    return orchestrator.analyze(media)


# --------------------------------------------------------------------------- #
# Stored cases
# --------------------------------------------------------------------------- #
@router.get("/cases", response_model=List[CaseResult])
def list_cases() -> List[CaseResult]:
    """Return recently analyzed cases, newest first."""
    try:
        return db.list_cases()
    except Exception:  # noqa: BLE001 - empty list beats a 500 on a cold DB
        return []


@router.get("/cases/{case_id}", response_model=CaseResult)
def get_case(case_id: str) -> CaseResult:
    """Return a single stored case, or 404 if it does not exist."""
    case = db.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Случай не найден")
    return case


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _suffix(filename: Optional[str]) -> str:
    """Safe lowercase extension (with dot) of an uploaded filename, or ''."""
    if not filename:
        return ""
    suffix = Path(filename).suffix
    return suffix if len(suffix) <= 16 else ""


async def _save_upload(file: UploadFile, dest: Path) -> int:
    """Stream the upload to ``dest`` in chunks, enforcing the size limit.

    Raises ``HTTPException(413)`` and removes the partial file when the upload
    exceeds ``settings.max_upload_mb``.
    """
    max_bytes = settings.max_upload_mb * 1024 * 1024
    total = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    out.close()
                    _safe_unlink(dest)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл превышает лимит {settings.max_upload_mb} МБ",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _safe_unlink(dest)
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки файла: {exc}")
    finally:
        await file.close()

    if total == 0:
        _safe_unlink(dest)
        raise HTTPException(status_code=400, detail="Пустой файл")
    return total


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _split_hashtags(raw: Optional[str]) -> List[str]:
    """Parse a comma-separated hashtag form field into a clean list."""
    if not raw:
        return []
    out: List[str] = []
    seen = set()
    for part in raw.split(","):
        tag = part.strip()
        if not tag:
            continue
        norm = "#" + tag.lstrip("#")
        low = norm.lower()
        if len(norm) > 1 and low not in seen:
            seen.add(low)
            out.append(norm)
    return out
