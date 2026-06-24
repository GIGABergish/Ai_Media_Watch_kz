"""Model-artifact registry for the custom risk model (``app.ml.registry``).

The single source of truth for *which* trained ``.npz`` artifact the engine
serves. It persists a small ``ACTIVE.json`` pointer under ``ML_DIR``, keeps a
per-version copy of every artifact it saves (weights + meta + metrics) so a
previous version can be inspected or re-activated, and exposes a **cached**,
**lazy**, **never-raising** :func:`load_active` for the serving firewall.

Design guarantees (DESIGN §14, the non-negotiable serving firewall):

* :func:`load_active` imports ``numpy`` and ``app.ml.model_np`` **lazily** and
  returns ``None`` on ``ImportError`` or a missing/corrupt artifact — it **never
  raises**, so the orchestrator always keeps its rule-based fallback (lite mode).
* The loaded model is **cached** keyed by the active version; the cache is
  invalidated automatically when :func:`save_artifact` activates a new version.

This is intentionally distinct from :mod:`app.models.registry`, which caches
heavy *optional capability* singletons (whisper / clip / ffmpeg). This module is
only about the learned-model artifact lifecycle.
"""
from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from app.ml.config import MLConfig, ml_config

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime numpy import
    from app.ml.types import RiskModel

# Layout under ML_DIR:
#   ML_DIR/ACTIVE.json                 -> pointer {version, paths, saved_at, metrics}
#   ML_DIR/risk_model.npz / .json      -> the ACTIVE weights + meta (default cfg paths)
#   ML_DIR/versions/<version>/         -> immutable per-version copy (model/meta/metrics)
_VERSIONS_DIRNAME = "versions"
_ARTIFACT_NPZ = "model.npz"
_ARTIFACT_META = "meta.json"
_ARTIFACT_METRICS = "metrics.json"

# Process-wide lock so concurrent save/activate/load stay consistent.
_LOCK = threading.RLock()

# Cache: (active_version_string) -> loaded RiskModel. Keyed by version so a fresh
# save_artifact (which bumps the active version stamp) transparently invalidates.
_CACHE: Dict[str, "RiskModel"] = {}


# --------------------------------------------------------------------------- #
# Pointer (ACTIVE.json) helpers
# --------------------------------------------------------------------------- #
def _ml_dir(cfg: MLConfig) -> Path:
    """Directory holding all artifacts (parent of ``cfg.active_pointer``)."""
    return Path(cfg.active_pointer).resolve().parent


def _pointer_path(cfg: MLConfig) -> Path:
    return Path(cfg.active_pointer)


def _versions_dir(cfg: MLConfig) -> Path:
    return _ml_dir(cfg) / _VERSIONS_DIRNAME


def _read_pointer(cfg: MLConfig = ml_config) -> Optional[dict]:
    """Load ``ACTIVE.json`` or ``None`` (missing / unreadable / malformed)."""
    path = _pointer_path(cfg)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_pointer(cfg: MLConfig, payload: dict) -> None:
    """Atomically write ``ACTIVE.json`` (write-temp-then-replace)."""
    path = _pointer_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _cache_key(cfg: MLConfig) -> Optional[str]:
    """Active-version string used as the load cache key, or ``None``."""
    ptr = _read_pointer(cfg)
    if not ptr:
        return None
    # ``saved_at`` makes the key change even if the version string is reused
    # across retrains, so a stale cached model is never served.
    version = str(ptr.get("version", "")) or None
    if version is None:
        return None
    return f"{version}@{ptr.get('saved_at', '')}"


# --------------------------------------------------------------------------- #
# Saving / activating an artifact
# --------------------------------------------------------------------------- #
def save_artifact(
    model: "RiskModel",
    metrics: Optional[dict] = None,
    cfg: MLConfig = ml_config,
) -> str:
    """Persist ``model`` as a new version and make it the ACTIVE artifact.

    Writes the served weights + meta to the default ``cfg.model_path`` /
    ``cfg.meta_path`` (so :meth:`NpRiskModel.load` with default args resolves the
    active model), keeps an immutable copy under ``ML_DIR/versions/<version>/``
    (weights, meta, and the ``metrics`` json), updates ``ACTIVE.json`` to point at
    it, and invalidates the in-process load cache.

    Returns the absolute path to the active ``.npz`` (``cfg.model_path``).
    """
    with _LOCK:
        ml_dir = _ml_dir(cfg)
        ml_dir.mkdir(parents=True, exist_ok=True)

        # 1) Serve from the default paths so load_active() == NpRiskModel.load().
        active_npz = Path(cfg.model_path)
        model.save(str(active_npz))          # writes cfg.model_path + cfg.meta_path
        active_meta = Path(cfg.meta_path)

        # 2) Resolve the version (prefer the model's own stamp, fall back to cfg).
        version = str(getattr(model, "model_version", "") or cfg.version)
        saved_at = datetime.now(timezone.utc).isoformat()

        # 3) Immutable per-version copy (best-effort; never blocks activation).
        vdir = _versions_dir(cfg) / _safe_name(version)
        version_npz = vdir / _ARTIFACT_NPZ
        version_meta = vdir / _ARTIFACT_META
        version_metrics = vdir / _ARTIFACT_METRICS
        try:
            vdir.mkdir(parents=True, exist_ok=True)
            if active_npz.exists():
                shutil.copyfile(active_npz, version_npz)
            if active_meta.exists():
                shutil.copyfile(active_meta, version_meta)
            with open(version_metrics, "w", encoding="utf-8") as fh:
                json.dump(metrics or {}, fh, ensure_ascii=False, indent=2)
        except OSError:
            # Versioned history is a convenience; the active pointer is what
            # serving needs and is written below regardless.
            pass

        # 4) Flip the ACTIVE pointer.
        pointer = {
            "version": version,
            "saved_at": saved_at,
            "model_path": str(active_npz),
            "meta_path": str(active_meta),
            "version_dir": str(vdir),
            "metrics": _compact_metrics(metrics),
        }
        _write_pointer(cfg, pointer)

        # 5) Invalidate cache so the next load_active picks up the new weights.
        _CACHE.clear()

        return str(active_npz)


# --------------------------------------------------------------------------- #
# Loading the active model (cached, lazy, never raises)
# --------------------------------------------------------------------------- #
def load_active(cfg: MLConfig = ml_config) -> Optional["RiskModel"]:
    """Return the ACTIVE :class:`RiskModel`, or ``None`` — **never raises**.

    Cached by active version. Imports ``numpy`` and ``app.ml.model_np`` lazily so
    this module (and the engine) stay importable without numpy; returns ``None``
    on ``ImportError`` or any missing/corrupt artifact, letting the orchestrator
    fall back to the rule engine.
    """
    try:
        with _LOCK:
            key = _cache_key(cfg)
            if key is None:
                return None
            cached = _CACHE.get(key)
            if cached is not None:
                return cached

            ptr = _read_pointer(cfg)
            if not ptr:
                return None
            npz_path = str(ptr.get("model_path") or cfg.model_path)
            if not Path(npz_path).exists():
                return None

            # Lazy heavy imports: numpy only enters the process here, on demand.
            from app.ml.model_np import NpRiskModel  # noqa: WPS433 (intentional)

            model = NpRiskModel.load(npz_path, cfg)
            _CACHE[key] = model
            return model
    except ImportError:
        # numpy / model_np unavailable -> lite mode, engine uses rules only.
        return None
    except Exception:  # noqa: BLE001 - serving firewall: degrade, never propagate.
        return None


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #
def current_version(cfg: MLConfig = ml_config) -> Optional[str]:
    """Version string of the ACTIVE artifact, or ``None`` if none is active."""
    ptr = _read_pointer(cfg)
    if not ptr:
        return None
    version = ptr.get("version")
    return str(version) if version else None


def list_versions(cfg: MLConfig = ml_config) -> List[dict]:
    """All persisted versions (newest first) with metadata for inspection.

    Each entry: ``{version, saved_at, active, model_path, has_metrics}``. Built
    from the ``ML_DIR/versions/`` history; the currently-active version is
    flagged. Never raises — returns ``[]`` if nothing is recorded.
    """
    try:
        active = current_version(cfg)
        vroot = _versions_dir(cfg)
        out: List[dict] = []
        if vroot.is_dir():
            for vdir in vroot.iterdir():
                if not vdir.is_dir():
                    continue
                meta = _safe_read_json(vdir / _ARTIFACT_META) or {}
                version = str(meta.get("version") or vdir.name)
                npz = vdir / _ARTIFACT_NPZ
                out.append({
                    "version": version,
                    "saved_at": _safe_mtime_iso(npz if npz.exists() else vdir),
                    "active": version == active,
                    "model_path": str(npz),
                    "has_metrics": (vdir / _ARTIFACT_METRICS).exists(),
                })
        out.sort(key=lambda e: e.get("saved_at") or "", reverse=True)
        return out
    except Exception:  # noqa: BLE001 - introspection must never break callers.
        return []


def activate_version(version: str, cfg: MLConfig = ml_config) -> Optional[str]:
    """Re-point ACTIVE at a previously-saved ``version`` (rollback helper).

    Copies that version's weights/meta back onto the default ``cfg`` paths and
    flips ``ACTIVE.json``. Returns the active ``.npz`` path, or ``None`` if the
    version is unknown / its files are missing. Never raises.
    """
    try:
        with _LOCK:
            vdir = _versions_dir(cfg) / _safe_name(version)
            v_npz = vdir / _ARTIFACT_NPZ
            v_meta = vdir / _ARTIFACT_META
            if not v_npz.exists():
                return None

            active_npz = Path(cfg.model_path)
            active_meta = Path(cfg.meta_path)
            active_npz.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(v_npz, active_npz)
            if v_meta.exists():
                shutil.copyfile(v_meta, active_meta)

            resolved = str((_safe_read_json(v_meta) or {}).get("version") or version)
            _write_pointer(cfg, {
                "version": resolved,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "model_path": str(active_npz),
                "meta_path": str(active_meta),
                "version_dir": str(vdir),
                "metrics": _compact_metrics(_safe_read_json(vdir / _ARTIFACT_METRICS)),
            })
            _CACHE.clear()
            return str(active_npz)
    except Exception:  # noqa: BLE001
        return None


def clear_cache() -> None:
    """Drop the in-process load cache (test / hot-reload helper)."""
    with _LOCK:
        _CACHE.clear()


# --------------------------------------------------------------------------- #
# Small, dependency-free utilities (no numpy here — keeps the module lite)
# --------------------------------------------------------------------------- #
def _safe_name(version: str) -> str:
    """Filesystem-safe directory name for a version string."""
    safe = "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(version))
    return safe or "unversioned"


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _safe_mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _compact_metrics(metrics: Optional[dict]) -> dict:
    """A small, JSON-safe summary of headline metrics for the pointer file.

    Keeps only flat scalar (numeric / string / bool) entries so ``ACTIVE.json``
    stays tiny and human-readable; the full metrics live in the version dir.
    """
    if not isinstance(metrics, dict):
        return {}
    out: dict = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[str(k)] = v
    return out
