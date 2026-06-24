"""FastAPI application factory for the AI Media Watch engine.

Wires CORS (for the Vite frontend), the ``/api`` router, a startup hook that
ensures the SQLite schema exists, and a small info root. Run directly with
``python -m app.main`` or via ``uvicorn app.main:app``.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app as app_pkg
from app.api.routes import router
from app.config import settings
from app.models import registry
from app.store import db


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title="AI Media Watch Engine",
        version=app_pkg.__version__,
        description=(
            "Многоуровневый мультимодальный движок оценки риска мошенничества "
            "для коротких видео из социальных сетей."
        ),
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)

    @application.on_event("startup")
    def _on_startup() -> None:
        """Ensure the cases table exists before serving requests."""
        try:
            db.init_db()
        except Exception:  # noqa: BLE001 - never block startup on the DB
            pass

    @application.get("/")
    def root() -> dict:
        """Small machine-readable info document about the running engine."""
        return {
            "name": "AI Media Watch Engine",
            "version": app_pkg.__version__,
            "engineMode": registry.engine_mode(),
            "capabilities": registry.capabilities(),
            "docs": "/docs",
            "api": "/api",
        }

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
