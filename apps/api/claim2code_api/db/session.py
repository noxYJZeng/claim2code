"""SQLAlchemy engine, session factory, and DB health probe.

The engine is **lazy** — it is constructed on first use so that the API
can boot even when the database is briefly unreachable (e.g. during a
fresh ``docker compose up``). :func:`check_db_ready` runs ``SELECT 1``
and reports a structured success/failure for ``/health/ready``.
"""
from __future__ import annotations

from threading import Lock

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from claim2code_api.app.config import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None
_engine_lock = Lock()


def get_engine() -> Engine:
    """Return the lazily-initialised process-wide engine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                settings = get_settings()
                _engine = create_engine(
                    settings.database_url,
                    pool_pre_ping=True,
                    connect_args={"connect_timeout": int(settings.db_health_timeout_seconds)},
                    future=True,
                )
    return _engine


def get_session_local() -> sessionmaker:
    """Return the lazily-initialised session factory bound to :func:`get_engine`."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            future=True,
        )
    return _SessionLocal


def check_db_ready() -> tuple[bool, str | None]:
    """Run ``SELECT 1`` and return ``(ok, error_message)``.

    ``error_message`` is ``None`` on success, otherwise a short string
    suitable for returning in the ``/health/ready`` JSON body. We truncate
    to keep the payload bounded.
    """
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        return False, f"{exc.__class__.__name__}: {str(exc)[:200]}"
    except Exception as exc:  # pragma: no cover — defensive
        return False, f"unexpected: {exc.__class__.__name__}: {str(exc)[:200]}"
    return True, None


def reset_for_tests() -> None:
    """Drop the cached engine / session factory.

    Used by tests to pick up a different ``DATABASE_URL``. Production code
    should never call this.
    """
    global _engine, _SessionLocal
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _SessionLocal = None
