"""FastAPI dependency providers."""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from claim2code_api.db.session import get_session_local


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and close it after the request."""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
