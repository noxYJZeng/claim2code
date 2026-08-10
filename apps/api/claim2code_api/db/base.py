"""SQLAlchemy declarative base used by all ORM models."""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base. All ORM models inherit from this."""
