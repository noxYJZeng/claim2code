"""Alembic environment.

Reads ``DATABASE_URL`` from :mod:`claim2code_api.app.config` and uses
:data:`claim2code_api.db.base.Base.metadata` for autogenerate support.
"""
from __future__ import annotations

# Make ``claim2code_api`` importable when alembic is invoked from any cwd.
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_APPS_API = os.path.abspath(os.path.join(_HERE, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APPS_API, ".."))
for p in (_APPS_API, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from claim2code_api.app.config import get_settings  # noqa: E402
from claim2code_api.db import models  # noqa: F401, E402  (register models)
from claim2code_api.db.base import Base  # noqa: E402

config = context.config

# Inject the runtime DATABASE_URL so `alembic upgrade head` works without
# requiring the operator to export it manually.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
