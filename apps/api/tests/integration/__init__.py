"""Integration tests — run with ``pytest -m integration`` from ``apps/api``.

These tests require a real Postgres+pgvector instance (the GitHub Actions
``backend-test`` job spins one up via the ``services`` block). They are
skipped by default so contributors without Docker can still run the unit
suite.
"""
