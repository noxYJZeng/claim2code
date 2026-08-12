"""Adversarial tests for the Phase 8 safety gate.

These tests are opt-in via the ``safety`` pytest marker. Run via:

    cd apps/api && pytest -m safety -v

The default suite (``pytest``) excludes this directory via
``addopts = '-m \"not integration and not safety\"'`` so adversarial
fixtures don't bloat the dev loop.
"""