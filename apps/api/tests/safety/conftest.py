"""Auto-mark every test under tests/safety/ with the ``safety`` marker.

Phase 8 ships adversarial fixtures in ``tests/safety/`` (redactor,
detector, gate, end-to-end LLMClient integration). These tests are
opt-in via the ``safety`` marker; ``pyproject.toml``'s ``addopts``
skips them from the default suite. This conftest ensures every test
in this directory is marked without requiring an explicit decorator
on each function.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-mark tests under tests/safety/ as ``safety``."""
    for item in items:
        if "tests/safety" in str(item.fspath):
            item.add_marker(pytest.mark.safety)