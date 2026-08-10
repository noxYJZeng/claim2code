"""Unit tests for :mod:`claim2code_api.papers.storage`.

These tests use :func:`monkeypatch.setenv` to redirect
``upload_dir()`` to a temporary directory so they don't touch the
real ``data/uploads`` tree. Every test is hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claim2code_api.papers import storage


@pytest.fixture
def tmp_uploads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``upload_dir`` at a fresh tmp dir for the test."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    # The upload_dir() function resolves the project root from
    # ``__file__`` and appends ``data/uploads``. We monkeypatch the
    # final computation by patching the parents calculation.
    monkeypatch.setattr(storage, "upload_dir", lambda: uploads)
    return uploads


def test_sha256_bytes_is_deterministic() -> None:
    a = storage.sha256_bytes(b"hello world")
    b = storage.sha256_bytes(b"hello world")
    assert a == b
    assert len(a) == 64


def test_sha256_bytes_distinguishes_content() -> None:
    assert storage.sha256_bytes(b"a") != storage.sha256_bytes(b"b")


def test_sha256_file_matches_sha256_bytes(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"same content here")
    assert storage.sha256_file(p) == storage.sha256_bytes(p.read_bytes())


def test_write_pdf_creates_file(tmp_uploads: Path) -> None:
    sha = storage.sha256_bytes(b"%PDF-1.4 fake")
    target = storage.write_pdf(sha, b"%PDF-1.4 fake")
    assert target.exists()
    assert target == tmp_uploads / f"{sha}.pdf"
    assert target.read_bytes() == b"%PDF-1.4 fake"


def test_write_pdf_is_atomic(tmp_uploads: Path) -> None:
    """A second write of the same sha256 must fail (no clobber)."""
    sha = storage.sha256_bytes(b"only once")
    storage.write_pdf(sha, b"only once")
    with pytest.raises(FileExistsError):
        storage.write_pdf(sha, b"only once again")


def test_path_for_returns_expected_location(tmp_uploads: Path) -> None:
    p = storage.path_for("deadbeef" * 8)
    assert p == tmp_uploads / ("deadbeef" * 8 + ".pdf")


def test_read_pdf_round_trips(tmp_uploads: Path) -> None:
    sha = storage.sha256_bytes(b"round-trip content")
    storage.write_pdf(sha, b"round-trip content")
    assert storage.read_pdf(sha) == b"round-trip content"


def test_read_pdf_raises_for_missing(tmp_uploads: Path) -> None:
    with pytest.raises(FileNotFoundError):
        storage.read_pdf("0" * 64)
