"""Disk storage for uploaded PDFs.

Phase 2 stores PDF bytes on disk under ``data/uploads/<sha256>.pdf``:

  - The sha256 doubles as the filename and the dedup key — uploading
    the same paper twice writes only one file.
  - We never overwrite an existing file; the second writer races on
    ``O_EXCL`` (atomic on POSIX) and the loser sees the winner's
    bytes via the read-back helpers below.
  - ``data/uploads/`` is created by ``scripts/dev-up.sh`` so the
    directory exists before the API process boots.

Why disk (not DB BLOB / S3)?
---------------------------
Phase 2 PDFs are 1-50 MB. ``bytea`` is capped at 1 GB and is slow to
back up; S3 is out of scope for the local-first dev story. A flat
directory under the project root is the cheapest option that lets
``du`` / ``rsync`` work and lets Phase 5 swap in S3 behind this same
interface when load demands it.

The functions in this module are the **only** path the rest of the
codebase uses to touch ``data/uploads/`` — never ``open()`` the files
directly. That keeps the on-disk-layout decision swappable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from claim2code_api.app.config import get_settings


def upload_dir() -> Path:
    """Return the configured upload directory, creating it if missing.

    The path is rooted at the project root (one level above
    ``apps/api``) so it survives a backend rebuild and matches the
    ``data/`` volume mounted by ``docker-compose.yml``.

    ``get_settings()`` is invoked so any future override of the
    upload path via Settings lands here automatically; for v1 the
    path is fixed at ``<project>/data/uploads``.
    """
    get_settings()  # touch the Settings cache so future overrides land here
    # apps/api/claim2code_api/papers/storage.py → project root is 4 levels up.
    project_root = Path(__file__).resolve().parents[4]
    path = project_root / "data" / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_bytes(data: bytes) -> str:
    """Compute sha256 of an in-memory byte string (upload-then-parse path)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Compute sha256 of a file by streaming (low-memory path)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_pdf(sha256: str, data: bytes) -> Path:
    """Write PDF bytes to ``data/uploads/<sha256>.pdf`` atomically.

    :raises FileExistsError: if another writer already produced this
        sha256 (use :func:`path_for` to read the winner's file).
    """
    target = upload_dir() / f"{sha256}.pdf"
    # ``xb`` = exclusive create — fails if the file already exists.
    with target.open("xb") as f:
        f.write(data)
    return target


def path_for(sha256: str) -> Path:
    """Return the on-disk path for ``sha256`` (does not check existence)."""
    return upload_dir() / f"{sha256}.pdf"


def read_pdf(sha256: str) -> bytes:
    """Read the stored PDF bytes for ``sha256``.

    :raises FileNotFoundError: if no file has ever been stored under
        this sha256 (i.e. the row in the ``papers`` table is
        corrupt).
    """
    target = path_for(sha256)
    if not target.exists():
        raise FileNotFoundError(f"Stored PDF not found: {target}")
    return target.read_bytes()
