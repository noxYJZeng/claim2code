"""``git clone`` wrapper for Phase 3 Repository Intelligence.

Public entry point: :func:`clone_repository`.

Responsibilities:

  1. Validate the URL — only ``https://github.com/<owner>/<repo>.git``
     in v1. Anything else raises :class:`RepoInvalidUrlError`.

  2. Optional HEAD pre-flight — when ``Settings.github_token`` is set
     we send one authenticated HEAD request to GitHub's API to learn
     the repo's reported size. If that exceeds
     ``Settings.max_repo_mb_megs``, we raise
     :class:`RepoTooLargeError` and don't even attempt the clone.

  3. Run ``git clone --depth 1 <url> <target>``. On non-zero exit,
     raise :class:`RepoCloneError` with the captured stderr.

  4. Return the :class:`CloneResult` so the workflow layer can insert
     a :class:`Repository` row with the canonical owner / name / path.

The cloner is **synchronous** — Phase 3 has no async queue. The
timeout is enforced by the workflow layer (``workflow_timeout_repo``,
declared in Phase 1 polish) wrapping the whole orchestrator, not by
``subprocess.run`` directly.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from claim2code_api.app.config import _PROJECT_ROOT as _CONFIG_PROJECT_ROOT
from claim2code_api.app.config import get_settings
from claim2code_api.repos.exceptions import (
    RepoCloneError,
    RepoInvalidUrlError,
    RepoTooLargeError,
)

# ============================================================
# URL validation
# ============================================================

# Only GitHub in v1. GitLab / Bitbucket / self-hosted are deferred
# (see ``phase-3-design.md`` §3). Both ``https://github.com/foo/bar``
# and ``https://github.com/foo/bar.git`` are accepted.
_GITHUB_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9._-]+)/(?P<name>[A-Za-z0-9._-]+?)(?:\.git)?/?$"
)


@dataclass(frozen=True)
class ParsedRepoUrl:
    """The structured result of :func:`parse_github_url`."""

    owner: str
    name: str
    url: str  # canonical ``https://github.com/<owner>/<repo>.git``


def parse_github_url(url: str) -> ParsedRepoUrl:
    """Validate a GitHub URL and return its owner / name.

    Raises :class:`RepoInvalidUrlError` for any non-GitHub URL or a URL
    whose owner / name segments don't match the regex (no spaces, no
    double-dots, etc.).
    """
    if not url:
        raise RepoInvalidUrlError("empty URL")
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise RepoInvalidUrlError(f"unsupported scheme {parsed.scheme!r}")
    if parsed.netloc.lower() != "github.com":
        raise RepoInvalidUrlError(f"only GitHub is supported in v1 (got host {parsed.netloc!r})")
    match = _GITHUB_URL_RE.match(url)
    if match is None:
        raise RepoInvalidUrlError(
            f"URL does not match 'https://github.com/<owner>/<repo>' shape: {url!r}"
        )
    owner = match.group("owner")
    name = match.group("name")
    if owner in {"", "."} or name in {"", "."}:
        raise RepoInvalidUrlError(f"empty owner / name in {url!r}")
    canonical = f"https://github.com/{owner}/{name}.git"
    return ParsedRepoUrl(owner=owner, name=name, url=canonical)


# ============================================================
# Path helpers
# ============================================================

# ``data/repos/`` is provisioned by ``scripts/dev-up.sh:18`` with a
# ``.gitkeep``. We resolve relative to the project root found by
# :mod:`claim2code_api.app.config` so the same code works in dev, CI,
# and Docker.
_PROJECT_ROOT = _CONFIG_PROJECT_ROOT


def data_repos_root() -> Path:
    """Return the absolute path to ``<project_root>/data/repos/``.

    The directory is created if missing (defensive — should already
    exist via ``scripts/dev-up.sh``).
    """
    root = _PROJECT_ROOT / "data" / "repos"
    root.mkdir(parents=True, exist_ok=True)
    return root


def clone_dir_for(owner: str, name: str) -> Path:
    """Return the absolute target directory for a clone.

    Owner / repo join via ``__`` (double underscore) to avoid path
    ambiguity when an owner or repo name itself contains an underscore.
    """
    return data_repos_root() / f"{owner}__{name}"


# ============================================================
# GitHub HEAD pre-flight (size guard)
# ============================================================


@dataclass(frozen=True)
class SizeEstimate:
    """Result of a GitHub API HEAD probe.

    ``reported_mb`` is ``None`` when the API didn't include a usable
    ``Content-Length`` or ``size`` hint — in that case the size guard
    is skipped and we rely on the post-clone ``n_files`` check.
    """

    reported_mb: float | None
    skipped_reason: str | None = None


def estimate_size(url: str) -> SizeEstimate:
    """HEAD-probe GitHub's API for the repo's reported size.

    Uses ``Settings.github_token`` if set (raises the rate limit from
    60/h to 5000/h). Best-effort — returns ``SizeEstimate(None,
    skipped_reason=...)`` if the API is unreachable or returns no size
    hint; the caller is expected to fall through to the clone.
    """
    settings = get_settings()
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    api_url = url.replace("https://github.com/", "https://api.github.com/repos/")
    api_url = api_url.removesuffix(".git")
    try:
        with httpx.Client(proxy=None, timeout=5.0) as client:
            r = client.get(api_url, headers=headers)
        if r.status_code != 200:
            return SizeEstimate(
                reported_mb=None,
                skipped_reason=f"api status {r.status_code}",
            )
        data = r.json()
        kb = data.get("size")  # GitHub returns size in KB
        if not isinstance(kb, (int, float)) or kb <= 0:
            return SizeEstimate(reported_mb=None, skipped_reason="no size field")
        reported_mb = float(kb) / 1024.0
        return SizeEstimate(reported_mb=reported_mb)
    except httpx.HTTPError as exc:
        return SizeEstimate(reported_mb=None, skipped_reason=str(exc))


def check_size_limit(estimate: SizeEstimate) -> None:
    """Raise :class:`RepoTooLargeError` if the estimate exceeds the cap.

    A ``None`` estimate is treated as "skip" — Phase 3 trades accuracy
    for availability; if GitHub rate-limits us, we still try the clone
    and rely on the post-clone ``n_files`` guard.
    """
    if estimate.reported_mb is None:
        return
    cap = get_settings().max_repo_mb_megs
    if estimate.reported_mb > cap:
        raise RepoTooLargeError(f"repo size {estimate.reported_mb:.1f} MB exceeds cap {cap} MB")


# ============================================================
# git clone
# ============================================================


@dataclass(frozen=True)
class CloneResult:
    """Outcome of a successful clone."""

    owner: str
    name: str
    url: str
    clone_path: Path
    default_branch: str


def clone_repository(
    url: str,
    *,
    github_token: str = "",
    size_estimate_timeout_seconds: float = 5.0,
) -> CloneResult:
    """Clone ``url`` into ``data/repos/<owner>__<repo>/``.

    Args:
        url: A GitHub URL (``https://github.com/<owner>/<repo>`` or
            the ``.git`` form).
        github_token: Optional GitHub PAT for the HEAD pre-flight.
            If empty, the pre-flight is anonymous (60 req/h cap).
        size_estimate_timeout_seconds: HTTP timeout on the pre-flight
            HEAD request.

    Returns:
        A :class:`CloneResult` describing the on-disk clone.

    Raises:
        RepoInvalidUrlError: URL is malformed or non-GitHub.
        RepoTooLargeError: GitHub's reported size exceeds the cap.
        RepoCloneError: ``git clone`` returned non-zero or git is not
            installed on the host.
    """
    parsed = parse_github_url(url)
    estimate = estimate_size(parsed.url)
    check_size_limit(estimate)

    target = clone_dir_for(parsed.owner, parsed.name)
    if target.exists() and (target / ".git").exists():
        raise RepoCloneError(f"clone already exists at {target}; DELETE /repositories first")
    # Wipe any stale partial clone from a previous failed attempt.
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    git = shutil.which("git")
    if git is None:
        raise RepoCloneError("git binary not found on PATH; install git in the runtime image")

    try:
        proc = subprocess.run(
            [
                git,
                "clone",
                "--depth",
                "1",
                "--quiet",
                parsed.url,
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RepoCloneError(f"could not invoke git: {exc}") from exc
    if proc.returncode != 0:
        raise RepoCloneError(
            f"git clone failed (exit {proc.returncode}): {proc.stderr.strip()[:500]}"
        )

    # Read the default branch out of HEAD. Shallow clones always
    # reference the upstream default via ``refs/remotes/origin/HEAD``
    # after the fetch; fall back to ``main`` if we can't resolve it.
    default_branch = _read_default_branch(target)
    return CloneResult(
        owner=parsed.owner,
        name=parsed.name,
        url=parsed.url,
        clone_path=target,
        default_branch=default_branch,
    )


def _read_default_branch(clone_path: Path) -> str:
    """Read the upstream default branch from ``.git/HEAD`` after clone.

    Returns ``"main"`` if the file is missing or malformed — never
    raises, because the default branch is purely advisory (the user
    can override by URL hash).
    """
    head_file = clone_path / ".git" / "HEAD"
    if not head_file.is_file():
        return "main"
    text = head_file.read_text(errors="replace").strip()
    # Refs look like ``ref: refs/remotes/origin/main``.
    if "/" not in text:
        return "main"
    return text.rsplit("/", 1)[-1] or "main"


def remove_clone(clone_path: Path) -> None:
    """Best-effort removal of a clone directory.

    Used by ``DELETE /repositories/{id}``. Errors are swallowed so
    the HTTP path stays idempotent.
    """
    if clone_path.exists():
        shutil.rmtree(clone_path, ignore_errors=True)
