"""Unit tests for :mod:`claim2code_api.repos.cloner`.

The cloner wraps ``git`` and ``httpx``. Both are mocked here so the
tests stay CI-safe (no network, no real git binary required).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from claim2code_api.repos.cloner import (
    CloneResult,
    ParsedRepoUrl,
    check_size_limit,
    clone_dir_for,
    clone_repository,
    data_repos_root,
    estimate_size,
    parse_github_url,
    remove_clone,
)
from claim2code_api.repos.exceptions import (
    RepoCloneError,
    RepoInvalidUrlError,
    RepoTooLargeError,
)

# ============================================================
# parse_github_url
# ============================================================


def test_parse_github_url_canonical() -> None:
    p = parse_github_url("https://github.com/owner/repo")
    assert p == ParsedRepoUrl(owner="owner", name="repo", url="https://github.com/owner/repo.git")


def test_parse_github_url_with_git_suffix() -> None:
    p = parse_github_url("https://github.com/owner/repo.git")
    assert p.owner == "owner"
    assert p.name == "repo"


def test_parse_github_url_with_trailing_slash() -> None:
    p = parse_github_url("https://github.com/owner/repo/")
    assert p.owner == "owner"
    assert p.name == "repo"


def test_parse_github_url_strips_owner_dot_files() -> None:
    p = parse_github_url("https://github.com/o.w.n.er/repo.git")
    assert p.owner == "o.w.n.er"


def test_parse_github_url_rejects_gitlab() -> None:
    with pytest.raises(RepoInvalidUrlError, match="only GitHub"):
        parse_github_url("https://gitlab.com/owner/repo.git")


def test_parse_github_url_rejects_ssh() -> None:
    with pytest.raises(RepoInvalidUrlError):
        parse_github_url("git@github.com:owner/repo.git")


def test_parse_github_url_rejects_empty() -> None:
    with pytest.raises(RepoInvalidUrlError):
        parse_github_url("")


def test_parse_github_url_rejects_malformed() -> None:
    with pytest.raises(RepoInvalidUrlError):
        parse_github_url("https://github.com/only-owner")


# ============================================================
# Path helpers
# ============================================================


def test_data_repos_root_creates_directory(tmp_path: Path, monkeypatch) -> None:
    """``data_repos_root`` should mkdir -p (idempotent)."""
    # Patch the module-level _PROJECT_ROOT to a tmp path.
    import claim2code_api.repos.cloner as cloner_mod

    monkeypatch.setattr(cloner_mod, "_PROJECT_ROOT", tmp_path)
    root = data_repos_root()
    assert root == tmp_path / "data" / "repos"
    assert root.is_dir()


def test_clone_dir_for_uses_double_underscore(tmp_path: Path, monkeypatch) -> None:
    import claim2code_api.repos.cloner as cloner_mod

    monkeypatch.setattr(cloner_mod, "_PROJECT_ROOT", tmp_path)
    target = clone_dir_for("owner_with_underscore", "repo")
    assert target == tmp_path / "data" / "repos" / "owner_with_underscore__repo"


# ============================================================
# Size guard
# ============================================================


def test_check_size_limit_passes_under_cap() -> None:
    check_size_limit(
        __import__("claim2code_api.repos.cloner", fromlist=["SizeEstimate"]).SizeEstimate(
            reported_mb=10.0
        )
    )


def test_check_size_limit_skips_when_estimate_is_none() -> None:
    from claim2code_api.repos.cloner import SizeEstimate

    # Should not raise even with a tiny cap.
    with patch("claim2code_api.repos.cloner.get_settings") as gs:
        gs.return_value.max_repo_mb_megs = 1
        check_size_limit(SizeEstimate(reported_mb=None, skipped_reason="api 403"))


def test_check_size_limit_raises_over_cap() -> None:
    from claim2code_api.repos.cloner import SizeEstimate

    with patch("claim2code_api.repos.cloner.get_settings") as gs:
        gs.return_value.max_repo_mb_megs = 5
        with pytest.raises(RepoTooLargeError):
            check_size_limit(SizeEstimate(reported_mb=10.0))


# ============================================================
# estimate_size
# ============================================================


def test_estimate_size_uses_api_size_field() -> None:
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"size": 1024}  # 1 MB
    fake_client = MagicMock()
    fake_client.__enter__.return_value.get.return_value = fake_response
    with patch.object(httpx, "Client", return_value=fake_client):
        est = estimate_size("https://github.com/foo/bar")
    assert est.reported_mb == pytest.approx(1.0)


def test_estimate_size_skips_on_non_200() -> None:
    fake_response = MagicMock()
    fake_response.status_code = 403
    fake_client = MagicMock()
    fake_client.__enter__.return_value.get.return_value = fake_response
    with patch.object(httpx, "Client", return_value=fake_client):
        est = estimate_size("https://github.com/foo/bar")
    assert est.reported_mb is None
    assert est.skipped_reason is not None


def test_estimate_size_skips_on_http_error() -> None:
    fake_client = MagicMock()
    fake_client.__enter__.return_value.get.side_effect = httpx.ConnectError("boom")
    with patch.object(httpx, "Client", return_value=fake_client):
        est = estimate_size("https://github.com/foo/bar")
    assert est.reported_mb is None
    assert "boom" in (est.skipped_reason or "")


# ============================================================
# clone_repository (with mocked subprocess)
# ============================================================


def test_clone_repository_success(tmp_path: Path, monkeypatch) -> None:
    """Happy path: parse URL, estimate size, git clone, return result."""
    import claim2code_api.repos.cloner as cloner_mod

    monkeypatch.setattr(cloner_mod, "_PROJECT_ROOT", tmp_path)

    # Pre-create the parent and ``.git/HEAD`` so the read-default-branch
    # helper finds what it expects after the mocked subprocess.
    def fake_run(*args, **kwargs):
        target = Path(args[0][-1])
        (target / ".git").mkdir(parents=True)
        (target / ".git" / "HEAD").write_text("ref: refs/remotes/origin/main\n")
        (target / "README.md").write_text("# hello\n")
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = ""
        return proc

    with patch.object(subprocess, "run", side_effect=fake_run):
        with patch("claim2code_api.repos.cloner.estimate_size") as mock_est:
            mock_est.return_value = cloner_mod.SizeEstimate(reported_mb=0.5)
            result = clone_repository("https://github.com/owner/repo.git")

    assert isinstance(result, CloneResult)
    assert result.owner == "owner"
    assert result.name == "repo"
    assert result.url == "https://github.com/owner/repo.git"
    assert result.default_branch == "main"
    assert result.clone_path.is_dir()


def test_clone_repository_rejects_non_github(tmp_path: Path, monkeypatch) -> None:
    import claim2code_api.repos.cloner as cloner_mod

    monkeypatch.setattr(cloner_mod, "_PROJECT_ROOT", tmp_path)
    with pytest.raises(RepoInvalidUrlError):
        clone_repository("https://gitlab.com/owner/repo.git")


def test_clone_repository_rejects_too_large(tmp_path: Path, monkeypatch) -> None:
    import claim2code_api.repos.cloner as cloner_mod

    monkeypatch.setattr(cloner_mod, "_PROJECT_ROOT", tmp_path)
    with patch("claim2code_api.repos.cloner.estimate_size") as mock_est:
        mock_est.return_value = cloner_mod.SizeEstimate(reported_mb=500.0)
        with patch("claim2code_api.repos.cloner.get_settings") as gs:
            gs.return_value.max_repo_mb_megs = 200
            with pytest.raises(RepoTooLargeError):
                clone_repository("https://github.com/owner/repo.git")


def test_clone_repository_raises_on_git_failure(tmp_path: Path, monkeypatch) -> None:
    import claim2code_api.repos.cloner as cloner_mod

    monkeypatch.setattr(cloner_mod, "_PROJECT_ROOT", tmp_path)
    proc = MagicMock()
    proc.returncode = 128
    proc.stderr = "fatal: repository not found"
    with patch.object(subprocess, "run", return_value=proc):
        with patch("claim2code_api.repos.cloner.estimate_size") as mock_est:
            mock_est.return_value = cloner_mod.SizeEstimate(reported_mb=0.1)
            with pytest.raises(RepoCloneError, match="repository not found"):
                clone_repository("https://github.com/owner/repo.git")


def test_clone_repository_raises_when_git_missing(tmp_path: Path, monkeypatch) -> None:
    import claim2code_api.repos.cloner as cloner_mod

    monkeypatch.setattr(cloner_mod, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloner_mod.shutil, "which", lambda _: None)
    with patch("claim2code_api.repos.cloner.estimate_size") as mock_est:
        mock_est.return_value = cloner_mod.SizeEstimate(reported_mb=0.1)
        with pytest.raises(RepoCloneError, match="git binary not found"):
            clone_repository("https://github.com/owner/repo.git")


def test_clone_repository_refuses_existing_clone(tmp_path: Path, monkeypatch) -> None:
    import claim2code_api.repos.cloner as cloner_mod

    monkeypatch.setattr(cloner_mod, "_PROJECT_ROOT", tmp_path)
    target = tmp_path / "data" / "repos" / "owner__repo"
    target.mkdir(parents=True)
    (target / ".git").mkdir()
    with patch("claim2code_api.repos.cloner.estimate_size") as mock_est:
        mock_est.return_value = cloner_mod.SizeEstimate(reported_mb=0.1)
        with pytest.raises(RepoCloneError, match="already exists"):
            clone_repository("https://github.com/owner/repo.git")


# ============================================================
# remove_clone
# ============================================================


def test_remove_clone_is_idempotent(tmp_path: Path) -> None:
    """Calling remove_clone on a missing path is fine."""
    remove_clone(tmp_path / "nope")
    # No exception; the path still doesn't exist.
    assert not (tmp_path / "nope").exists()


def test_remove_clone_removes_existing(tmp_path: Path) -> None:
    d = tmp_path / "clone"
    d.mkdir()
    (d / "file.txt").write_text("x")
    remove_clone(d)
    assert not d.exists()
