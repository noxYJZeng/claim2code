"""Pinned benchmark preparation must preserve checkouts and reject incomplete indexes."""
# Bootstrap the repository-root evaluation package for isolated API test runs.
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from claim2code_api.app.config import get_settings
from claim2code_api.db.models import Base, CodeEmbedding, CodeSymbol, Repository
from claim2code_api.indexing.indexer import embed_and_persist_symbols
from evaluation import prepare_benchmark as prepare


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    path = tmp_path / "clone"
    path.mkdir()
    (path / "model.py").write_text("import torch\n\ndef scale(x):\n    return x * 2\n")
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", "https://github.com/example/pinned.git"], check=True)
    subprocess.run(["git", "-C", str(path), "add", "model.py"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m", "fixture"], check=True)
    return path


@pytest.fixture
def annotation(clone: Path) -> dict:
    return {
        "pair_id": "fixture", "paper": {},
        "repo": {"url": "https://github.com/example/pinned", "commit": prepare._git(clone, "rev-parse", "HEAD"),
                 "framework": "pytorch", "framework_evidence": [{"file_path": "model.py", "quote": "import torch"}]},
        "claims": [{"claim_id": "fixture_c1", "text": "Scale doubles its input.",
                    "relevant_symbols": [{"file_path": "model.py", "name": "scale", "kind": "function"}],
                    "expected_verification": "VERIFIED"}],
    }


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_existing_clone_is_preserved_when_dirty(clone: Path, annotation: dict) -> None:
    path = clone / "model.py"
    path.write_text(path.read_text() + "# user's edit\n")
    with pytest.raises(ValueError, match="local changes"):
        prepare.ensure_clone(clone, annotation["repo"]["url"], annotation["repo"]["commit"])
    assert path.read_text().endswith("# user's edit\n")


def test_existing_clone_rejects_wrong_pin_and_origin(clone: Path, annotation: dict) -> None:
    with pytest.raises(ValueError, match="pinned revision"):
        prepare.check_clone(clone, annotation["repo"]["url"], "f" * 40)
    with pytest.raises(ValueError, match="origin differs"):
        prepare.check_clone(clone, "https://github.com/example/other", annotation["repo"]["commit"])


def test_existing_clone_rejects_untracked_files(clone: Path, annotation: dict) -> None:
    (clone / "user-notes.txt").write_text("preserve this")
    with pytest.raises(ValueError, match="local changes"):
        prepare.check_clone(clone, annotation["repo"]["url"], annotation["repo"]["commit"])
    assert (clone / "user-notes.txt").read_text() == "preserve this"


def test_suite_rejects_unpinned_and_unknown_pairs(tmp_path: Path, annotation: dict) -> None:
    manifest = tmp_path / "suite.json"
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    manifest.write_text(json.dumps({"pair_ids": ["fixture"]}))
    annotation["repo"]["commit"] = "main"
    (annotations / "fixture.json").write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match="full immutable"):
        prepare.load_suite(manifest, annotations)
    manifest.write_text(json.dumps({"pair_ids": ["unknown"]}))
    with pytest.raises(ValueError, match="Unknown"):
        prepare.load_suite(manifest, annotations)


def test_parse_only_indexes_real_source_without_model(
    clone: Path, annotation: dict, db: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "ensure_clone", lambda *_: clone)
    monkeypatch.setattr(prepare, "embed_and_persist_symbols", lambda *_: pytest.fail("parse-only called a model"))
    result = prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)
    assert result["n_implementation_symbols"] == 1
    assert result["n_resolved_annotation_symbols"] == 1
    assert result["n_embedded_implementation_symbols"] == 0
    assert result["n_embeddings"] == 0
    assert db.query(CodeSymbol).count() == 2
    second = prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)
    assert second["reused"] is True
    assert second["repository_id"] == result["repository_id"]
    assert db.query(CodeSymbol).count() == 2


def test_preparation_can_upgrade_parse_only_and_reuse_complete_vectors(
    clone: Path, annotation: dict, db: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "ensure_clone", lambda *_: clone)
    prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)

    class Embeddings:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * (get_settings().openai_embed_dim - 1) for _ in texts]

    calls = []

    def embed(session: Session, repository_id) -> int:
        calls.append(repository_id)
        return embed_and_persist_symbols(session, repository_id, embedding_client=Embeddings())

    monkeypatch.setattr(prepare, "embed_and_persist_symbols", embed)
    result = prepare.prepare_pair(db, annotation, clone.parent, parse_only=False)
    assert result["n_embedded_implementation_symbols"] == 1
    assert result["n_embeddings"] == 2
    assert result["embedding_model"] == get_settings().openai_model_embed
    prepare.prepare_pair(db, annotation, clone.parent, parse_only=False)
    assert len(calls) == 1


def test_preparation_rejects_missing_gold_symbol_and_wrong_framework_quote(
    clone: Path, annotation: dict, db: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "ensure_clone", lambda *_: clone)
    annotation["claims"][0]["relevant_symbols"][0]["name"] = "missing"
    with pytest.raises(ValueError, match="annotated symbol was not indexed"):
        prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)
    annotation["claims"][0]["relevant_symbols"][0]["name"] = "scale"
    annotation["repo"]["framework_evidence"][0]["quote"] = "import nonexistent"
    with pytest.raises(ValueError, match="framework evidence"):
        prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)


def test_preparation_rejects_partial_vectors_without_mutating_index(
    clone: Path, annotation: dict, db: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "ensure_clone", lambda *_: clone)
    prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)
    repo = db.query(Repository).one()
    imported = db.query(CodeSymbol).filter_by(kind="import").one()
    db.add(CodeEmbedding(symbol_id=imported.symbol_id, repository_id=repo.repository_id,
                         analysis_id=repo.analysis_id, model_name=get_settings().openai_model_embed,
                         embedding=[1.0] + [0.0] * (get_settings().openai_embed_dim - 1)))
    db.commit()
    monkeypatch.setattr(prepare, "embed_and_persist_symbols", lambda *_: pytest.fail("overwrote partial vectors"))
    with pytest.raises(ValueError, match="partial/stale"):
        prepare.prepare_pair(db, annotation, clone.parent, parse_only=False)
    assert db.query(CodeEmbedding).count() == 1


def test_preparation_checks_method_parent_and_exact_annotation_source(
    clone: Path, annotation: dict, db: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "ensure_clone", lambda *_: clone)
    target = annotation["claims"][0]["relevant_symbols"][0]
    target["parent_name"] = "NotItsParent"
    with pytest.raises(ValueError, match="annotated symbol was not indexed"):
        prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)
    target.pop("parent_name")
    annotation["claims"][0]["source_evidence"] = [
        {"file_path": "model.py", "start_line": 1, "end_line": 1, "quote": "return x * 2"},
    ]
    with pytest.raises(ValueError, match="exact lines"):
        prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)
    annotation["claims"][0]["source_evidence"][0].update(start_line=4, end_line=4)
    assert prepare.prepare_pair(db, annotation, clone.parent, parse_only=True)["n_resolved_annotation_symbols"] == 1


def test_failed_embedding_is_an_error_not_successful_preparation(
    clone: Path, annotation: dict, db: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "ensure_clone", lambda *_: clone)
    monkeypatch.setattr(prepare, "embed_and_persist_symbols", lambda *_: 0)
    with pytest.raises(ValueError, match="embedding failed"):
        prepare.prepare_pair(db, annotation, clone.parent, parse_only=False)


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="requires migrated PostgreSQL")
def test_postgres_preparation_persists_complete_index_and_checks_vector_dimensions(
    clone: Path, annotation: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "ensure_clone", lambda *_: clone)

    class Embeddings:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * (get_settings().openai_embed_dim - 1) for _ in texts]

    monkeypatch.setattr(prepare, "embed_and_persist_symbols", lambda db, repo_id:
                        embed_and_persist_symbols(db, repo_id, embedding_client=Embeddings()))
    engine = create_engine(os.environ["DATABASE_URL"])
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                with Session(bind=connection, join_transaction_mode="create_savepoint") as db:
                    result = prepare.prepare_pair(db, annotation, clone.parent, parse_only=False)
                    assert result["n_embeddings"] == 2
                    assert result["n_embedded_implementation_symbols"] == 1
                    again = prepare.prepare_pair(db, annotation, clone.parent, parse_only=False)
                    assert again["repository_id"] == result["repository_id"]
                    assert again["reused"] is True
            finally:
                transaction.rollback()
    finally:
        engine.dispose()
