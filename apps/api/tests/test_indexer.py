"""Unit tests for :mod:`claim2code_api.indexing.indexer`.

Coverage:

  - ``embed_and_persist_symbols`` persists one CodeEmbedding per
    CodeSymbol with the correct symbol_id / repository_id /
    analysis_id / model_name.
  - The function records the active model on
    ``Repository.embedding_model``.
  - Empty repo (zero symbols) writes zero rows but still records
    the model.
  - On ``EmbeddingError`` the repo's ``error`` column is set; the
    parse-step ``index_status='done'`` is left untouched (the
    retriever gracefully returns ``[]`` until the user retries).
    and a structured entry is appended to ``analysis_states.errors``.
  - Source body is read from the clone path between
    ``start_line`` and ``end_line`` (1-indexed, inclusive).
  - Imports are filtered to ``kind='import'`` and projected.
  - Missing files / out-of-range lines fall back to empty source
    (the other fields still pin the symbol).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from claim2code_api.app.config import Settings
from claim2code_api.db.models import AnalysisState, CodeEmbedding, CodeSymbol, Repository
from claim2code_api.indexing.exceptions import EmbeddingError
from claim2code_api.indexing.indexer import embed_and_persist_symbols

# ============================================================
# Helpers
# ============================================================


def _make_vec(seed: int) -> list[float]:
    """Deterministic 4-dim vector placeholder."""
    return [float(seed + j) / 100.0 for j in range(4)]


def _fake_settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        openai_model_embed="text-embedding-3-small",
        openai_embed_dim=1536,
        embed_batch_size=128,
        embed_max_retries=3,
    )


def _fake_embedding_client(payload) -> Any:
    """A stub that returns canned vectors or raises.

    Returned as ``Any`` so the test sites can read ``stub.calls``
    (a duck-typed attribute the EmbeddingClient Protocol doesn't
    declare) without tripping mypy. The runtime protocol is
    unchanged: ``Stub.embed`` matches :meth:`EmbeddingClient.embed`.
    """

    class _Stub:
        def __init__(self, payload) -> None:
            self.payload = payload
            self.calls: list[list[str]] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            if isinstance(self.payload, Exception):
                raise self.payload
            return self.payload

    return _Stub(payload)


_MISSING = object()


def _build_session(
    repo: Repository | MagicMock,
    symbols: Sequence[CodeSymbol | SimpleNamespace],
    *,
    analysis_id=None,
    repo_lookup: object = _MISSING,
) -> MagicMock:
    """Build a MagicMock session with the methods the indexer uses.

    The session's ``query(CodeSymbol).filter(...).order_by(...).all()``
    returns the symbols list. ``db.get(Repository, id)`` returns
    ``repo_lookup`` (defaults to ``repo``). Caller passes None
    here to simulate a missing repository. ``db.query(AnalysisState)
    .filter(...).update(...)`` records the payload so tests can
    assert it.

    ``symbols`` is a ``Sequence`` (covariant) so test sites can
    pass ``list[SimpleNamespace]`` or ``list[CodeSymbol]``
    interchangeably without mypy invariance complaints.
    """
    session = MagicMock(name="session")
    session.commit = MagicMock()

    # db.get(Repository, repository_id) → repo_lookup (or repo by default)
    if repo_lookup is _MISSING:
        session.get.return_value = repo
    else:
        session.get.return_value = repo_lookup  # may be None to simulate missing

    # db.query(CodeSymbol).filter(...).order_by(...).all() → symbols
    sym_query = MagicMock()
    sym_query.filter.return_value.order_by.return_value.all.return_value = symbols

    # db.query(AnalysisState).filter(...).update(dict) → records payload
    analysis_query = MagicMock()

    def _capture_update(payload) -> None:
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if analysis_id is not None:
                    analysis.errors = value
                return

    analysis_query.filter.return_value.update.side_effect = _capture_update

    def _query_dispatch(model):
        if model is CodeSymbol:
            return sym_query
        if model is AnalysisState and analysis_id is not None:
            return analysis_query
        return MagicMock()

    session.query.side_effect = _query_dispatch

    # Track db.add()ed rows so tests can assert.
    session.add = MagicMock()

    # Local analysis state for the failure tests.
    if analysis_id is not None:
        analysis = SimpleNamespace(analysis_id=analysis_id, errors=[])
        session._analysis = analysis  # type: ignore[attr-defined]
    else:
        session._analysis = None  # type: ignore[attr-defined]

    return session


def _make_repository(
    *,
    repository_id=None,
    analysis_id=None,
    clone_path: str = "",
) -> MagicMock:
    repo = MagicMock(spec=Repository)
    repo.repository_id = repository_id or uuid4()
    repo.analysis_id = analysis_id or uuid4()
    repo.clone_path = clone_path
    repo.embedding_model = None
    repo.index_status = "pending"
    repo.error = None
    return repo


def _make_symbol(
    *,
    repository_id,
    file_path: str,
    name: str,
    kind: str = "function",
    language: str = "python",
    signature: str = "()",
    docstring: str | None = None,
    start_line: int = 1,
    end_line: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol_id=uuid4(),
        repository_id=repository_id,
        kind=kind,
        language=language,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        start_col=0,
        end_col=0,
        name=name,
        signature=signature,
        docstring=docstring,
        parent_symbol_id=None,
    )


@pytest.fixture
def tmp_clone(tmp_path: Path) -> Path:
    """A fake clone directory with a couple of source files."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "core.py").write_text(
        "import numpy as np\n"
        "import torch\n"
        "\n"
        "def add(a, b):\n"
        '    """Add two ints."""\n'
        "    return a + b\n"
    )
    (src / "math.py").write_text("def subtract(a, b):\n    return a - b\n")
    return tmp_path


# ============================================================
# Happy path
# ============================================================


def test_embed_persists_one_row_per_symbol(tmp_clone: Path) -> None:
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="add",
            signature="(a, b)",
            docstring="Add two ints.",
            start_line=4,
            end_line=6,
        ),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(1)])

    written = embed_and_persist_symbols(
        session, repo_id, embedding_client=client, settings=_fake_settings()
    )

    assert written == 1
    assert session.add.call_count == 1
    added_row = session.add.call_args_list[0].args[0]
    assert isinstance(added_row, CodeEmbedding)
    assert added_row.symbol_id == symbols[0].symbol_id
    assert added_row.repository_id == repo_id
    assert added_row.analysis_id == aid
    assert added_row.model_name == "text-embedding-3-small"
    assert added_row.embedding == _make_vec(1)
    # Repository is updated to record the model used.
    assert repo.embedding_model == "text-embedding-3-small"
    assert session.commit.called


def test_embed_writes_zero_rows_when_repo_has_no_symbols(tmp_clone: Path) -> None:
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    session = _build_session(repo, [], analysis_id=aid)
    client = _fake_embedding_client([])

    written = embed_and_persist_symbols(
        session, repo_id, embedding_client=client, settings=_fake_settings()
    )

    assert written == 0
    assert session.add.call_count == 0
    # The model is still recorded so the retriever knows the
    # index was attempted.
    assert repo.embedding_model == "text-embedding-3-small"


def test_embed_groups_dense_corpus_in_single_batch(tmp_clone: Path) -> None:
    """A repo with 3 symbols → 1 embed() call returning 3 vectors."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(repository_id=repo_id, file_path="src/core.py", name="add"),
        _make_symbol(repository_id=repo_id, file_path="src/core.py", name="sub"),
        _make_symbol(repository_id=repo_id, file_path="src/math.py", name="mul"),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(1), _make_vec(2), _make_vec(3)])

    written = embed_and_persist_symbols(
        session, repo_id, embedding_client=client, settings=_fake_settings()
    )

    assert written == 3
    assert len(client.calls) == 1
    assert len(client.calls[0]) == 3


def test_embed_records_model_name_on_repository(tmp_clone: Path) -> None:
    """The function writes ``embedding_model`` from Settings."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    session = _build_session(repo, [], analysis_id=aid)
    client = _fake_embedding_client([])

    settings = Settings(
        openai_api_key="sk-test",
        openai_model_embed="text-embedding-3-large",
        openai_embed_dim=3072,
        embed_batch_size=128,
        embed_max_retries=3,
    )
    embed_and_persist_symbols(session, repo_id, embedding_client=client, settings=settings)

    assert repo.embedding_model == "text-embedding-3-large"


# ============================================================
# Source body reading
# ============================================================


def test_embed_reads_source_body_between_lines(tmp_clone: Path) -> None:
    """The Source: block in the embedding text reflects the line range."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="add",
            signature="(a, b)",
            docstring="Add two ints.",
            start_line=4,
            end_line=6,
        ),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(0)])

    embed_and_persist_symbols(session, repo_id, embedding_client=client, settings=_fake_settings())

    text = client.calls[0][0]
    assert "def add(a, b):" in text
    assert "return a + b" in text
    # The import lines (1-2) should NOT be in the symbol's body.
    assert "import numpy as np" not in text


def test_embed_falls_back_to_empty_source_when_file_missing(tmp_clone: Path) -> None:
    """A missing file yields empty source; the other fields still pin the symbol."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(
            repository_id=repo_id,
            file_path="src/missing.py",
            name="phantom",
            signature="()",
            start_line=1,
            end_line=1,
        ),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(0)])

    written = embed_and_persist_symbols(
        session, repo_id, embedding_client=client, settings=_fake_settings()
    )

    assert written == 1
    # No "Source:" block in the embedding text because the file is missing.
    assert "Source:" not in client.calls[0][0]


def test_embed_handles_out_of_range_line_numbers(tmp_clone: Path) -> None:
    """start_line past the end of the file → empty source, no crash."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="phantom",
            signature="()",
            start_line=9999,
            end_line=10000,
        ),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(0)])

    embed_and_persist_symbols(session, repo_id, embedding_client=client, settings=_fake_settings())

    assert "Source:" not in client.calls[0][0]


# ============================================================
# Imports
# ============================================================


def test_embed_includes_imports_in_text(tmp_clone: Path) -> None:
    """The file's import symbols appear in the embedding text's Imports: line."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="numpy",
            kind="import",
            signature="numpy",
            start_line=1,
            end_line=1,
        ),
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="torch",
            kind="import",
            signature="torch",
            start_line=2,
            end_line=2,
        ),
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="add",
            signature="(a, b)",
            start_line=4,
            end_line=6,
        ),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(0), _make_vec(0), _make_vec(0)])

    embed_and_persist_symbols(session, repo_id, embedding_client=client, settings=_fake_settings())

    # The third call's text (the `add` function) carries the imports.
    function_text = client.calls[0][2]
    assert "Imports: numpy, torch" in function_text


def test_embed_uses_signature_then_name_for_imports(tmp_clone: Path) -> None:
    """An import with no signature falls back to the symbol's name."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="numpy",
            kind="import",
            signature="",
            start_line=1,
            end_line=1,
        ),
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="add",
            signature="()",
            start_line=4,
            end_line=4,
        ),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(0), _make_vec(0)])

    embed_and_persist_symbols(session, repo_id, embedding_client=client, settings=_fake_settings())

    assert "Imports: numpy" in client.calls[0][1]


def test_embed_skips_non_import_symbols_in_imports_list(tmp_clone: Path) -> None:
    """A function/class in the same file is NOT listed as an import."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="numpy",
            kind="import",
            signature="numpy",
            start_line=1,
            end_line=1,
        ),
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="Helper",
            kind="class",
            signature="Helper",
            start_line=3,
            end_line=3,
        ),
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="add",
            kind="function",
            signature="()",
            start_line=4,
            end_line=4,
        ),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(0), _make_vec(0), _make_vec(0)])

    embed_and_persist_symbols(session, repo_id, embedding_client=client, settings=_fake_settings())

    fn_text = client.calls[0][2]
    assert "Imports: numpy" in fn_text
    # The class symbol is NOT in the imports list.
    pre_import = fn_text.split("Imports:", 1)[1]
    pre_source = pre_import.split("Source:", 1)[0]
    assert "Helper" not in pre_source


# ============================================================
# Failure mode
# ============================================================


def test_embed_error_marks_repo_as_errored(tmp_clone: Path) -> None:
    """An ``EmbeddingError`` records the message on the row + analysis.

    We deliberately do NOT touch ``index_status`` — the parse step
    already marked the repo ``'done'``; only embedding failed. The
    retriever gracefully returns ``[]`` until the user retries.
    """
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(repository_id=repo_id, file_path="src/core.py", name="add"),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client(EmbeddingError("simulated outage"))

    written = embed_and_persist_symbols(
        session, repo_id, embedding_client=client, settings=_fake_settings()
    )

    assert written == 0
    # index_status was NOT touched — the parse step's 'done' sticks.
    assert repo.index_status == "pending"
    assert repo.error is not None
    assert "simulated outage" in repo.error
    # The error is also appended to analysis_states.errors.
    analysis = session._analysis  # type: ignore[attr-defined]
    assert len(analysis.errors) == 1
    assert analysis.errors[0]["where"] == "indexer"
    assert analysis.errors[0]["type"] == "EmbeddingError"


def test_embed_error_suppresses_embedding_raise(tmp_clone: Path) -> None:
    """The function does NOT re-raise EmbeddingError — it returns 0."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(repository_id=repo_id, file_path="src/core.py", name="add"),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client(EmbeddingError("boom"))

    # Should not raise.
    written = embed_and_persist_symbols(
        session, repo_id, embedding_client=client, settings=_fake_settings()
    )
    assert written == 0


# ============================================================
# Repository lookup
# ============================================================


def test_embed_raises_if_repository_not_found(tmp_clone: Path) -> None:
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=uuid4(), analysis_id=aid, clone_path=str(tmp_clone))
    session = _build_session(repo, [], analysis_id=aid, repo_lookup=None)
    client = _fake_embedding_client([])

    with pytest.raises(ValueError, match=r"Repository .* not found"):
        embed_and_persist_symbols(
            session, repo_id, embedding_client=client, settings=_fake_settings()
        )


# ============================================================
# Ordering
# ============================================================


def test_embed_inputs_match_symbol_order(tmp_clone: Path) -> None:
    """The Nth input text corresponds to the Nth symbol row."""
    repo_id = uuid4()
    aid = uuid4()
    repo = _make_repository(repository_id=repo_id, analysis_id=aid, clone_path=str(tmp_clone))
    symbols = [
        _make_symbol(
            repository_id=repo_id,
            file_path="src/core.py",
            name="add",
            signature="(a, b)",
            start_line=4,
            end_line=6,
        ),
        _make_symbol(
            repository_id=repo_id,
            file_path="src/math.py",
            name="subtract",
            signature="(a, b)",
            start_line=1,
            end_line=2,
        ),
    ]
    session = _build_session(repo, symbols, analysis_id=aid, repo_lookup=repo)
    client = _fake_embedding_client([_make_vec(0), _make_vec(0)])

    embed_and_persist_symbols(session, repo_id, embedding_client=client, settings=_fake_settings())

    texts = client.calls[0]
    assert "Name: add" in texts[0]
    assert "Name: subtract" in texts[1]
    # The rows are persisted in the same order the symbols were processed.
    added = [call.args[0] for call in session.add.call_args_list]
    assert added[0].symbol_id == symbols[0].symbol_id
    assert added[1].symbol_id == symbols[1].symbol_id
