"""Pydantic models for the repository layer.

Three layers of schema (mirrors ``papers/schemas.py``):

  - :class:`CodeSymbol` / :class:`CodeSymbolKind` / :class:`CodeLanguage`
    are the **parsed-symbol** representation produced by the
    tree-sitter / ``ast`` parser in :mod:`claim2code_api.repos.parser`.
    They flow into the ``code_symbols`` table and into Phase 4
    embeddings.

  - :class:`RepositoryConfig` / :class:`RepositoryDependency` /
    :class:`ConfigFormat` / :class:`DependencyEcosystem` are the
    **parsed-config / parsed-lockfile** representation.

  - :class:`RepoCreateRequest` / :class:`RepoCreateResponse` /
    :class:`RepoMetadataResponse` / :class:`SymbolsResponse` /
    :class:`ConfigsResponse` / :class:`DependenciesResponse` /
    :class:`ErrorResponse` are the **API response** shapes that the
    FastAPI router returns. They are the only models the HTTP layer
    ever imports.

The separation is deliberate: a future Phase 4 ``embedding`` field can
be added to :class:`CodeSymbol` without touching the API-response models.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ============================================================
# 1. Parsed code symbol
# ============================================================


class CodeLanguage(StrEnum):
    """The source language of a :class:`CodeSymbol`.

    Phase 3 ships six grammars; C / C++ / PHP / Ruby / Kotlin / Swift /
    Scala are deferred (see ``phase-3-design.md`` §3).
    """

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"


class CodeSymbolKind(StrEnum):
    """What the AST node represents.

    ``CONFIG_SCHEMA`` is the Python-specific extra for argparse /
    dataclass nodes (see ``phase-3-design.md`` §2.2).
    """

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    IMPORT = "import"
    CONSTANT = "constant"
    CONFIG_SCHEMA = "config_schema"


class CodeSymbol(BaseModel):
    """One extracted code symbol.

    Line/column fields are 1-indexed (matching tree-sitter's row /
    column conventions) so the frontend can render "file:line:col"
    directly without re-indexing.

    ``signature`` is best-effort textual — for functions it is the
    parenthesised parameter list; for imports it is the full dotted
    import path; for classes it is empty.
    """

    model_config = ConfigDict(extra="forbid")

    kind: CodeSymbolKind
    language: CodeLanguage
    file_path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    start_col: int = Field(ge=0)
    end_line: int = Field(ge=1)
    end_col: int = Field(ge=0)
    name: str = Field(min_length=1)
    signature: str = ""
    docstring: str | None = None


class CodeSymbolList(BaseModel):
    """Wrapper returned by the parser module.

    Mirrors :class:`claim2code_api.papers.schemas.ClaimList` shape so
    the workflow layer can iterate uniformly.
    """

    model_config = ConfigDict(extra="forbid")

    symbols: list[CodeSymbol]


# ============================================================
# 2. Parsed config + lockfile
# ============================================================


class ConfigFormat(StrEnum):
    """The on-disk format of a config file."""

    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    INI = "ini"


class RepositoryConfig(BaseModel):
    """One parsed config file.

    ``content_json`` is the parsed tree (a ``dict`` for YAML/TOML/INI;
    whatever JSON returned for JSON). If the file is unparseable we
    still store the raw text under the same key so the frontend can
    show it; the parse error is logged in :attr:`error`.
    """

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(min_length=1)
    format: ConfigFormat
    size_bytes: int = Field(ge=0)
    content_json: dict
    error: str | None = None


class DependencyEcosystem(StrEnum):
    """Which package manager a :class:`RepositoryDependency` came from."""

    PYPI = "pypi"
    NPM = "npm"


class RepositoryDependency(BaseModel):
    """One flattened dependency row from a lockfile or requirements file.

    ``version_spec`` is the raw version constraint (``==2.1.0``,
    ``^1.0.0``, etc.). Phase 6 normalises these for cross-checking
    against paper text.
    """

    model_config = ConfigDict(extra="forbid")

    ecosystem: DependencyEcosystem
    name: str = Field(min_length=1)
    version_spec: str = ""
    source_file: str = Field(min_length=1)


# ============================================================
# 3. API response shapes
# ============================================================


class RepoCreateRequest(BaseModel):
    """Request body for ``POST /repositories``."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    analysis_id: UUID


class RepoCreateResponse(BaseModel):
    """Response body for ``POST /repositories``."""

    repository_id: UUID
    analysis_id: UUID
    url: str
    owner: str
    name: str
    n_files: int
    n_symbols: int
    n_configs: int
    n_dependencies: int
    index_status: str


class RepoMetadataResponse(BaseModel):
    """Response body for ``GET /repositories/{repository_id}``."""

    repository_id: UUID
    analysis_id: UUID
    url: str
    owner: str
    name: str
    default_branch: str
    clone_path: str
    n_files: int | None
    n_symbols: int | None
    index_status: str
    cloned_at: datetime
    indexed_at: datetime | None
    error: str | None


class SymbolsResponse(BaseModel):
    """Response body for ``GET /repositories/{repository_id}/symbols``.

    In v1 the response is the full flat list (paginated client-side);
    Phase 5+ can add cursor-based pagination once the row count
    becomes painful.
    """

    repository_id: UUID
    symbols: list[CodeSymbol]


class ConfigsResponse(BaseModel):
    """Response body for ``GET /repositories/{repository_id}/configs``."""

    repository_id: UUID
    configs: list[RepositoryConfig]


class DependenciesResponse(BaseModel):
    """Response body for ``GET /repositories/{repository_id}/dependencies``."""

    repository_id: UUID
    dependencies: list[RepositoryDependency]


class RepoErrorResponse(BaseModel):
    """Standard error body returned by the repo router."""

    error: str
    detail: str | None = None


# ============================================================
# 4. Phase 4 — semantic retrieval
# ============================================================


class RetrieveRequest(BaseModel):
    """Request body for ``POST /repositories/{repository_id}/retrieve``.

    The ``claim`` is a natural-language sentence (e.g. extracted
    from a paper). The endpoint returns the top-``top_k`` symbols
    from this repository whose structured representation is most
    semantically similar to the claim.

    ``top_k`` defaults to 5 — the design doc's acceptance criterion.
    A claim like "uses softmax(QK^T / sqrt(d))" might return 5
    candidate symbols; the user picks one (or none) and the
    Phase 6 verifier does the deterministic check.
    """

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class RetrieveMatch(BaseModel):
    """One hit in the returned retrieval list.

    Mirrors the parsed :class:`CodeSymbol` shape so the frontend
    can render the same widget it uses for ``GET /symbols``; the
    additional ``similarity`` field is the cosine score (0..1,
    higher is better).
    """

    model_config = ConfigDict(extra="forbid")

    symbol_id: UUID
    kind: CodeSymbolKind
    language: CodeLanguage
    file_path: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    name: str
    signature: str
    docstring: str | None
    similarity: float = Field(ge=0.0, le=1.0)


class RetrieveResponse(BaseModel):
    """Response body for ``POST /repositories/{repository_id}/retrieve``.

    ``symbols`` is the list of :class:`RetrieveMatch` (top-k by
    similarity descending); ``scores`` is the parallel list of
    similarity values, kept separate so the frontend can chart
    historical scores without re-parsing the symbol rows.
    """

    model_config = ConfigDict(extra="forbid")

    repository_id: UUID
    claim: str
    symbols: list[RetrieveMatch]
    scores: list[float]
