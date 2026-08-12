"""Semantic + lexical indexing layer (Phase 4 + Phase 5).

This package turns code symbols into embedding vectors and
supports the per-leg retrieval scorers that the Phase 5 hybrid
ranker fuses. Each leg is a leaf module — Phase 5+ can swap
or extend any one of them without touching the others.

Submodules:

  - :mod:`claim2code_api.indexing.embedding_client` — the
    provider-agnostic ``EmbeddingClient`` interface + the
    OpenAI adapter (Phase 4).
  - :mod:`claim2code_api.indexing.symbol_repr` — turns a
    ``CodeSymbol`` ORM row into a single string suitable for
    embedding (Phase 4 commit 2).
  - :mod:`claim2code_api.indexing.indexer` — batched embed
    + persist (Phase 4 commit 4).
  - :mod:`claim2code_api.indexing.retriever` — semantic cosine
    search (Phase 4 commit 5).
  - :mod:`claim2code_api.indexing.lexical` — Postgres tsvector
    + ts_rank_cd lexical search (Phase 5 commit 2).
  - :mod:`claim2code_api.indexing.structural` — import-graph
    + file-name proximity boosters (Phase 5 commit 3).
"""

from __future__ import annotations
