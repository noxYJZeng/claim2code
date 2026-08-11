"""Semantic indexing layer (Phase 4).

This package turns code symbols into embedding vectors and
back, so Phase 5+ can retrieve ``top_k`` candidate symbols for
a paper claim via cosine similarity.

Submodules:

  - :mod:`claim2code_api.indexing.embedding_client` — the
    provider-agnostic ``EmbeddingClient`` interface + the
    OpenAI adapter.
  - :mod:`claim2code_api.indexing.symbol_repr` — turns a
    ``CodeSymbol`` ORM row into a single string suitable for
    embedding (added in commit 2).
  - :mod:`claim2code_api.indexing.indexer` — batched embed
    + persist (added in commit 4).
  - :mod:`claim2code_api.indexing.retriever` — cosine search
    (added in commit 5).
"""
from __future__ import annotations
