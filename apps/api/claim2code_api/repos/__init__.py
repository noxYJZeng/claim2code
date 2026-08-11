"""Repository Intelligence — Phase 3.

Sub-modules:

  - :mod:`claim2code_api.repos.exceptions` — typed errors
  - :mod:`claim2code_api.repos.schemas` — Pydantic models (parsed +
    API-response layers, mirroring ``papers/schemas.py``)
  - :mod:`claim2code_api.repos.parser` — tree-sitter multi-language
    symbol extraction (Python ``ast`` fallback for argparse / dataclass)
  - :mod:`claim2code_api.repos.cloner` — ``git clone --depth 1`` wrapper
  - :mod:`claim2code_api.repos.configs` — YAML / JSON / TOML / INI readers
  - :mod:`claim2code_api.repos.lockfiles` — requirements / poetry /
    npm / Pipfile extractors
  - :mod:`claim2code_api.repos.storage` — disk helpers under
    ``data/repos/``
  - :mod:`claim2code_api.repos.workflow` — sync orchestrator
    (clone → walk → parse → persist)
  - :mod:`claim2code_api.repos.router` — FastAPI routes

Phase 3 deliberately does **not** include embeddings, retrieval, or
verdict assignment — those land in Phase 4-6.
"""
