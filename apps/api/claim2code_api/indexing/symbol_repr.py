"""Build a deterministic text representation of a ``CodeSymbol`` for embedding.

Why structured (not just the source body)
----------------------------------------
The Phase 4 design doc says one embedding per ``CodeSymbol``, and the
input to the embedding model shapes the retrieval behaviour. We
deliberately use a key-value structured format (File: / Language: /
Kind: / Name: / Signature: / Docstring: / Imports: / Source:) rather
than raw prose for two reasons:

  1. **Identifiability.** A claim like "uses softmax(QK^T / sqrt(d))"
     matches against ``Signature`` + ``Docstring`` + ``Source`` of the
     ``scaled_dot_product_attention`` symbol. With raw source only,
     the embedding has to learn that "softmax" appearing in the file
     body is the intent. Structured fields give the model a head start.
  2. **Truncation tolerance.** The OpenAI embedding API accepts 8 192
     tokens per input, but we cap the source at 2 000 chars to keep
     cost predictable. Structuring other fields first means even a
     truncated body still pins the symbol's identity.

Field order is fixed (the tests assert it byte-for-byte). Anything
that changes the order changes the embedding vector — the retrievers
in commit 5 depend on this string being stable.

What this function does NOT do
------------------------------
  - It does not read the source from disk. Callers (the indexer in
    commit 4) read ``<clone_path>/<file_path>`` between
    ``start_line`` and ``end_line`` and pass the result as
    ``source``. This keeps ``format_symbol`` pure and CI-testable.
  - It does not resolve imports. The caller computes
    ``imports_for_file(file_path)`` and passes the list. Pulling
    that logic in here would require a DB session, which would
    break the function's purity.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from claim2code_api.db.models import CodeSymbol

# Source body cap. The OpenAI embedding API accepts 8 192 tokens,
# but 2 000 chars (~500 tokens) keeps a single symbol's embedding
# cost under ~$0.0001 and prevents one huge class from "ballooning"
# the index.
_SOURCE_CHAR_CAP = 2000

# Trailing marker when source is truncated. The trailing newline
# keeps the appended (...) on its own line so the model treats it
# as a delimiter, not part of the body.
_TRUNCATION_MARKER = "\n[... truncated ...]"

# Compiled once: collapse runs of whitespace (including newlines)
# in the docstring so it fits on one line for the model. Multi-line
# docstrings are common in Python but the embedding model handles
# single-line fine, and we save tokens.
_WS_RE = re.compile(r"\s+")

# Cap import-list length. Twenty imports is enough for 99 % of
# symbols; long files often have hundreds of imports that just
# add noise.
_IMPORT_CAP = 20


def format_symbol(
    symbol: CodeSymbol,
    source: str = "",
    imports: Iterable[str] = (),
) -> str:
    """Build the embedding text for one ``CodeSymbol``.

    The returned string is deterministic — same input → same
    output — so the same symbol always gets the same embedding
    (modulo model upgrades; see ``EmbeddingClient`` for the
    ``model_name`` versioning).

    :param symbol: the ORM row from ``code_symbols``.
    :param source: the source body between ``start_line`` and
        ``end_line``. May be empty if the caller hasn't read the
        file (e.g. tests asserting basic shape).
    :param imports: the file's import targets (e.g. ``["numpy as np",
        "torch"]``). Empty for files that have no imports.
    """
    parts: list[str] = []

    # --- Header (key-value, fixed order) ---
    parts.append(f"File: {symbol.file_path}")
    parts.append(f"Language: {symbol.language}")
    parts.append(f"Kind: {symbol.kind}")
    parts.append(f"Name: {symbol.name}")
    # Signature is already best-effort from the parser (parenthesised
    # arg list for functions/methods, dotted path for imports).
    parts.append(f"Signature: {symbol.signature or '(no signature)'}")

    # --- Docstring (optional, single-line) ---
    doc = (symbol.docstring or "").strip()
    if doc:
        parts.append(f"Docstring: {_squash_ws(doc)}")

    # --- Imports (optional, capped) ---
    imports_list = list(imports)
    if imports_list:
        visible = imports_list[:_IMPORT_CAP]
        suffix = ""
        if len(imports_list) > _IMPORT_CAP:
            suffix = f", ... (+{len(imports_list) - _IMPORT_CAP} more)"
        parts.append(f"Imports: {', '.join(visible)}{suffix}")

    # --- Source (optional, capped) ---
    if source:
        body = source.rstrip("\n")
        if len(body) > _SOURCE_CHAR_CAP:
            body = body[:_SOURCE_CHAR_CAP] + _TRUNCATION_MARKER
        parts.append("Source:\n" + body)

    return "\n".join(parts)


def _squash_ws(text: str) -> str:
    """Collapse internal whitespace runs to a single space.

    Multi-line docstrings become single-line, which (a) saves
    tokens and (b) reads cleanly to the embedding model.
    """
    return _WS_RE.sub(" ", text).strip()
