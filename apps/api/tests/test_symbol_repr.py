"""Unit tests for :mod:`claim2code_api.indexing.symbol_repr`.

These tests assert the **exact** string shape for every supported
symbol kind. The string is the input to the embedding model —
any change here silently changes retrieval results, so we lock
the shape byte-for-byte.

Coverage:

  - ``format_symbol`` returns a deterministic string for a
    well-formed function symbol.
  - Optional fields (docstring, imports, source) are omitted
    when empty.
  - The signature fallback ``'(no signature)'`` is used when the
    parser couldn't extract one.
  - Docstring whitespace is collapsed to a single line.
  - Imports are capped at 20 entries with a ``'+N more'`` suffix.
  - Source is truncated at 2 000 chars with a marker.
  - The function is pure: same input → same output.
  - Each symbol kind (function / class / import) gets a distinct
    output that still pins the symbol's identity.
"""

from __future__ import annotations

from types import SimpleNamespace

from claim2code_api.db.models import CodeSymbol
from claim2code_api.indexing.symbol_repr import format_symbol

# ============================================================
# Helpers
# ============================================================


def _sym(
    *,
    kind: str = "function",
    language: str = "python",
    file_path: str = "src/pkg/core.py",
    name: str = "add",
    signature: str = "(a: int, b: int) -> int",
    docstring: str | None = None,
    start_line: int = 10,
    end_line: int = 12,
) -> CodeSymbol:
    """Build a CodeSymbol-like row for tests.

    Most fields are irrelevant to ``format_symbol``; we use
    ``SimpleNamespace`` rather than a real ORM row so the tests
    don't need a DB session.
    """
    return SimpleNamespace(  # type: ignore[return-value]
        kind=kind,
        language=language,
        file_path=file_path,
        start_line=start_line,
        start_col=0,
        end_line=end_line,
        end_col=0,
        name=name,
        signature=signature,
        docstring=docstring,
        parent_symbol_id=None,
    )


# ============================================================
# Happy path
# ============================================================


def test_format_symbol_full_shape() -> None:
    """All fields populated → exact string match."""
    sym = _sym(docstring="Add two ints.")
    out = format_symbol(
        sym,
        source="def add(a: int, b: int) -> int:\n    return a + b",
        imports=["numpy as np", "torch"],
    )
    assert out == (
        "File: src/pkg/core.py\n"
        "Language: python\n"
        "Kind: function\n"
        "Name: add\n"
        "Signature: (a: int, b: int) -> int\n"
        "Docstring: Add two ints.\n"
        "Imports: numpy as np, torch\n"
        "Source:\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b"
    )


def test_format_symbol_emits_fields_in_fixed_order() -> None:
    """The header order is fixed: File → Language → Kind → Name → Signature."""
    sym = _sym()
    out = format_symbol(sym)
    lines = out.split("\n")
    assert lines[0] == "File: src/pkg/core.py"
    assert lines[1] == "Language: python"
    assert lines[2] == "Kind: function"
    assert lines[3] == "Name: add"
    assert lines[4] == "Signature: (a: int, b: int) -> int"


# ============================================================
# Optional fields
# ============================================================


def test_format_symbol_without_docstring() -> None:
    """No docstring → no Docstring: line."""
    sym = _sym(docstring=None)
    out = format_symbol(sym)
    assert "Docstring:" not in out


def test_format_symbol_without_imports() -> None:
    """Empty imports → no Imports: line."""
    sym = _sym()
    out = format_symbol(sym, imports=[])
    assert "Imports:" not in out


def test_format_symbol_without_source() -> None:
    """Empty source → no Source: line."""
    sym = _sym()
    out = format_symbol(sym, source="")
    assert "Source:" not in out


def test_format_symbol_signature_fallback() -> None:
    """Missing signature → '(no signature)' placeholder."""
    sym = _sym(signature="")
    out = format_symbol(sym)
    assert "Signature: (no signature)" in out


def test_format_symbol_includes_blank_docstring_as_none() -> None:
    """A whitespace-only docstring is treated as no docstring."""
    sym = _sym(docstring="   \n  ")
    out = format_symbol(sym)
    assert "Docstring:" not in out


# ============================================================
# Docstring whitespace
# ============================================================


def test_format_symbol_squashes_docstring_whitespace() -> None:
    """Newlines and tabs in docstrings collapse to single spaces."""
    sym = _sym(docstring="Line 1.\nLine 2.\n\nLine 4.")
    out = format_symbol(sym)
    assert "Docstring: Line 1. Line 2. Line 4." in out


def test_format_symbol_strips_docstring() -> None:
    """Leading/trailing whitespace in docstring is stripped."""
    sym = _sym(docstring="   the docstring   ")
    out = format_symbol(sym)
    assert "Docstring: the docstring" in out


# ============================================================
# Imports cap
# ============================================================


def test_format_symbol_caps_imports_at_20() -> None:
    """More than 20 imports get a '+N more' suffix."""
    sym = _sym()
    imports = [f"mod_{i}" for i in range(25)]
    out = format_symbol(sym, imports=imports)
    # First 20 visible, the rest summarised.
    assert "+5 more" in out
    # The 21st import is NOT listed verbatim.
    assert "mod_20" not in out
    # The 20th is still listed.
    assert "mod_19" in out


def test_format_symbol_includes_all_when_under_cap() -> None:
    """Up to 20 imports are listed verbatim."""
    sym = _sym()
    imports = [f"mod_{i}" for i in range(10)]
    out = format_symbol(sym, imports=imports)
    assert "Imports: " in out
    assert ", ... (+" not in out
    assert "mod_0" in out
    assert "mod_9" in out


# ============================================================
# Source truncation
# ============================================================


def test_format_symbol_keeps_source_under_cap() -> None:
    """Source shorter than 2 000 chars is unchanged."""
    sym = _sym()
    body = "x = 1\n" * 100  # 600 chars
    out = format_symbol(sym, source=body)
    assert "truncated" not in out
    # The body is included verbatim (minus the trailing newline,
    # which format_symbol strips before measuring).
    assert body.rstrip("\n") in out


def test_format_symbol_truncates_source_above_cap() -> None:
    """Source longer than 2 000 chars is truncated with a marker."""
    sym = _sym()
    body = "y" * 3000
    out = format_symbol(sym, source=body)
    # First 2000 chars present, marker present.
    assert "[... truncated ...]" in out
    # The marker is in the "Source:" block, not the header.
    assert out.split("Source:\n", 1)[1].startswith("y" * 2000)


def test_format_symbol_strips_trailing_whitespace_before_truncation() -> None:
    """Trailing newlines are stripped before measuring length."""
    sym = _sym()
    body = "x = 1" + "\n" * 50
    out = format_symbol(sym, source=body)
    # The trailing whitespace is stripped during measurement, so
    # the body is only 5 chars and no truncation occurs.
    assert "truncated" not in out


# ============================================================
# Purity / determinism
# ============================================================


def test_format_symbol_is_deterministic() -> None:
    """Same input → same output (byte-for-byte)."""
    sym = _sym(docstring="the doc")
    out1 = format_symbol(sym, source="def x(): pass", imports=["a"])
    out2 = format_symbol(sym, source="def x(): pass", imports=["a"])
    assert out1 == out2


def test_format_symbol_does_not_mutate_input() -> None:
    """The function must not mutate the symbol or its source."""
    sym = _sym(docstring="doc")
    sym.kind = "function"
    format_symbol(sym, source="source", imports=["a"])
    assert sym.kind == "function"
    assert sym.signature == "(a: int, b: int) -> int"


# ============================================================
# Per-kind identity
# ============================================================


def test_format_symbol_function_vs_class() -> None:
    """A class symbol's output pins the kind but shares the format."""
    fn = _sym(kind="function", name="helper")
    cls = _sym(kind="class", name="Helper")
    fn_out = format_symbol(fn)
    cls_out = format_symbol(cls)
    assert "Kind: function" in fn_out
    assert "Kind: class" in cls_out
    assert "Name: helper" in fn_out
    assert "Name: Helper" in cls_out


def test_format_symbol_import_uses_dotted_signature() -> None:
    """An import symbol's signature is typically its dotted path."""
    sym = _sym(
        kind="import",
        name="numpy",
        signature="numpy",
    )
    out = format_symbol(sym)
    assert "Kind: import" in out
    assert "Name: numpy" in out
    assert "Signature: numpy" in out


# ============================================================
# Imports as iterable
# ============================================================


def test_format_symbol_accepts_imports_as_iterator() -> None:
    """``imports`` may be any iterable, not just a list."""
    sym = _sym()
    out = format_symbol(sym, imports=(m for m in ["a", "b", "c"]))
    assert "Imports: a, b, c" in out
