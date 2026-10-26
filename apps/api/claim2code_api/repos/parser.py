"""Multi-language code symbol extractor.

Public entry point: :func:`parse_file`. Detects the language from the
file extension, parses the source with tree-sitter (or stdlib
:mod:`ast` for Python-only argparse / dataclass nodes), and returns a
list of normalised :class:`CodeSymbol` records.

Design notes:

  - **Why tree-sitter over Python** :mod:`ast`: tree-sitter survives
    malformed code (common in half-finished research repos) and gives
    us one normalised shape across six languages. The Python-only
    :func:`_python_argparse_dataclass` pass uses the stdlib ``ast``
    module — that's the one case where tree-sitter would be
    awkward because the AST for an ``ArgumentParser`` call is just a
    regular function call.

  - **Why manual AST walks instead of tree-sitter queries**: queries
    differ subtly between grammars (TS uses ``member_expression``,
    JS uses ``member_expression`` too but with different node shapes).
    A small per-language walker is easier to read and debug than a
    query-string DSL.

  - **Parent linkage**: a method's ``parent_symbol_id`` points at the
    enclosing class. The mapping is built in two passes — first collect
    all top-level symbols, then assign method parents.

  - **Line / column**: tree-sitter returns 0-indexed byte offsets and
    (row, column) points. We convert to 1-indexed lines (matching what
    editors / the frontend display) and 0-indexed columns (tree-sitter
    columns are byte offsets within the line, not Unicode codepoints,
    but that's good enough for "click here to jump").
"""

from __future__ import annotations

import ast as _ast
import re
import shlex
from pathlib import Path

from tree_sitter import Language, Node, Parser

from claim2code_api.repos.exceptions import (
    ParseError,
    UnsupportedLanguageError,
)
from claim2code_api.repos.schemas import (
    CodeLanguage,
    CodeSymbol,
    CodeSymbolKind,
)

# ============================================================
# Language detection + grammar registry
# ============================================================

# File extension → (CodeLanguage, tree-sitter grammar module name,
# language-function attribute name on the grammar module). The TS
# grammar is special: it exposes ``language_typescript`` and
# ``language_tsx`` rather than a single ``language()``. We resolve
# the function lazily so importing this module doesn't pay for
# grammars we don't use.
_EXTENSION_MAP: dict[str, tuple[CodeLanguage, str, str]] = {
    ".py": (CodeLanguage.PYTHON, "tree_sitter_python", "language"),
    ".js": (CodeLanguage.JAVASCRIPT, "tree_sitter_javascript", "language"),
    ".jsx": (CodeLanguage.JAVASCRIPT, "tree_sitter_javascript", "language"),
    ".mjs": (CodeLanguage.JAVASCRIPT, "tree_sitter_javascript", "language"),
    ".cjs": (CodeLanguage.JAVASCRIPT, "tree_sitter_javascript", "language"),
    ".ts": (CodeLanguage.TYPESCRIPT, "tree_sitter_typescript", "language_typescript"),
    ".tsx": (CodeLanguage.TYPESCRIPT, "tree_sitter_typescript", "language_tsx"),
    ".java": (CodeLanguage.JAVA, "tree_sitter_java", "language"),
    ".go": (CodeLanguage.GO, "tree_sitter_go", "language"),
    ".rs": (CodeLanguage.RUST, "tree_sitter_rust", "language"),
}

# Shell launch scripts carry no grammar: :func:`_parse_shell` reads
# their ``--flag value`` arguments without tree-sitter.
_SHELL_EXTENSIONS: set[str] = {".sh", ".bash"}

_LANGUAGE_CACHE: dict[CodeLanguage, Language] = {}


def _get_language(code_lang: CodeLanguage) -> Language:
    """Load the tree-sitter Language for a code language, with caching."""
    cached = _LANGUAGE_CACHE.get(code_lang)
    if cached is not None:
        return cached

    # Find the grammar module by mapping back from CodeLanguage.
    grammar_module_name = next(
        mod for _ext, (lang, mod, _fn) in _EXTENSION_MAP.items() if lang == code_lang
    )
    # TS has two language functions — pick the .ts one for caching
    # purposes (tsx is a superset of ts grammar in most node types).
    language_fn_name = next(
        fn for _ext, (lang, _mod, fn) in _EXTENSION_MAP.items() if lang == code_lang
    )
    try:
        grammar_module = __import__(grammar_module_name)
        capsule = getattr(grammar_module, language_fn_name)()
    except (ImportError, AttributeError) as exc:
        raise UnsupportedLanguageError(
            f"tree-sitter grammar for {code_lang!r} is not installed"
        ) from exc
    lang = Language(capsule)
    _LANGUAGE_CACHE[code_lang] = lang
    return lang


def detect_language(path: Path) -> CodeLanguage:
    """Return the :class:`CodeLanguage` for a path's extension.

    Raises :class:`UnsupportedLanguageError` if the extension is not in
    the v1 supported-languages set.
    """
    ext = path.suffix.lower()
    if ext in _SHELL_EXTENSIONS:
        return CodeLanguage.SHELL
    if ext not in _EXTENSION_MAP:
        raise UnsupportedLanguageError(f"unsupported file extension {ext!r} for {path.name!r}")
    return _EXTENSION_MAP[ext][0]


# ============================================================
# Public entry point
# ============================================================


def parse_file(path: Path) -> list[CodeSymbol]:
    """Parse a single source file into a list of :class:`CodeSymbol`.

    Args:
        path: Absolute or relative path to a source file. The file's
            extension determines the language.

    Returns:
        A list of normalised symbol records. The list may be empty if
        the file parses but contains no top-level definitions.

    Raises:
        UnsupportedLanguageError: If the extension is not supported.
        ParseError: If tree-sitter or ``ast`` raises an unexpected
            exception (e.g. file too large, IO error, etc.).
    """
    code_lang = detect_language(path)
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise ParseError(f"could not read {path}: {exc}") from exc

    if code_lang == CodeLanguage.SHELL:
        return _parse_shell(path, source_bytes)

    parser = Parser(_get_language(code_lang))
    tree = parser.parse(source_bytes)

    if code_lang == CodeLanguage.PYTHON:
        return _parse_python(path, source_bytes, tree)
    if code_lang == CodeLanguage.JAVASCRIPT:
        return _parse_javascript(path, source_bytes, tree)
    if code_lang == CodeLanguage.TYPESCRIPT:
        return _parse_typescript(path, source_bytes, tree)
    if code_lang == CodeLanguage.JAVA:
        return _parse_java(path, source_bytes, tree)
    if code_lang == CodeLanguage.GO:
        return _parse_go(path, source_bytes, tree)
    if code_lang == CodeLanguage.RUST:
        return _parse_rust(path, source_bytes, tree)
    # Unreachable; detect_language enforces the union.
    raise UnsupportedLanguageError(str(code_lang))


# ============================================================
# Shell launch-script parser
# ============================================================


def _parse_shell(path: Path, source: bytes) -> list[CodeSymbol]:
    """Extract one ``CONFIG_SCHEMA`` symbol per command that passes flags.

    Research repos set the paper's hyperparameters in launch scripts
    (``python run_glue.py --lora_r 8 --lora_alpha 16 ...``), not in the
    Python defaults, so a claim about them can only be checked here.
    Backslash continuations are joined into one logical command whose
    line span covers all of them. The symbol's name is the entry point
    (the first ``*.py`` argument, else the ``-m`` module, else the
    program) and its signature the ``--flag [value]`` arguments.

    The docstring spells the script's stem and ``--model_name_or_path``
    out as words: the lexical index reads ``deberta_v2_xxlarge_cola.sh``
    or ``microsoft/deberta-v2-xxlarge`` as one opaque token, so a claim
    about "DeBERTa XXL on CoLA" would otherwise never match it.
    """
    text = source.decode("utf-8", errors="replace")
    stem_words = _shell_words(path.stem)
    symbols: list[CodeSymbol] = []
    for start_line, end_line, command in _shell_logical_lines(text):
        try:
            tokens = shlex.split(command, comments=True)
        except ValueError:
            # Unbalanced quotes: fall back to whitespace splitting.
            tokens = command.split()
        flags = _shell_flag_args(tokens)
        if not flags:
            continue
        model = _shell_flag_value(tokens, "--model_name_or_path")
        doc = f"Launch script {stem_words}"
        if model:
            doc += f"; model {_shell_words(model)}"
        symbols.append(
            CodeSymbol(
                kind=CodeSymbolKind.CONFIG_SCHEMA,
                language=CodeLanguage.SHELL,
                file_path=str(path),
                start_line=start_line,
                start_col=0,
                end_line=end_line,
                end_col=0,
                name=_shell_entry_point(tokens),
                signature=" ".join(flags),
                docstring=doc,
            )
        )
    return symbols


def _shell_logical_lines(text: str) -> list[tuple[int, int, str]]:
    """Join backslash-continued lines; return ``(start, end, command)``."""
    out: list[tuple[int, int, str]] = []
    pending: list[str] = []
    start = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not pending:
            start = lineno
        line = raw.rstrip()
        if line.endswith("\\"):
            pending.append(line[:-1])
            continue
        pending.append(line)
        command = " ".join(pending).strip()
        pending = []
        if command and not command.startswith("#"):
            out.append((start, lineno, command))
    if pending:
        out.append((start, start + len(pending) - 1, " ".join(pending).strip()))
    return out


def _shell_flag_args(tokens: list[str]) -> list[str]:
    """Return the ``--flag [value]`` pairs of a command, one string each."""
    flags: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") and len(tok) > 2:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            takes_value = nxt is not None and (
                not nxt.startswith("-") or nxt.lstrip("-").replace(".", "", 1).isdigit()
            )
            if "=" not in tok and takes_value:
                flags.append(f"{tok} {nxt}")
                i += 2
                continue
            flags.append(tok)
        i += 1
    return flags


def _shell_flag_value(tokens: list[str], flag: str) -> str | None:
    """Return the value passed to ``flag`` (``--f v`` or ``--f=v``)."""
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(flag + "="):
            return tok[len(flag) + 1 :]
    return None


def _shell_words(text: str) -> str:
    """``microsoft/deberta-v2-xxlarge`` → ``microsoft deberta v2 xxlarge``."""
    return " ".join(w for w in re.split(r"[^A-Za-z0-9]+", text) if w)


def _shell_entry_point(tokens: list[str]) -> str:
    """Name a command by the script it runs (``run_glue.py``, ``-m pkg``)."""
    for tok in tokens:
        if tok.endswith(".py"):
            return tok
    for i, tok in enumerate(tokens[:-1]):
        if tok == "-m":
            return tokens[i + 1]
    return tokens[0] if tokens else "<command>"


# ============================================================
# Python parser
# ============================================================


def _parse_python(path: Path, source: bytes, tree: object) -> list[CodeSymbol]:
    """Extract symbols from a Python source file.

    Two passes:
      1. tree-sitter — captures top-level functions, classes, imports,
         and module-level assignments. Walks into class bodies to
         assign method ``parent_symbol_id`` to the enclosing class.
      2. stdlib :mod:`ast` — augments with argparse / dataclass
         records, which tree-sitter treats as plain function calls and
         decorated classes respectively.

    Both passes share ``path`` so the frontend can render a unified
    symbol list.
    """
    symbols: list[CodeSymbol] = []
    root = _tree_root(tree)

    # ---- tree-sitter pass ----
    class_stack: list[str] = []
    for child in root.children:
        node: Node = child
        if node.type == "decorated_definition":
            # Unwrap to the actual function/class def.
            inner = node.child(node.named_child_count - 1) if node.named_child_count else None
            if inner is None:
                continue
            node = inner
        if node.type not in {"function_definition", "class_definition"}:
            continue
        name_node = _first_named_child_of_type(node, "identifier")
        if name_node is None:
            continue
        name = _text(source, name_node)
        if not name:
            continue
        kind = (
            CodeSymbolKind.CLASS
            if node.type == "class_definition"
            else (CodeSymbolKind.METHOD if class_stack else CodeSymbolKind.FUNCTION)
        )
        signature = _python_signature(source, node)
        docstring = _python_docstring(source, node)
        symbols.append(
            CodeSymbol(
                kind=kind,
                language=CodeLanguage.PYTHON,
                file_path=str(path),
                start_line=_line(node.start_point),
                start_col=node.start_point[1],
                end_line=_line(node.end_point),
                end_col=node.end_point[1],
                name=name,
                signature=signature,
                docstring=docstring,
            )
        )
        if node.type == "class_definition":
            class_stack.append(name)
            body = _first_named_child_of_type(node, "block")
            if body is not None:
                _walk_python_methods(source, body, path, name, symbols)
            class_stack.pop()
        continue

    # Top-level imports + constants
    for child in root.children:
        node = child
        if node.type == "import_statement":
            for name_node in _find_all_of_type(node, "dotted_name"):
                symbols.append(
                    CodeSymbol(
                        kind=CodeSymbolKind.IMPORT,
                        language=CodeLanguage.PYTHON,
                        file_path=str(path),
                        start_line=_line(name_node.start_point),
                        start_col=name_node.start_point[1],
                        end_line=_line(name_node.end_point),
                        end_col=name_node.end_point[1],
                        name=_text(source, name_node),
                        signature=_text(source, node),
                    )
                )
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            name_text = _text(source, module_node) if module_node else ""
            symbols.append(
                CodeSymbol(
                    kind=CodeSymbolKind.IMPORT,
                    language=CodeLanguage.PYTHON,
                    file_path=str(path),
                    start_line=_line(node.start_point),
                    start_col=node.start_point[1],
                    end_line=_line(node.end_point),
                    end_col=node.end_point[1],
                    name=name_text or "<unknown>",
                    signature=_text(source, node),
                )
            )
        elif node.type == "expression_statement":
            # Top-level constant assignment: NAME = <literal>
            assign = _first_named_child_of_type(node, "assignment")
            if assign is None:
                continue
            left = assign.child_by_field_name("left")
            if left is None or left.type != "identifier":
                continue
            symbols.append(
                CodeSymbol(
                    kind=CodeSymbolKind.CONSTANT,
                    language=CodeLanguage.PYTHON,
                    file_path=str(path),
                    start_line=_line(left.start_point),
                    start_col=left.start_point[1],
                    end_line=_line(assign.end_point),
                    end_col=assign.end_point[1],
                    name=_text(source, left),
                    signature=_text(source, assign),
                )
            )

    # ---- ast pass for argparse + dataclass ----
    try:
        ast_tree = _ast.parse(source.decode("utf-8", errors="replace"))
    except SyntaxError:
        # Tree-sitter survived the malformed code; we don't double-fault.
        return symbols
    _python_argparse_dataclass(ast_tree, path, symbols)
    return symbols


def _walk_python_methods(
    source: bytes,
    block: object,
    path: Path,
    class_name: str,
    out: list[CodeSymbol],
) -> None:
    """Recursively walk a class body and emit one :class:`CodeSymbol` per method.

    Methods are linked back to the class via the parent_symbol_id we
    store at the workflow layer (Phase 3 v1 keeps parent linkage as a
    name-based lookup; the DB-side FK is wired in commit 3).
    """
    for child in block.children:  # type: ignore[attr-defined]
        node = child
        if node.type == "decorated_definition":
            inner = node.child(node.named_child_count - 1) if node.named_child_count else None
            if inner is not None:
                node = inner
        if node.type != "function_definition":
            continue
        name_node = _first_named_child_of_type(node, "identifier")
        if name_node is None:
            continue
        name = _text(source, name_node)
        if not name:
            continue
        out.append(
            CodeSymbol(
                kind=CodeSymbolKind.METHOD,
                language=CodeLanguage.PYTHON,
                file_path=str(path),
                start_line=_line(node.start_point),
                start_col=node.start_point[1],
                end_line=_line(node.end_point),
                end_col=node.end_point[1],
                name=name,
                signature=_python_signature(source, node),
                docstring=_python_docstring(source, node),
            )
        )


def _python_signature(source: bytes, node: object) -> str:
    """Return the parenthesised parameter list of a Python function/method."""
    params = _first_named_child_of_type(node, "parameters")  # type: ignore[attr-defined]
    if params is None:
        return ""
    return _text(source, params)


def _python_docstring(source: bytes, node: object) -> str | None:
    """Return the first string literal in a function/class body, if any."""
    body = _first_named_child_of_type(node, "block")  # type: ignore[attr-defined]
    if body is None:
        return None
    for child in body.children:  # type: ignore[attr-defined]
        if child.type == "expression_statement":
            inner = child.named_children[0] if child.named_children else None
            if inner is not None and inner.type in {"string", "concatenated_string"}:
                return _text(source, child).strip()
        break
    return None


def _python_argparse_dataclass(tree: _ast.AST, path: Path, out: list[CodeSymbol]) -> None:
    """Augment ``out`` with argparse / dataclass records (stdlib :mod:`ast`).

    Walks the module-level AST. For each ``ast.ClassDef`` with one or
    more ``@dataclasses.dataclass`` (or ``@dataclass``) decorators,
    emits one :class:`CodeSymbol` of kind ``CONFIG_SCHEMA``. For each
    ``ast.Call`` whose function ends in ``add_argument``, emits one
    :class:`CodeSymbol` of kind ``CONFIG_SCHEMA`` with the argument
    name and any default.
    """
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            decorator_names = (
                {_ast.unparse(d) for d in node.decorator_list}
                if hasattr(_ast, "unparse")
                else set()
            )
            is_dataclass = (
                any(
                    d.endswith("dataclass") or d.endswith("dataclasses.dataclass")
                    for d in decorator_names
                )
                if decorator_names
                else False
            )
            if is_dataclass:
                out.append(
                    CodeSymbol(
                        kind=CodeSymbolKind.CONFIG_SCHEMA,
                        language=CodeLanguage.PYTHON,
                        file_path=str(path),
                        start_line=node.lineno,
                        start_col=node.col_offset,
                        end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                        end_col=(getattr(node, "end_col_offset", 0) or 0),
                        name=node.name,
                        signature=f"@dataclass {node.name}",
                        docstring=_ast.get_docstring(node),
                    )
                )
        elif isinstance(node, _ast.Call):
            func = node.func
            func_name = (
                func.attr
                if isinstance(func, _ast.Attribute)
                else func.id
                if isinstance(func, _ast.Name)
                else None
            )
            if func_name != "add_argument":
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, _ast.Constant) and isinstance(first.value, str):
                out.append(
                    CodeSymbol(
                        kind=CodeSymbolKind.CONFIG_SCHEMA,
                        language=CodeLanguage.PYTHON,
                        file_path=str(path),
                        start_line=first.lineno,
                        start_col=first.col_offset,
                        end_line=getattr(node, "end_lineno", first.lineno) or first.lineno,
                        end_col=(getattr(node, "end_col_offset", 0) or 0),
                        name=first.value,
                        signature=_format_call_sig(node),
                    )
                )


def _format_call_sig(node: _ast.Call) -> str:
    """Best-effort textual signature for an ``add_argument`` call."""
    parts: list[str] = []
    for a in node.args:
        try:
            parts.append(_ast.unparse(a))
        except Exception:  # noqa: BLE001 — best-effort
            parts.append("<expr>")
    for kw in node.keywords:
        try:
            parts.append(f"{kw.arg}={_ast.unparse(kw.value)}")
        except Exception:  # noqa: BLE001
            parts.append(f"{kw.arg}=<expr>")
    return f"add_argument({', '.join(parts)})"


# ============================================================
# JavaScript / TypeScript parsers
# ============================================================


def _parse_javascript(path: Path, source: bytes, tree: object) -> list[CodeSymbol]:
    return _parse_c_like(
        path,
        source,
        tree,
        language=CodeLanguage.JAVASCRIPT,
        function_types={
            "function_declaration",
            "method_definition",
            "generator_function_declaration",
        },
        class_types={"class_declaration"},
        import_types={"import_statement"},
    )


def _parse_typescript(path: Path, source: bytes, tree: object) -> list[CodeSymbol]:
    return _parse_c_like(
        path,
        source,
        tree,
        language=CodeLanguage.TYPESCRIPT,
        function_types={
            "function_declaration",
            "method_definition",
            "generator_function_declaration",
        },
        class_types={"class_declaration"},
        import_types={"import_statement"},
    )


def _parse_c_like(
    path: Path,
    source: bytes,
    tree: object,
    language: CodeLanguage,
    function_types: set[str],
    class_types: set[str],
    import_types: set[str],
) -> list[CodeSymbol]:
    """Walk a C-family tree-sitter AST (JS/TS/Java/Go/Rust)."""
    symbols: list[CodeSymbol] = []
    root = _tree_root(tree)
    _walk_c_like(
        source,
        root,
        path,
        language,
        function_types,
        class_types,
        symbols,
        in_class=False,
    )
    # Imports are always top-level.
    for child in root.children:  # type: ignore[attr-defined]
        if child.type in import_types:
            _emit_c_import(child, source, path, language, symbols)
    return symbols


def _walk_c_like(
    source: bytes,
    node: object,
    path: Path,
    language: CodeLanguage,
    function_types: set[str],
    class_types: set[str],
    out: list[CodeSymbol],
    in_class: bool,
) -> None:
    """Recursively walk a C-family AST, emitting one symbol per matching node.

    Wrapper nodes (``export_statement``, ``lexical_declaration``,
    ``modifiers``, etc.) are transparent — we recurse into their
    children rather than emitting them as symbols.
    """
    wrapper_types = {
        "export_statement",
        "lexical_declaration",
        "variable_declaration",
        "modifiers",
        "decorator",
        "attribute",
        "visibility_modifier",
    }
    for child in node.children:  # type: ignore[attr-defined]
        if child.type in wrapper_types:
            _walk_c_like(
                source,
                child,
                path,
                language,
                function_types,
                class_types,
                out,
                in_class,
            )
            continue
        if child.type in class_types:
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _text(source, name_node)
            if not name:
                continue
            out.append(
                CodeSymbol(
                    kind=CodeSymbolKind.CLASS,
                    language=language,
                    file_path=str(path),
                    start_line=_line(child.start_point),
                    start_col=child.start_point[1],
                    end_line=_line(child.end_point),
                    end_col=child.end_point[1],
                    name=name,
                    signature="",
                )
            )
            body = child.child_by_field_name("body")
            if body is not None:
                _walk_c_like(
                    source,
                    body,
                    path,
                    language,
                    function_types,
                    class_types,
                    out,
                    in_class=True,
                )
        elif child.type in function_types:
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _text(source, name_node)
            if not name:
                continue
            # Go's ``method_declaration`` is always a method (it carries
            # the receiver); JS/TS/Java methods are inside class bodies.
            is_method = in_class or (
                language == CodeLanguage.GO and child.type == "method_declaration"
            )
            out.append(
                CodeSymbol(
                    kind=CodeSymbolKind.METHOD if is_method else CodeSymbolKind.FUNCTION,
                    language=language,
                    file_path=str(path),
                    start_line=_line(child.start_point),
                    start_col=child.start_point[1],
                    end_line=_line(child.end_point),
                    end_col=child.end_point[1],
                    name=name,
                    signature=_c_signature(source, child),
                )
            )


def _emit_c_import(
    node: Node,
    source: bytes,
    path: Path,
    language: CodeLanguage,
    out: list[CodeSymbol],
) -> None:
    name_node = node.child_by_field_name("source") or node
    if name_node is None:
        return
    name = _text(source, name_node)
    if not name:
        return
    out.append(
        CodeSymbol(
            kind=CodeSymbolKind.IMPORT,
            language=language,
            file_path=str(path),
            start_line=_line(node.start_point),
            start_col=node.start_point[1],
            end_line=_line(node.end_point),
            end_col=node.end_point[1],
            name=name,
            signature=_text(source, node),
        )
    )


def _c_signature(source: bytes, node: Node) -> str:
    params = node.child_by_field_name("parameters")
    if params is None:
        return ""
    return _text(source, params)


# Java / Go / Rust all use the C-family walker with slightly different
# node-type sets. Each language exposes its own thin wrapper so the
# type sets stay local and readable.


def _parse_java(path: Path, source: bytes, tree: object) -> list[CodeSymbol]:
    return _parse_c_like(
        path,
        source,
        tree,
        language=CodeLanguage.JAVA,
        function_types={"method_declaration", "constructor_declaration"},
        class_types={"class_declaration", "interface_declaration"},
        import_types={"import_declaration"},
    )


def _parse_go(path: Path, source: bytes, tree: object) -> list[CodeSymbol]:
    return _parse_c_like(
        path,
        source,
        tree,
        language=CodeLanguage.GO,
        function_types={"function_declaration", "method_declaration"},
        class_types={"type_declaration"},
        import_types={"import_declaration"},
    )


def _parse_rust(path: Path, source: bytes, tree: object) -> list[CodeSymbol]:
    return _parse_c_like(
        path,
        source,
        tree,
        language=CodeLanguage.RUST,
        function_types={"function_item"},
        class_types={"struct_item", "trait_item", "enum_item", "impl_item"},
        import_types={"use_declaration"},
    )


# ============================================================
# tree-sitter node helpers
# ============================================================


def _tree_root(tree: object) -> Node:
    """Return the root node of a tree-sitter Tree."""
    return tree.root_node  # type: ignore[attr-defined]


def _first_named_child_of_type(node: object, type_name: str) -> Node | None:
    """Return the first named child whose ``.type`` matches ``type_name``."""
    for child in node.children:  # type: ignore[attr-defined]
        if child.type == type_name:
            return child
    return None


def _find_all_of_type(node: object, type_name: str) -> list[Node]:
    """Return all named children (recursively flattened) of a given type."""
    found: list[Node] = []

    def visit(n: object) -> None:
        if n.type == type_name:  # type: ignore[attr-defined]
            found.append(n)  # type: ignore[arg-type]
        for c in n.children:  # type: ignore[attr-defined]
            visit(c)

    visit(node)
    return found


def _text(source: bytes, node: Node) -> str:
    """Return the source bytes for a node, decoded as UTF-8 with replacement."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _line(point: tuple[int, int]) -> int:
    """Convert tree-sitter's 0-indexed row to a 1-indexed line number."""
    return point[0] + 1
