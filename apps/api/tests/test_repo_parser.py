"""Unit tests for :mod:`claim2code_api.repos.parser`.

The tree-sitter grammars are real (not mocked) — they are installed
under :mod:`tree_sitter_python` etc. and have no network dependency.
Each test writes a synthetic source file to a ``tmp_path`` and calls
:func:`parse_file` directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claim2code_api.repos.exceptions import UnsupportedLanguageError
from claim2code_api.repos.parser import detect_language, parse_file
from claim2code_api.repos.schemas import CodeLanguage, CodeSymbolKind

# ============================================================
# detect_language
# ============================================================


def test_detect_language_python(tmp_path: Path) -> None:
    p = tmp_path / "foo.py"
    p.write_text("")
    assert detect_language(p) == CodeLanguage.PYTHON


def test_detect_language_typescript_tsx(tmp_path: Path) -> None:
    for ext, lang in [
        (".js", CodeLanguage.JAVASCRIPT),
        (".ts", CodeLanguage.TYPESCRIPT),
        (".tsx", CodeLanguage.TYPESCRIPT),
        (".java", CodeLanguage.JAVA),
        (".go", CodeLanguage.GO),
        (".rs", CodeLanguage.RUST),
    ]:
        p = tmp_path / f"file{ext}"
        p.write_text("")
        assert detect_language(p) == lang, ext


def test_detect_language_unsupported(tmp_path: Path) -> None:
    p = tmp_path / "foo.rb"
    p.write_text("")
    with pytest.raises(UnsupportedLanguageError):
        detect_language(p)


# ============================================================
# Python
# ============================================================


def test_python_function_and_class(tmp_path: Path) -> None:
    src = """\
def greet(name: str) -> str:
    \"\"\"Say hello.\"\"\"
    return f\"hi {name}\"


class Foo:
    def bar(self) -> int:
        return 1

    def baz(self) -> None:
        pass
"""
    p = tmp_path / "mod.py"
    p.write_text(src)
    symbols = parse_file(p)

    by_name = {(s.kind.value, s.name): s for s in symbols}

    assert (CodeSymbolKind.FUNCTION, "greet") in by_name
    assert (CodeSymbolKind.CLASS, "Foo") in by_name
    assert (CodeSymbolKind.METHOD, "bar") in by_name
    assert (CodeSymbolKind.METHOD, "baz") in by_name

    greet = by_name[(CodeSymbolKind.FUNCTION, "greet")]
    assert greet.start_line == 1
    assert greet.docstring == '"""Say hello."""'
    assert "name: str" in greet.signature


def test_python_import(tmp_path: Path) -> None:
    src = "import os\nfrom pathlib import Path\n"
    p = tmp_path / "mod.py"
    p.write_text(src)
    symbols = parse_file(p)

    imports = [s for s in symbols if s.kind == CodeSymbolKind.IMPORT]
    names = {s.name for s in imports}
    assert "os" in names
    assert "pathlib" in names  # from-import's module_name


def test_python_top_level_constant(tmp_path: Path) -> None:
    src = "MAX_DEPTH = 8\nDEFAULT_LR = 0.001\n"
    p = tmp_path / "cfg.py"
    p.write_text(src)
    symbols = parse_file(p)

    consts = [s for s in symbols if s.kind == CodeSymbolKind.CONSTANT]
    names = {s.name for s in consts}
    assert names == {"MAX_DEPTH", "DEFAULT_LR"}


def test_python_dataclass(tmp_path: Path) -> None:
    src = """\
from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float = 0.001
    batch_size: int = 32
"""
    p = tmp_path / "cfg.py"
    p.write_text(src)
    symbols = parse_file(p)

    config_schema = [s for s in symbols if s.kind == CodeSymbolKind.CONFIG_SCHEMA]
    assert any(s.name == "TrainConfig" for s in config_schema)


def test_python_argparse(tmp_path: Path) -> None:
    src = """\
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()
"""
    p = tmp_path / "cli.py"
    p.write_text(src)
    symbols = parse_file(p)

    config_schema = [s for s in symbols if s.kind == CodeSymbolKind.CONFIG_SCHEMA]
    names = {s.name for s in config_schema}
    assert "--lr" in names
    assert "--batch-size" in names


def test_python_survives_malformed_source(tmp_path: Path) -> None:
    """tree-sitter must not raise on incomplete Python."""
    src = "def broken(\n"
    p = tmp_path / "broken.py"
    p.write_text(src)
    # Should not raise; the tree may be empty or partial.
    symbols = parse_file(p)
    assert isinstance(symbols, list)


def test_python_lines_are_one_indexed(tmp_path: Path) -> None:
    src = "\n\ndef hello():\n    pass\n"
    p = tmp_path / "leading.py"
    p.write_text(src)
    symbols = parse_file(p)
    fn = next(s for s in symbols if s.name == "hello")
    assert fn.start_line == 3


# ============================================================
# JavaScript
# ============================================================


def test_javascript_function_and_class(tmp_path: Path) -> None:
    src = """\
export function greet(name) {
    return `hi ${name}`;
}

export class Animal {
    constructor(name) { this.name = name; }
    speak() { return '...'; }
}
"""
    p = tmp_path / "mod.js"
    p.write_text(src)
    symbols = parse_file(p)

    by_name = {(s.kind.value, s.name): s for s in symbols}
    assert (CodeSymbolKind.FUNCTION, "greet") in by_name
    assert (CodeSymbolKind.CLASS, "Animal") in by_name
    # ``speak`` is a method; ``constructor`` is also a method.
    methods = [s for s in symbols if s.kind == CodeSymbolKind.METHOD]
    names = {s.name for s in methods}
    assert "speak" in names


# ============================================================
# TypeScript
# ============================================================


def test_typescript_class_with_method(tmp_path: Path) -> None:
    src = """\
class Counter {
    private n: number = 0;
    inc(): void { this.n += 1; }
}
"""
    p = tmp_path / "counter.ts"
    p.write_text(src)
    symbols = parse_file(p)

    by_name = {(s.kind.value, s.name): s for s in symbols}
    assert (CodeSymbolKind.CLASS, "Counter") in by_name
    assert (CodeSymbolKind.METHOD, "inc") in by_name


# ============================================================
# Java
# ============================================================


def test_java_class_with_methods(tmp_path: Path) -> None:
    src = """\
public class Hello {
    public static void main(String[] args) { System.out.println(\"hi\"); }
    private int add(int a, int b) { return a + b; }
}
"""
    p = tmp_path / "Hello.java"
    p.write_text(src)
    symbols = parse_file(p)

    by_name = {(s.kind.value, s.name): s for s in symbols}
    assert (CodeSymbolKind.CLASS, "Hello") in by_name
    assert (CodeSymbolKind.METHOD, "main") in by_name
    assert (CodeSymbolKind.METHOD, "add") in by_name


# ============================================================
# Go
# ============================================================


def test_go_function_and_method(tmp_path: Path) -> None:
    src = """\
package main

func TopLevel() int { return 0 }

type Counter struct{}

func (c Counter) Inc() int { return 1 }
"""
    p = tmp_path / "main.go"
    p.write_text(src)
    symbols = parse_file(p)

    by_name = {(s.kind.value, s.name): s for s in symbols}
    assert (CodeSymbolKind.FUNCTION, "TopLevel") in by_name
    assert (CodeSymbolKind.METHOD, "Inc") in by_name


# ============================================================
# Rust
# ============================================================


def test_rust_struct_and_function(tmp_path: Path) -> None:
    src = """\
struct Counter { n: u32 }

impl Counter {
    fn inc(&mut self) { self.n += 1; }
}

fn top() -> u32 { 0 }
"""
    p = tmp_path / "lib.rs"
    p.write_text(src)
    symbols = parse_file(p)

    by_name = {(s.kind.value, s.name): s for s in symbols}
    # Rust struct counts as a class.
    assert (CodeSymbolKind.CLASS, "Counter") in by_name
    assert (CodeSymbolKind.FUNCTION, "top") in by_name


# ============================================================
# Misc
# ============================================================


def test_empty_file_returns_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "empty.py"
    p.write_text("")
    assert parse_file(p) == []


# ============================================================
# Shell launch scripts
# ============================================================


def test_detect_language_shell(tmp_path: Path) -> None:
    p = tmp_path / "train.sh"
    p.write_text("")
    assert detect_language(p) == CodeLanguage.SHELL


def test_shell_launch_command_becomes_config_schema(tmp_path: Path) -> None:
    p = tmp_path / "roberta_base_mrpc.sh"
    p.write_text(
        "export num_gpus=8\n"
        "# launch\n"
        "python -m torch.distributed.launch --nproc_per_node=$num_gpus \\\n"
        "examples/text-classification/run_glue.py \\\n"
        "--model_name_or_path roberta-base \\\n"
        "--do_train \\\n"
        "--lora_r 8 \\\n"
        "--lora_alpha 16 \\\n"
        "--seed -1\n"
        "echo done\n"
    )
    symbols = parse_file(p)
    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.kind == CodeSymbolKind.CONFIG_SCHEMA
    assert sym.language == CodeLanguage.SHELL
    assert sym.name == "examples/text-classification/run_glue.py"
    assert (sym.start_line, sym.end_line) == (3, 9)
    assert sym.signature == (
        "--nproc_per_node=$num_gpus --model_name_or_path roberta-base "
        "--do_train --lora_r 8 --lora_alpha 16 --seed -1"
    )
    assert sym.docstring == "Launch script roberta base mrpc; model roberta base"


def test_shell_module_entry_and_unbalanced_quotes(tmp_path: Path) -> None:
    p = tmp_path / "run.sh"
    p.write_text('python -m src.train --lr 2e-4 --note "unclosed\n')
    (sym,) = parse_file(p)
    assert sym.name == "src.train"
    assert sym.signature.startswith("--lr 2e-4")
    assert sym.docstring == "Launch script run"
