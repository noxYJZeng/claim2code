"""Unit tests for :mod:`claim2code_api.repos.configs`."""

from __future__ import annotations

from pathlib import Path

from claim2code_api.repos.configs import config_symbol, detect_format, parse_config_file
from claim2code_api.repos.schemas import CodeLanguage, CodeSymbolKind, ConfigFormat


def test_detect_format_yaml(tmp_path: Path) -> None:
    for ext in (".yaml", ".yml"):
        assert detect_format(tmp_path / f"f{ext}") is ConfigFormat.YAML


def test_detect_format_json(tmp_path: Path) -> None:
    assert detect_format(tmp_path / "f.json") is ConfigFormat.JSON


def test_detect_format_toml(tmp_path: Path) -> None:
    assert detect_format(tmp_path / "f.toml") is ConfigFormat.TOML


def test_detect_format_ini(tmp_path: Path) -> None:
    for ext in (".ini", ".cfg", ".conf"):
        assert detect_format(tmp_path / f"f{ext}") is ConfigFormat.INI


def test_detect_format_unknown(tmp_path: Path) -> None:
    assert detect_format(tmp_path / "f.txt") is None


# ============================================================
# YAML
# ============================================================


def test_yaml_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("name: hello\ncount: 3\nnested:\n  a: 1\n  b: 2\n")
    result = parse_config_file(p, "cfg.yaml")
    assert result.format is ConfigFormat.YAML
    assert result.error is None
    assert result.content_json == {"name": "hello", "count": 3, "nested": {"a": 1, "b": 2}}


def test_yaml_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    result = parse_config_file(p, "empty.yaml")
    assert result.content_json == {}
    assert result.error is None


def test_yaml_top_level_list_wrapped(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- one\n- two\n")
    result = parse_config_file(p, "list.yaml")
    assert result.error is None
    assert result.content_json == {"_root": ["one", "two"]}


# ============================================================
# JSON
# ============================================================


def test_json_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text('{"name": "hello", "items": [1, 2, 3]}')
    result = parse_config_file(p, "cfg.json")
    assert result.format is ConfigFormat.JSON
    assert result.content_json == {"name": "hello", "items": [1, 2, 3]}


def test_json_top_level_list_wrapped(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    result = parse_config_file(p, "list.json")
    assert result.error is None
    assert result.content_json == {"_root": [1, 2, 3]}


def test_json_syntax_error_surfaced(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    result = parse_config_file(p, "bad.json")
    assert result.error is not None
    assert "JSONDecodeError" in result.error
    assert result.content_json == {}


# ============================================================
# TOML
# ============================================================


def test_toml_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "pyproject.toml"
    p.write_text("[tool.black]\nline-length = 100\n[tool.ruff]\nline-length = 100\n")
    result = parse_config_file(p, "pyproject.toml")
    assert result.format is ConfigFormat.TOML
    assert result.content_json["tool"]["black"]["line-length"] == 100


def test_toml_syntax_error_surfaced(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("= invalid toml")
    result = parse_config_file(p, "bad.toml")
    assert result.error is not None
    assert result.content_json == {}


# ============================================================
# INI
# ============================================================


def test_ini_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "settings.ini"
    p.write_text("[database]\nhost = localhost\nport = 5432\n[cache]\nenabled = true\n")
    result = parse_config_file(p, "settings.ini")
    assert result.format is ConfigFormat.INI
    assert result.content_json == {
        "database": {"host": "localhost", "port": "5432"},
        "cache": {"enabled": "true"},
    }


def test_ini_default_only_no_sections(tmp_path: Path) -> None:
    p = tmp_path / "setup.cfg"
    p.write_text("key = value\nother = 1\n")
    result = parse_config_file(p, "setup.cfg")
    assert result.format is ConfigFormat.INI
    # No sections, but a DEFAULT block — surface it under that key.
    assert "DEFAULT" in result.content_json


def test_ini_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.ini"
    p.write_text("")
    result = parse_config_file(p, "empty.ini")
    assert result.content_json == {}
    assert result.error is None


# ============================================================
# Misc
# ============================================================


def test_unknown_extension_returns_error(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("hi")
    result = parse_config_file(p, "f.txt")
    assert result.error is not None
    assert "unrecognised" in result.error


def test_size_bytes_recorded(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("a: 1\n")
    result = parse_config_file(p, "cfg.yaml")
    assert result.size_bytes == len("a: 1\n")


# ============================================================
# config_symbol
# ============================================================


def test_config_symbol_spans_file_and_flattens_leaves(tmp_path: Path) -> None:
    text = "# DPO loss\nname: dpo\nbeta: 0.1\nopt:\n  lr: 1e-6\n  warmup: 150\n"
    p = tmp_path / "dpo.yaml"
    p.write_text(text)
    cfg = parse_config_file(p, "config/loss/dpo.yaml")

    sym = config_symbol(cfg, text)

    assert sym is not None
    assert sym.kind is CodeSymbolKind.CONFIG_SCHEMA
    assert sym.language is CodeLanguage.CONFIG
    assert sym.name == sym.file_path == "config/loss/dpo.yaml"
    assert (sym.start_line, sym.end_line) == (1, 6)
    assert sym.signature == "name: dpo; beta: 0.1; opt.lr: 1e-6; opt.warmup: 150"
    assert sym.docstring == "Config file config loss dpo yaml"


def test_config_symbol_skips_unparsed_and_oversized_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert config_symbol(parse_config_file(bad, "bad.json"), "{not json") is None

    big_text = "{" + ", ".join(f'"k{i}": {i}' for i in range(4000)) + "}"
    big = tmp_path / "vocab.json"
    big.write_text(big_text)
    assert config_symbol(parse_config_file(big, "vocab.json"), big_text) is None
