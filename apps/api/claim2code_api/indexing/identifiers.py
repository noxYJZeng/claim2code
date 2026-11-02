"""Tokenize prose and code identifiers without a project-specific vocabulary."""
from __future__ import annotations

import re

_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def identifier_words(text: str, *, min_length: int = 1) -> set[str]:
    """Keep compound names and their snake/camel/acronym-separated words.

    Keeping the original token matters for mixed-case acronyms such as
    ``LoRA``; splitting alone would turn it into two unhelpful syllables.
    """
    if not text:
        return set()
    acronym_separated = _ACRONYM_BOUNDARY.sub(r"\1 \2", text)
    separated = _CAMEL_BOUNDARY.sub(r"\1 \2", acronym_separated)
    words = {w.lower() for w in _WORDS.findall(text)}
    words.update(w.lower() for w in _WORDS.findall(acronym_separated))
    words.update(w.lower() for w in _WORDS.findall(separated))
    return {word for word in words if len(word) >= min_length}


__all__ = ["identifier_words"]
