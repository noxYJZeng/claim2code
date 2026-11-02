"""PyMuPDF-based PDF parser.

Phase 2 v1 extracts **sections + text** (the foundation every claim
cites). Equations and tables are detected and returned in the
:class:`Paper` model for later phases; the LLM claim extractor
**does not** include them in its prompt in v1 — see the
``docs/architecture/phase-2-design.md`` §3 "what we explicitly do
NOT do" list.

Section detection is font-size based: a paragraph whose font is
notably larger than the body font starts a new section. This is
robust enough for ACL / NeurIPS / arXiv-style papers and avoids
the brittleness of regex-on-toc heuristics. The "preamble" before
the first detected heading (Abstract + intro material) is grouped
under a synthetic ``""`` title.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pymupdf  # type: ignore[import-untyped]  # pymupdf ships no type stubs

from claim2code_api.papers.exceptions import PdfNoTextLayerError
from claim2code_api.papers.schemas import Equation, Paper, PaperSection, Table

# Minimum total text length (across all pages) below which we consider
# the PDF to be image-only / scanned. 200 chars ≈ two paragraphs of
# body text; real papers produce ~50k chars so this is a conservative
# floor.
_MIN_TOTAL_CHARS = 200

# A line whose font is this much larger than the body font is treated
# as a heading. arXiv / ICLR / CVPR / NeurIPS section headings are 12pt
# over a 10pt body (1.2x), so the threshold sits just below that.
_HEADING_FONT_RATIO = 1.15

# A "heading" line longer than this is almost always body text.
_HEADING_MAX_CHARS = 200


def parse_pdf(path: Path, *, filename: str | None = None) -> Paper:
    """Parse a PDF file into a :class:`Paper`.

    :raises PdfNoTextLayerError: if the PDF has effectively no
        extractable text (image-only / scanned).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = pymupdf.open(path)
    try:
        sections, equations, tables, total_chars = _parse_pages(doc)
        n_pages = doc.page_count
    finally:
        doc.close()

    if total_chars < _MIN_TOTAL_CHARS:
        raise PdfNoTextLayerError(
            f"PDF has only {total_chars} extractable chars "
            f"(min {_MIN_TOTAL_CHARS}); likely image-only"
        )

    return Paper(
        filename=filename or path.name,
        sha256=_sha256_file(path),
        n_pages=n_pages or 1,
        n_chars=total_chars,
        sections=sections,
        equations=equations,
        tables=tables,
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_pages(
    doc: pymupdf.Document,
) -> tuple[list[PaperSection], list[Equation], list[Table], int]:
    """Walk every page, collect sections / equations / tables.

    Implementation notes
    --------------------
    ``page.get_text("dict")`` returns a tree of blocks → lines → spans.
    Each span carries ``font`` (font name), ``size`` (pt), ``flags``,
    and the literal text. We:

      * establish the "body" font size as the size that carries the
        most characters in the document;
      * start a new :class:`PaperSection` at a line that opens in a
        font of at least ``body × _HEADING_FONT_RATIO`` AND is short
        (< ``_HEADING_MAX_CHARS``) — see :func:`_is_heading_line`.
        Consecutive heading lines merge into one title;
      * accumulate body lines into the current section's ``text``
        buffer, tracking the running char offset;
      * record every page break so we can answer
        "what page is char offset N on?" without re-scanning.
    """
    # Body size = the font size carrying the most characters. ``min``
    # breaks on real papers: a 5pt subscript or footnote mark puts the
    # threshold below the 10pt body, turning every line into a heading.
    # Weighting by characters (not spans) keeps short heading spans from
    # outvoting the body even on heading-heavy pages.
    chars_by_size: Counter[float] = Counter()
    for page in doc:  # type: ignore[attr-defined]
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:  # 0=text, 1=image
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    chars_by_size[round(span["size"], 1)] += len(span["text"].strip())

    if not chars_by_size:
        return [], [], [], 0

    body_size = chars_by_size.most_common(1)[0][0]
    heading_threshold = body_size * _HEADING_FONT_RATIO

    sections: list[PaperSection] = []
    equations: list[Equation] = []
    tables: list[Table] = []
    total_chars = 0

    current_title = ""
    current_page = 1
    current_text_parts: list[str] = []
    current_start = 0

    def flush_section(end_char: int) -> None:
        if not current_text_parts:
            return
        sections.append(
            PaperSection(
                title=current_title,
                page=current_page,
                char_start=current_start,
                char_end=end_char,
                text="".join(current_text_parts),
            )
        )

    running_offset = 0
    for page in doc:  # type: ignore[attr-defined]
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 1:
                # Image block — Phase 2 ignores.
                continue
            for line in block["lines"]:
                spans = [span for span in line["spans"] if span["text"].strip()]
                if not spans:
                    continue
                line_text = "".join(span["text"] for span in line["spans"])
                if _is_heading_line(line, spans, line_text, heading_threshold):
                    if current_text_parts:
                        # Close the previous section, open a new one.
                        flush_section(running_offset)
                        current_title = line_text.strip()
                        current_page = page.number + 1
                        current_start = running_offset
                        current_text_parts = []
                    else:
                        # Nothing since the last heading line: this one
                        # continues it ("1" then "Introduction", or a
                        # title wrapped over two lines).
                        current_title = f"{current_title} {line_text.strip()}".strip()
                    # Don't include the heading text itself in the
                    # section body — Phase 6 verdicts cite the
                    # *body*, not the heading.
                    running_offset += len(line_text) + 1  # +1 for the joiner
                    continue
                current_text_parts.append(line_text + "\n")
                running_offset += len(line_text) + 1
                total_chars += len(line_text) + 1
            # Detect equations (display-math block heuristically —
            # centre-aligned, single line, no periods).
            _maybe_record_equation(line, page.number + 1, equations)
        # Detect tables (heuristic: block with >3 lines, all of similar
        # short length, no terminal punctuation).
        _maybe_record_table(block, page.number + 1, tables)

    flush_section(running_offset)
    return sections, equations, tables, total_chars


def _is_heading_line(line: dict, spans: list[dict], text: str, heading_threshold: float) -> bool:
    """A short horizontal line whose first span is set in heading size.

    Judged per line, not per span: small-caps headings ("3 V" at 12pt +
    "ARIATIONAL" at 9.6pt) start large and continue small. Rotated text
    (the arXiv side stamp) is never a heading.
    """
    if tuple(round(c) for c in line.get("dir", (1, 0))) != (1, 0):
        return False
    return spans[0]["size"] >= heading_threshold and len(text) < _HEADING_MAX_CHARS


def _maybe_record_equation(line: dict, page: int, out: list[Equation]) -> None:
    """Heuristic: lines containing ``=`` with no terminal period."""
    text = "".join(span["text"] for span in line["spans"]).strip()
    if "=" in text and not text.endswith(".") and len(text) > 3 and len(text) < 200:
        out.append(Equation(text=text, page=page))


def _maybe_record_table(block: dict, page: int, out: list[Table]) -> None:
    """Heuristic: blocks with many lines of similar short length."""
    if block["type"] != 0:
        return
    lines = block["lines"]
    if len(lines) < 4:
        return
    lengths = [sum(len(span["text"]) for span in line["spans"]) for line in lines]
    avg = sum(lengths) / len(lengths)
    if avg > 60:  # body text is much longer than table cells
        return
    out.append(
        Table(
            caption="",  # Phase 6 may extract captions via PyMuPDF's XML mode
            page=page,
            n_rows=len(lines),
            n_cols=_estimate_cols(lines),
        )
    )


def _estimate_cols(lines: list[dict]) -> int:
    """Cheap column estimate: max spans-per-line across the block."""
    return max(len(line["spans"]) for line in lines) if lines else 0
