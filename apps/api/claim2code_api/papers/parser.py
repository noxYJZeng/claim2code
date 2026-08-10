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

from pathlib import Path

import pymupdf  # type: ignore[import-untyped]  # pymupdf ships no type stubs

from claim2code_api.papers.exceptions import PdfNoTextLayerError
from claim2code_api.papers.schemas import Equation, Paper, PaperSection, Table

# Minimum total text length (across all pages) below which we consider
# the PDF to be image-only / scanned. 200 chars ≈ two paragraphs of
# body text; real papers produce ~50k chars so this is a conservative
# floor.
_MIN_TOTAL_CHARS = 200

# A paragraph whose font is this much larger than the median body font
# is treated as a heading. 1.5x is the empirically-tuned sweet spot
# for arXiv-style typography where body ≈ 10pt and headings are 12-16pt.
_HEADING_FONT_RATIO = 1.5


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
        n_pages=len(sections) and max(s.page for s in sections) or 1,
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

      * establish the "body" font size as the **minimum** span size in
        the document — headings are by definition larger than body
        text, so the smallest span is body (possibly captions /
        footnotes / subscripts — all smaller than headings). ``min``
        is more robust than median against synthetic PDFs or
        heading-heavy pages where the median is pulled into
        heading size;
      * start a new :class:`PaperSection` when we see a span whose
        font size exceeds ``body × _HEADING_FONT_RATIO`` AND its text
        is short (< 200 chars) — a long "heading" is almost always
        body text;
      * accumulate contiguous spans into the current section's
        ``text`` buffer, tracking the running char offset;
      * record every page break so we can answer
        "what page is char offset N on?" without re-scanning.
    """
    body_sizes: list[float] = []
    for page in doc:  # type: ignore[attr-defined]
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:  # 0=text, 1=image
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    body_sizes.append(span["size"])

    if not body_sizes:
        return [], [], [], 0

    # Use the smallest font size as the body baseline. Headings are by
    # definition larger than body text, so the smallest span is body
    # (possibly captions / footnotes / subscripts — all smaller than
    # headings). Using ``min`` instead of median sidesteps the
    # synthetic-PDF edge case where the number of heading spans equals
    # or exceeds body spans and pulls the median into heading size.
    body_size = min(body_sizes)
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
                for span in line["spans"]:
                    text = span["text"]
                    if not text.strip():
                        continue
                    size = span["size"]
                    is_heading = size >= heading_threshold and len(text) < 200
                    if is_heading:
                        # Close the previous section, open a new one.
                        flush_section(running_offset)
                        current_title = text.strip()
                        current_page = page.number + 1
                        current_start = running_offset
                        current_text_parts = []
                        # Don't include the heading text itself in the
                        # section body — Phase 6 verdicts cite the
                        # *body*, not the heading.
                        running_offset += len(text) + 1  # +1 for the joiner
                        continue
                    current_text_parts.append(text)
                    running_offset += len(text)
                    total_chars += len(text)
                current_text_parts.append("\n")
                running_offset += 1
                total_chars += 1
            # Detect equations (display-math block heuristically —
            # centre-aligned, single line, no periods).
            _maybe_record_equation(line, page.number + 1, equations)
        # Detect tables (heuristic: block with >3 lines, all of similar
        # short length, no terminal punctuation).
        _maybe_record_table(block, page.number + 1, tables)

    flush_section(running_offset)
    return sections, equations, tables, total_chars


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
