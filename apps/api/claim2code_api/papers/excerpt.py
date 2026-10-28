"""Render the part of a paper a claim cites, with the quote highlighted.

A claim's ``paper_evidence`` carries the quote and a page number. The
quote is the LLM's copy of the paper text, so it can differ from the
PDF in whitespace, hyphenation, ligatures or math glyphs, and the page
can be off by one. :func:`render_excerpt` therefore:

  1. searches for the whole quote, then for overlapping word windows
     of it, so a quote that matches only in pieces still lights up;
  2. tries the cited page first, then its neighbours, then the rest;
  3. renders the region around the hits (or the whole page) to PNG
     with the hits highlighted.

The PDF is opened read-only per call; highlights go on that in-memory
copy and are never saved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # type: ignore[import-untyped]  # pymupdf ships no type stubs

# Words per search window when the whole quote is not found verbatim.
_WINDOW = 6
# Points of page shown above and below the highlighted lines.
_MARGIN = 90.0
_ZOOM = 2.0
_HIGHLIGHT = (1.0, 0.85, 0.2)


@dataclass(frozen=True)
class PaperExcerpt:
    page: int  # 1-based page the image shows
    n_pages: int
    found: bool  # whether any part of the quote was located
    png: bytes
    width: int
    height: int


def render_excerpt(pdf_path: Path, *, page: int, quote: str, full_page: bool = False) -> PaperExcerpt:
    """Render the cited region of ``pdf_path``.

    :param page: the cited 1-based page (clamped to the document).
    :param full_page: render the whole page instead of the region
        around the hits.
    """
    with pymupdf.open(pdf_path) as doc:
        n_pages = doc.page_count
        cited = min(max(page, 1), n_pages) - 1
        order = [cited, cited - 1, cited + 1] + [i for i in range(n_pages) if abs(i - cited) > 1]
        hits: list[pymupdf.Rect] = []
        index = cited
        for i in order:
            if not 0 <= i < n_pages:
                continue
            found, exact = _find(doc[i], quote)
            # Off the cited page, one short window could be a common
            # phrase; ask for the whole quote or two separate pieces.
            if found and (i == cited or exact or len(found) >= 2):
                hits, index = found, i
                break
        target = doc[index]
        for rect in hits:
            annot = target.add_highlight_annot(rect)
            annot.set_colors(stroke=_HIGHLIGHT)
            annot.update()
        clip = target.rect if full_page or not hits else _region(_text_area(target), hits)
        pix = target.get_pixmap(matrix=pymupdf.Matrix(_ZOOM, _ZOOM), clip=clip, annots=True)
        return PaperExcerpt(
            page=index + 1, n_pages=n_pages, found=bool(hits),
            png=pix.tobytes("png"), width=pix.width, height=pix.height,
        )


def _find(page: pymupdf.Page, quote: str) -> tuple[list[pymupdf.Rect], bool]:
    """Rects of the quote on ``page``, and whether it matched whole."""
    text = " ".join(quote.split())
    if not text:
        return [], False
    hits = page.search_for(text)
    if hits:
        return list(hits), True
    # Piecewise: overlapping windows, skipping ones that are mostly
    # symbols (math the text layer spells differently).
    words = text.split(" ")
    seen: list[pymupdf.Rect] = []
    step = max(1, _WINDOW // 2)
    for start in range(0, max(1, len(words) - _WINDOW + 1), step):
        window = " ".join(words[start:start + _WINDOW])
        if len(re.findall(r"[A-Za-z]{3,}", window)) < 2:
            continue
        for rect in page.search_for(window):
            if not any(rect.intersects(r) for r in seen):
                seen.append(rect)
    return seen, False


def _text_area(page: pymupdf.Page) -> pymupdf.Rect:
    """The page minus its blank margins (falls back to the full page)."""
    blocks = page.get_text("blocks")
    if not blocks:
        return page.rect
    pad = 8.0
    return pymupdf.Rect(
        max(page.rect.x0, min(b[0] for b in blocks) - pad), page.rect.y0,
        min(page.rect.x1, max(b[2] for b in blocks) + pad), page.rect.y1,
    )


def _region(page_rect: pymupdf.Rect, hits: list[pymupdf.Rect]) -> pymupdf.Rect:
    top = min(r.y0 for r in hits) - _MARGIN
    bottom = max(r.y1 for r in hits) + _MARGIN
    return pymupdf.Rect(page_rect.x0, max(page_rect.y0, top), page_rect.x1, min(page_rect.y1, bottom))


__all__ = ["PaperExcerpt", "render_excerpt"]
