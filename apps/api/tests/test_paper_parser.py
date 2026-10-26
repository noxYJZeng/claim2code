"""Unit tests for :mod:`claim2code_api.papers.parser`.

These tests build a synthetic PDF in-memory with PyMuPDF (no fixture
files on disk) so the suite is hermetic and CI-safe.

What we exercise
----------------
- A multi-page, multi-section paper returns the right section count
  and the right per-section page attribution.
- An image-only PDF triggers :class:`PdfNoTextLayerError`.
- A missing file raises :class:`FileNotFoundError`.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf  # type: ignore[import-untyped]
import pytest

from claim2code_api.papers.exceptions import PdfNoTextLayerError
from claim2code_api.papers.parser import parse_pdf


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Build a 3-page PDF with one heading per page and a long body.

    The body is intentionally long enough to dominate the font-size
    distribution so the median body size is well below the heading
    size — without enough body text, the median is pulled up by the
    headings and heading detection fails.
    """
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    # Body font 10pt, headings 16pt — well above the 1.5× threshold.
    body_font = ("helv", 10)
    head_font = ("helv", 16)

    long_body = (
        "This paper proposes a novel approach to scalable verification "
        "of paper claims against the corresponding implementation "
        "repositories. Our method combines structured-output language "
        "models with deterministic parsers for both sides of the "
        "equation: a PyMuPDF-based parser that produces sections, "
        "equations, and tables, and a tree-sitter-based parser that "
        "produces a symbol graph for the repository under review. "
        "We then embed every code symbol as a structured string "
        "representation, store the embeddings in a pgvector index, "
        "and retrieve the top-K candidate symbols for every claim via "
        "a hybrid ranker that combines semantic cosine similarity, "
        "BM25 lexical overlap on algorithm names, and a structural "
        "boost for symbols imported by the training-loop module. "
        "Finally, a verifier language model assigns each claim one of "
        "four statuses — VERIFIED, PARTIALLY_VERIFIED, "
        "CONTRADICTED, or UNVERIFIED — based on the paper evidence "
        "and the retrieved code evidence. The verifier is forbidden "
        "from using its own model knowledge as evidence."
    )

    for _page_num, heading in enumerate(["Introduction", "Method", "Experiments"]):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), heading, fontname=head_font[0], fontsize=head_font[1])
        page.insert_text((72, 110), long_body, fontname=body_font[0], fontsize=body_font[1])
    doc.save(str(path))
    doc.close()
    return path


def test_parse_returns_three_sections(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    assert paper.n_pages == 3
    assert len(paper.sections) == 3
    titles = [s.title for s in paper.sections]
    assert titles == ["Introduction", "Method", "Experiments"]


def test_parse_records_page_numbers(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    pages = [s.page for s in paper.sections]
    assert pages == [1, 2, 3]


def test_parse_records_total_chars(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    assert paper.n_chars > 100  # body + headings combined


def test_parse_computes_sha256(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    assert len(paper.sha256) == 64
    # sha256 must be stable across re-parses.
    paper2 = parse_pdf(sample_pdf)
    assert paper.sha256 == paper2.sha256


@pytest.fixture
def paper_like_pdf(tmp_path: Path) -> Path:
    """Real-paper typography: 10pt body, 12pt headings, 5pt subscripts.

    The section number sits on its own line above the title (as in
    ICLR / NeurIPS layouts) and an arXiv stamp runs up the margin.
    """
    path = tmp_path / "paper_like.pdf"
    doc = pymupdf.open()
    body = "We train with a learning rate of 0.001 and batch size 64. " * 6
    for heading_no, heading in [("1", "Introduction"), ("2", "Method")]:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60), heading_no, fontname="helv", fontsize=12)
        page.insert_text((72, 76), heading, fontname="helv", fontsize=12)
        page.insert_textbox(pymupdf.Rect(72, 90, 520, 400), body, fontname="helv", fontsize=10)
        page.insert_text((300, 420), "t", fontname="helv", fontsize=5)
        page.insert_text((20, 600), "arXiv:2010.02502v4", fontname="helv", fontsize=20, rotate=90)
    doc.save(str(path))
    doc.close()
    return path


def test_parse_real_typography_keeps_body_out_of_headings(paper_like_pdf: Path) -> None:
    paper = parse_pdf(paper_like_pdf)
    assert [s.title for s in paper.sections] == ["1 Introduction", "2 Method"]
    for section in paper.sections:
        assert "learning rate of 0.001" in section.text
    assert paper.n_pages == 2


def test_parse_rejects_image_only_pdf(tmp_path: Path) -> None:
    """A PDF with only an image (no text layer) must raise."""
    path = tmp_path / "image_only.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(
        page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100))
    )
    doc.save(str(path))
    doc.close()

    with pytest.raises(PdfNoTextLayerError):
        parse_pdf(path)


def test_parse_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "does_not_exist.pdf")


def test_parse_uses_filename_kwarg(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf, filename="custom_name.pdf")
    assert paper.filename == "custom_name.pdf"
