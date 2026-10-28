"""Rendering the cited part of a paper (papers/excerpt.py + routes)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pymupdf  # type: ignore[import-untyped]
import pytest
from fastapi import HTTPException

from claim2code_api.papers import router as router_mod
from claim2code_api.papers.excerpt import render_excerpt


@pytest.fixture
def pdf(tmp_path):
    """Three pages; the cited sentence sits on page 2."""
    path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    texts = [
        "Introduction. Diffusion models are latent variable models.",
        "We set the number of timesteps to one thousand for all experiments.\n"
        "The variances increase linearly from small to large values.",
        "Appendix. Training details follow below in this section.",
    ]
    for text in texts:
        page = doc.new_page()
        page.insert_text((72, 100), text, fontsize=11)
    doc.save(path)
    return path


def _png_ok(data: bytes) -> bool:
    return data.startswith(b"\x89PNG")


def test_finds_and_crops_around_the_quote(pdf) -> None:
    out = render_excerpt(pdf, page=2, quote="the number of timesteps to one thousand")
    assert (out.page, out.n_pages, out.found) == (2, 3, True)
    assert _png_ok(out.png)
    full = render_excerpt(pdf, page=2, quote="the number of timesteps", full_page=True)
    assert full.height > out.height  # the crop is a region, not the page


def test_corrects_an_off_by_one_page(pdf) -> None:
    out = render_excerpt(pdf, page=1, quote="The variances increase linearly from small to large values.")
    assert (out.page, out.found) == (2, True)


def test_matches_a_quote_in_pieces(pdf) -> None:
    # The LLM's copy differs mid-sentence; the windows still match.
    quote = "We set the number of timesteps to one thousand (T = 1000) for every experiment"
    out = render_excerpt(pdf, page=2, quote=quote)
    assert out.found


def test_unfound_quote_shows_the_cited_page_unmarked(pdf) -> None:
    out = render_excerpt(pdf, page=3, quote="zzzz qqqq xxxx not in the paper anywhere")
    assert (out.page, out.found) == (3, False)
    assert _png_ok(out.png)


def _db(row):
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = row
    return db


def test_excerpt_route_returns_a_png_data_url(pdf, monkeypatch) -> None:
    monkeypatch.setattr(router_mod.storage, "path_for", lambda _sha: pdf)
    row = SimpleNamespace(sha256="abc", filename="paper.pdf")
    out = router_mod.get_paper_excerpt(
        analysis_id=uuid4(), page=2, quote="number of timesteps", full=False, db=_db(row),
    )
    assert out.found and out.page == 2
    assert out.image.startswith("data:image/png;base64,")


def test_pdf_routes_404_without_a_paper_or_file(tmp_path, monkeypatch) -> None:
    with pytest.raises(HTTPException) as excinfo:
        router_mod.get_paper_pdf(analysis_id=uuid4(), db=_db(None))
    assert excinfo.value.status_code == 404

    monkeypatch.setattr(router_mod.storage, "path_for", lambda _sha: tmp_path / "gone.pdf")
    with pytest.raises(HTTPException) as excinfo:
        router_mod.get_paper_excerpt(
            analysis_id=uuid4(), page=1, quote="x", full=False,
            db=_db(SimpleNamespace(sha256="abc", filename="p.pdf")),
        )
    assert excinfo.value.status_code == 404
