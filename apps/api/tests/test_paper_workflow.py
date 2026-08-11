"""Unit tests for :mod:`claim2code_api.papers.workflow`.

The workflow is the trickiest piece in Phase 2 because it touches
the DB, the disk, and the LLM in one synchronous request. We
mock the DB session (with a MagicMock) and the LLM (via the
``extract_claims_from_paper`` seam) so the test stays CI-safe
while still exercising the orchestration logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pymupdf  # type: ignore[import-untyped]
import pytest

from claim2code_api.papers import workflow
from claim2code_api.papers.exceptions import (
    ClaimExtractionError,
    PdfInvalidError,
    PdfTooLargeError,
)
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence
from claim2code_api.papers.workflow import (
    _validate_upload,
    run_paper_workflow,
)

# ============================================================
# Helpers
# ============================================================


def _make_pdf(tmp_path: Path, body: str | None = None) -> Path:
    """Build a tiny 1-page PDF with enough text to clear the
    ``_MIN_TOTAL_CHARS`` floor (200 chars).

    ``insert_text`` wraps a long body to one line and clips at the
    page edge; we use ``insert_textbox`` with an explicit rect so
    the text actually lands in the document. We also keep the body
    size small (10pt) and repeat the sentence several times so the
    floor is comfortably cleared.
    """
    if body is None:
        body = (
            "The proposed method achieves state-of-the-art results on the "
            "benchmark by combining contrastive learning with structured "
            "retrieval and verification, with no parameter updates at "
            "inference time, yielding robust gains across nine datasets "
            "that span language understanding, code generation, and "
            "structured reasoning. "
            "The model uses beta equals 0.1 as the regularization "
            "coefficient, trained for three epochs on a curated dataset "
            "of approximately fifty thousand examples, with a learning "
            "rate of 1e-4 and a batch size of thirty-two. "
        )
    path = tmp_path / "tiny.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Title", fontname="helv", fontsize=16)
    # Use insert_textbox so the body wraps within the page rect.
    rect = pymupdf.Rect(72, 110, 523, 800)
    page.insert_textbox(rect, body, fontname="helv", fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def _sample_claim() -> Claim:
    return Claim(
        claim_id="c1",
        text="Method uses beta=0.1.",
        claim_type=ClaimType.HYPERPARAMETER,
        confidence=0.9,
        paper_evidence=PaperEvidence(
            section_title="Method",
            page=1,
            char_start=0,
            char_end=10,
            quote="beta=0.1",
        ),
    )


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    return _make_pdf(tmp_path)


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    """A pre-created uploads dir that ``storage.upload_dir()`` returns."""
    d = tmp_path / "uploads"
    d.mkdir()
    return d


# ============================================================
# _validate_upload
# ============================================================


def test_validate_upload_accepts_valid_bytes() -> None:
    _validate_upload(b"%PDF-1.4 content", max_size_bytes=1024)


def test_validate_upload_rejects_oversize() -> None:
    with pytest.raises(PdfTooLargeError):
        _validate_upload(b"%PDF-x" * 200, max_size_bytes=100)


def test_validate_upload_rejects_bad_magic_bytes() -> None:
    with pytest.raises(PdfInvalidError):
        _validate_upload(b"NOT-A-PDF", max_size_bytes=1024)


# ============================================================
# run_paper_workflow — happy path
# ============================================================


def test_workflow_happy_path_persists_claims(
    upload_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full pipeline runs end-to-end with mocks: writes disk,
    parses PDF, calls LLM, persists claims, returns summary."""
    monkeypatch.setattr(workflow.storage, "upload_dir", lambda: upload_dir)

    pdf_path = _make_pdf(upload_dir.parent)
    pdf_bytes = pdf_path.read_bytes()

    db = MagicMock()
    # First flush produces the analysis_id; subsequent flushes produce
    # the paper_id. We don't care about the specific UUIDs.
    flush_id = 0

    def fake_flush() -> None:
        nonlocal flush_id
        flush_id += 1
        # Pre-load analysis_id and paper_id so the dataclass field access
        # in WorkflowResult doesn't trip.
        if flush_id == 1:
            db._analysis_id = "11111111-1111-1111-1111-111111111111"
        elif flush_id == 2:
            db._paper_id = "22222222-2222-2222-2222-222222222222"

    db.flush.side_effect = fake_flush

    # db.add is called for AnalysisState and Paper rows; we let MagicMock
    # silently accept the calls. We need to expose ``analysis_id`` and
    # ``paper_id`` on the just-added rows so the workflow can read them
    # back after flush.
    added_objects: list[MagicMock] = []

    def fake_add(obj: MagicMock) -> None:
        added_objects.append(obj)

    db.add.side_effect = fake_add

    # After flush, the workflow reads ``.analysis_id`` / ``.paper_id``
    # off the just-added ORM rows. We assign those in fake_flush.
    with patch.object(workflow, "extract_claims_from_paper") as fake_extract:
        fake_extract.return_value = [_sample_claim()]

        result = run_paper_workflow(
            db=db,
            llm=MagicMock(name="fake_llm"),
            upload_bytes=pdf_bytes,
            filename="tiny.pdf",
            max_size_bytes=10 * 1024 * 1024,
        )

    assert result.parse_status == "done"
    assert result.n_claims == 1
    db.commit.assert_called()


def test_workflow_raises_when_larger_than_cap(tmp_path: Path) -> None:
    """No DB writes happen before the size check."""
    db = MagicMock()
    with pytest.raises(PdfTooLargeError):
        run_paper_workflow(
            db=db,
            llm=MagicMock(),
            upload_bytes=b"%PDF-1.4" + b"x" * 10_000,
            filename="x.pdf",
            max_size_bytes=100,
        )
    db.add.assert_not_called()


def test_workflow_raises_when_magic_bytes_bad(tmp_path: Path) -> None:
    db = MagicMock()
    with pytest.raises(PdfInvalidError):
        run_paper_workflow(
            db=db,
            llm=MagicMock(),
            upload_bytes=b"NOT-A-PDF",
            filename="x.pdf",
            max_size_bytes=1024,
        )


def test_workflow_records_error_on_extraction_failure(
    upload_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When LLM.extract_claims raises, the workflow rolls back,
    records the error in AnalysisState, and re-raises."""
    monkeypatch.setattr(workflow.storage, "upload_dir", lambda: upload_dir)
    pdf_path = _make_pdf(upload_dir.parent)
    db = MagicMock()

    # Track analysis_id and paper_id across flushes. TypedDict so mypy
    # narrows each key's value type (a plain dict[str, int | str] would
    # still leave ``state["flushes"]`` widened to ``int | str``).
    class _FlushState(TypedDict):
        flushes: int
        analysis_id: str
        paper_id: str

    state: _FlushState = {"flushes": 0, "analysis_id": "a", "paper_id": "b"}

    def fake_flush() -> None:
        state["flushes"] += 1

    db.flush.side_effect = fake_flush

    def fake_add(obj: MagicMock) -> None:
        # Make the added rows expose the IDs the workflow will read.
        if state["flushes"] == 0:  # first add: AnalysisState
            obj.analysis_id = state["analysis_id"]
        elif state["flushes"] == 1:  # second add: Paper
            obj.paper_id = state["paper_id"]

    db.add.side_effect = fake_add

    with patch.object(workflow, "extract_claims_from_paper") as fake_extract:
        fake_extract.side_effect = ClaimExtractionError("llm down")
        with pytest.raises(ClaimExtractionError):
            run_paper_workflow(
                db=db,
                llm=MagicMock(),
                upload_bytes=pdf_path.read_bytes(),
                filename="tiny.pdf",
                max_size_bytes=10 * 1024 * 1024,
            )

    # The workflow must roll back + record the error.
    db.rollback.assert_called()


# ============================================================
# run_paper_workflow — duplicate upload
# ============================================================


def test_workflow_maps_duplicate_sha256_to_pdf_invalid(
    upload_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second upload with the same sha256 raises PdfInvalidError."""
    monkeypatch.setattr(workflow.storage, "upload_dir", lambda: upload_dir)
    # We can't predict the sha256 of arbitrary bytes here without
    # recomputing; instead we patch write_pdf itself.
    monkeypatch.setattr(
        workflow.storage, "write_pdf", MagicMock(side_effect=FileExistsError("dup"))
    )
    db = MagicMock()

    with pytest.raises(PdfInvalidError, match="already uploaded"):
        run_paper_workflow(
            db=db,
            llm=MagicMock(),
            upload_bytes=b"%PDF-1.4\n",
            filename="x.pdf",
            max_size_bytes=1024,
        )
