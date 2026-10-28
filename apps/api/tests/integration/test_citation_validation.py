"""The HTTP verification path must persist only source-validated citations.

Real code cited under the wrong symbol or lines is moved to where it is;
invented code is dropped.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from claim2code_api.app.config import get_settings
from claim2code_api.app.deps import get_db
from claim2code_api.app.main import create_app
from claim2code_api.db.models import AnalysisState, CodeEmbedding, CodeSymbol, Repository
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence
from claim2code_api.verification import workflow
from claim2code_api.verification.schemas import CodeEvidence, Verification, VerificationStatus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="requires migrated PostgreSQL"),
]


@pytest.mark.parametrize("citation_kind", ["valid", "invented_quote", "unknown_symbol", "wrong_line"])
def test_http_persists_only_source_validated_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, citation_kind: str,
) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "model.py").write_text("def forward(x):\n    # preserve input\n    return x\n")
    claim = Claim(
        claim_id="returns-input", claim_type=ClaimType.METHOD,
        text="The forward function returns its input.", confidence=1.0,
        paper_evidence=PaperEvidence(
            section_title="Method", page=1, char_start=0, char_end=43,
            quote="The forward function returns its input.",
        ),
    )
    engine = create_engine(get_settings().database_url)
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                with Session(bind=connection, join_transaction_mode="create_savepoint") as db:
                    state = AnalysisState(claims=[claim.model_dump(mode="json")])
                    db.add(state)
                    db.flush()
                    repo = Repository(
                        analysis_id=state.analysis_id, url="https://github.com/test/citations",
                        owner="test", name=f"citations-{uuid4().hex}",
                        clone_path=str(clone), embedding_model="test-model",
                    )
                    db.add(repo)
                    db.flush()
                    symbol = CodeSymbol(
                        repository_id=repo.repository_id, kind="function", language="python",
                        file_path="model.py", start_line=1, start_col=0, end_line=3, end_col=12,
                        name="forward", signature="def forward(x):", docstring=None,
                    )
                    db.add(symbol)
                    db.flush()
                    vector = [1.0] + [0.0] * (get_settings().openai_embed_dim - 1)
                    db.add(CodeEmbedding(
                        symbol_id=symbol.symbol_id, repository_id=repo.repository_id,
                        analysis_id=state.analysis_id, embedding=vector, model_name="test-model",
                    ))
                    db.flush()
                    analysis_id = state.analysis_id
                    citation = CodeEvidence(
                        symbol_id=symbol.symbol_id, file_path="model.py",
                        start_line=3, end_line=3, quote="return x",
                    )
                    if citation_kind == "invented_quote":
                        citation.quote = "return invented_result"
                    elif citation_kind == "unknown_symbol":
                        citation.symbol_id = uuid4()
                    elif citation_kind == "wrong_line":
                        citation.start_line = citation.end_line = 1
                    llm = MagicMock()
                    llm.verdict_for_claim.return_value = Verification(
                        claim_id=claim.claim_id, status=VerificationStatus.VERIFIED,
                        confidence=0.99, explanation="Model-proposed verdict.",
                        paper_evidence=claim.paper_evidence, code_evidence=[citation],
                    )
                    embed = MagicMock()
                    embed.embed.side_effect = lambda texts: [vector for _ in texts]
                    monkeypatch.setattr(workflow, "get_llm_client", lambda: llm)
                    monkeypatch.setattr(workflow, "get_embedding_client", lambda: embed)
                    app = create_app()
                    app.dependency_overrides[get_db] = lambda: db
                    client = TestClient(app)
                    try:
                        response = client.post(f"/analyses/{analysis_id}/verifications")
                        assert response.status_code == 200, response.text
                        llm.verdict_for_claim.assert_called_once()
                        db.expire_all()
                        stored = client.get(f"/analyses/{analysis_id}/verifications")
                        assert stored.status_code == 200
                        verdict = stored.json()["verifications"][0]
                        if citation_kind != "invented_quote":
                            assert verdict["status"] == "VERIFIED"
                            [kept] = verdict["code_evidence"]
                            assert kept["quote"] == "return x"
                            assert (kept["start_line"], kept["end_line"]) == (3, 3)
                            assert kept["symbol_id"] == str(symbol.symbol_id)
                        else:
                            assert verdict["status"] == "UNVERIFIED"
                            assert verdict["code_evidence"] == []
                            assert verdict["confidence"] == 0.0
                    finally:
                        client.close()
            finally:
                transaction.rollback()
    finally:
        engine.dispose()
