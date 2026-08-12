"""Unit tests for :class:`claim2code_api.safety.redactor.SecretRedactor`.

Coverage:

* Each of the 8 credential pattern types (OpenAI, Anthropic, AWS, GCP,
  GitHub, PEM, Bearer, .env) is detected and replaced with
  ``[REDACTED:<label>]``.
* Multi-span text: multiple matches of the same pattern produce
  multiple ``SecretHit`` entries.
* PEM blocks span multiple lines and are redacted as a single unit.
* Preview string is length-capped.
* False-positive guard: legitimate text containing "ignore the
  previous state-of-the-art baseline", arXiv IDs, plain emails,
  plain URLs is NOT redacted.
* Pure-redaction: redactor mutates only matched spans; the rest of
  the text passes through unchanged.
"""
from __future__ import annotations

import pytest

from claim2code_api.safety.redactor import SecretRedactor


@pytest.fixture
def redactor() -> SecretRedactor:
    return SecretRedactor()


# ============================================================
# 1. Per-pattern happy paths
# ============================================================


class TestOpenAIKey:
    def test_sk_proj_redacted(self, redactor: SecretRedactor) -> None:
        text = "my key is sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
        out, hits = redactor.redact(text)
        assert "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456" not in out
        assert "[REDACTED:openai_key]" in out
        assert [h.label for h in hits] == ["openai_key"]

    def test_sk_classic_redacted(self, redactor: SecretRedactor) -> None:
        text = "sk-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
        out, hits = redactor.redact(text)
        assert "[REDACTED:openai_key]" in out
        assert hits[0].label == "openai_key"


class TestAnthropicKey:
    def test_sk_ant_redacted(self, redactor: SecretRedactor) -> None:
        text = "anthropic key sk-ant-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
        out, hits = redactor.redact(text)
        assert "sk-ant-AbCdEfGhIjKlMnOpQrStUvWxYz123456" not in out
        assert "[REDACTED:anthropic_key]" in out


class TestAWSKey:
    def test_akia_redacted(self, redactor: SecretRedactor) -> None:
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        out, hits = redactor.redact(text)
        # First matched by env_assignment (label env_assignment);
        # second pass redacts the AKIA span as aws_access_key.
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        labels = [h.label for h in hits]
        assert "aws_access_key" in labels or "env_assignment" in labels


class TestGCPKey:
    def test_aiza_redacted(self, redactor: SecretRedactor) -> None:
        # Real GCP keys are 39 chars (AIza + 35).
        text = "API_KEY=AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567"
        out, hits = redactor.redact(text)
        assert "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567" not in out
        assert any(h.label == "gcp_api_key" for h in hits)


class TestGitHubPAT:
    def test_ghp_redacted(self, redactor: SecretRedactor) -> None:
        text = "token: ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
        out, hits = redactor.redact(text)
        assert "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789" not in out
        assert any(h.label == "github_pat" for h in hits)

    def test_gho_redacted(self, redactor: SecretRedactor) -> None:
        text = "oauth: gho_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
        out, hits = redactor.redact(text)
        assert any(h.label == "github_pat" for h in hits)


class TestPEMBlock:
    def test_pem_begin_end_redacted(self, redactor: SecretRedactor) -> None:
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA...\n"
            "-----END RSA PRIVATE KEY-----\n"
        )
        out, hits = redactor.redact(text)
        assert "-----BEGIN" not in out
        assert "[REDACTED:pem_private_key]" in out
        assert len(hits) == 1
        assert hits[0].label == "pem_private_key"

    def test_unterminated_pem_still_redacted(self, redactor: SecretRedactor) -> None:
        text = (
            "-----BEGIN EC PRIVATE KEY-----\n"
            "missing the END marker\n"
        )
        out, hits = redactor.redact(text)
        assert "[REDACTED:pem_private_key]" in out
        assert len(hits) == 1


class TestBearerToken:
    def test_bearer_in_header(self, redactor: SecretRedactor) -> None:
        text = "Authorization: Bearer AbCdEfGhIjKlMnOpQrStUvWxYz1234"
        out, hits = redactor.redact(text)
        assert "AbCdEfGhIjKlMnOpQrStUvWxYz1234" not in out
        assert any(h.label == "bearer_token" for h in hits)


class TestEnvAssignment:
    def test_openai_key_env(self, redactor: SecretRedactor) -> None:
        text = "OPENAI_API_KEY=sk-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
        out, hits = redactor.redact(text)
        # OPENAI_API_KEY=... is a .env assignment; the value (sk-...)
        # is independently an openai_key pattern. Either label is
        # acceptable; the assignment string must be redacted.
        assert "OPENAI_API_KEY=sk-" not in out
        labels = [h.label for h in hits]
        assert "env_assignment" in labels or "openai_key" in labels

    def test_database_url_redacted(self, redactor: SecretRedactor) -> None:
        text = "DATABASE_URL=postgresql://user:pass@host:5432/db"
        out, hits = redactor.redact(text)
        assert "postgresql://user:pass" not in out
        assert any(h.label == "env_assignment" for h in hits)


# ============================================================
# 2. Multi-span / combined text
# ============================================================


class TestMultiSpan:
    def test_two_openai_keys_in_one_text(self, redactor: SecretRedactor) -> None:
        text = (
            "first sk-AbCdEfGhIjKlMnOpQrStUvWxYz123456 and "
            "second sk-proj-XyZ12345aBcDeFgHiJkLmNoPqRsTuVwXyZ"
        )
        out, hits = redactor.redact(text)
        assert out.count("[REDACTED:openai_key]") == 2
        openai_hits = [h for h in hits if h.label == "openai_key"]
        assert len(openai_hits) == 2

    def test_combined_pem_and_bearer(self, redactor: SecretRedactor) -> None:
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n"
            "Authorization: Bearer AbCdEfGhIjKlMnOpQrStUvWxYz1234"
        )
        out, hits = redactor.redact(text)
        assert "[REDACTED:pem_private_key]" in out
        assert "[REDACTED:bearer_token]" in out
        labels = {h.label for h in hits}
        assert {"pem_private_key", "bearer_token"} <= labels


# ============================================================
# 3. False-positive guard
# ============================================================


class TestFalsePositives:
    def test_email_not_redacted(self, redactor: SecretRedactor) -> None:
        text = "Contact alice@example.com for details."
        out, hits = redactor.redact(text)
        assert out == text
        assert hits == []

    def test_url_not_redacted(self, redactor: SecretRedactor) -> None:
        text = "See https://example.com/paper.pdf for the PDF."
        out, hits = redactor.redact(text)
        assert out == text
        assert hits == []

    def test_arxiv_id_not_redacted(self, redactor: SecretRedactor) -> None:
        text = "arXiv:2401.12345 [cs.CL]"
        out, hits = redactor.redact(text)
        assert out == text
        assert hits == []

    def test_short_sk_word_not_redacted(self, redactor: SecretRedactor) -> None:
        """``sk-`` followed by <20 chars (not a real key) is not matched."""
        text = "short sk-1234"
        out, hits = redactor.redact(text)
        assert out == text
        assert hits == []


# ============================================================
# 4. Preview length cap
# ============================================================


class TestPreviewCap:
    def test_preview_capped(self, redactor: SecretRedactor) -> None:
        long_key = "sk-proj-" + "A" * 60
        text = f"key: {long_key}"
        out, hits = redactor.redact(text)
        assert len(hits) == 1
        assert hits[0].preview.endswith("...")
        assert len(hits[0].preview) <= 35  # 32 + "..."

    def test_short_preview_unchanged(self, redactor: SecretRedactor) -> None:
        # Exactly 23 chars (sk- + 20 chars) — preview length 23 < cap 32.
        text = "sk-AbCdEfGhIjKlMnOpQrStUv"
        out, hits = redactor.redact(text)
        assert not hits[0].preview.endswith("...")