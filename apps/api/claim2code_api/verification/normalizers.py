"""Name normalizers for model / dataset / metric mentions.

Three sibling classes share the same shape:

  - :class:`ModelNameNormalizer` — canonicalises model names
    (Qwen2.5-7B ↔ Qwen/Qwen2.5-7B). Emits
    :attr:`DiscrepancyType.CONFIGURATION_MISMATCH`.
  - :class:`DatasetNormalizer` — canonicalises dataset names
    (SQuAD ↔ squad ↔ Stanford Question Answering Dataset).
    Emits :attr:`DiscrepancyType.DATASET_MISMATCH`.
  - :class:`MetricNameComparator` — canonicalises metric names
    (F1 ↔ f1 ↔ F1-score). Emits
    :attr:`DiscrepancyType.METRIC_MISMATCH`.

All three load a static alias dict from
``verification/aliases/<thing>.json`` at import time. Each alias
JSON has the shape ``{"<canonical>": ["<variant1>", "<variant2>", ...]}``.
The normalizers build a ``variant -> canonical`` reverse lookup
at construction so :meth:`normalize` is O(1) per name.

Fuzzy match fallback
--------------------

When a name is not in any alias list, the normalizers fall back
to a trigram-ratio similarity check against every canonical key
via :class:`difflib.SequenceMatcher`. The threshold is
``FUZZY_THRESHOLD = 0.7`` (configurable per-instance). This
catches obvious typos (e.g. ``Qwen2-7B`` vs ``Qwen2.5-7B``) that
exact-string matching would miss.

The :class:`pg_trgm` trigram GIN index shipped in migration 0005
is a parallel fuzzy-match capability at the SQL layer; the
in-Python implementation here keeps the normalizer testable
without a database.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from claim2code_api.verification.exceptions import VerificationConfigError
from claim2code_api.verification.schemas import (
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
)

if TYPE_CHECKING:
    from claim2code_api.repos.schemas import RetrieveMatch

# Default fuzzy-match threshold; tunable per-instance.
FUZZY_THRESHOLD = 0.7

# Substring match for variant tokens. We use a non-anchored regex
# so partial matches work (e.g. ``Qwen/Qwen2.5-7B`` appears inside
# ``"we use the Qwen/Qwen2.5-7B model"``). Boundaries are
# conservative word-boundary chars to avoid matching the middle
# of larger identifiers.
_VARIANT_PATTERN_TEMPLATE = r"(?<![A-Za-z0-9]){variant}(?![A-Za-z0-9])"


# ============================================================
# Shared base
# ============================================================


@dataclass(frozen=True)
class _AliasNormalizer:
    """Common machinery shared by the three normalizer subclasses.

    Subclasses set ``_alias_filename`` and override
    ``_discrepancy_type``. The base class handles JSON loading,
    reverse-lookup construction, fuzzy fallback, and the
    ``compare(claim, candidates)`` skeleton that returns a list
    of :class:`Discrepancy`.
    """

    _alias_filename: ClassVar[str] = ""  # subclasses override
    _discrepancy_type: ClassVar[DiscrepancyType]  # subclasses override
    _default_severity: ClassVar[DiscrepancySeverity] = DiscrepancySeverity.LOW

    # Populated at construction:
    _canonical_to_variants: dict[str, list[str]] = field(default_factory=dict)
    _variant_to_canonical: dict[str, str] = field(default_factory=dict)
    # Task canonical -> the benchmark canonicals that contain it
    # (``mnli`` -> ``{"glue"}``), from the alias file's ``_members``.
    _member_of: dict[str, set[str]] = field(default_factory=dict)
    _fuzzy_threshold: float = FUZZY_THRESHOLD

    def __post_init__(self) -> None:
        aliases_path = (
            Path(__file__).resolve().parent / "aliases" / self._alias_filename
        )
        try:
            raw = json.loads(aliases_path.read_text())
        except FileNotFoundError as exc:
            raise VerificationConfigError(
                f"alias file missing: {aliases_path}",
            ) from exc
        except json.JSONDecodeError as exc:
            raise VerificationConfigError(
                f"alias file malformed: {aliases_path}: {exc}",
            ) from exc

        # Drop the optional ``_doc`` key; it's documentation only.
        canonical_to_variants = {
            k: v for k, v in raw.items() if not k.startswith("_")
        }
        if not canonical_to_variants:
            raise VerificationConfigError(
                f"alias file empty: {aliases_path}",
            )

        # Build reverse lookup. Variants are matched case-insensitively,
        # so we store the lowercased variant in the map.
        variant_to_canonical: dict[str, str] = {}
        for canonical, variants in canonical_to_variants.items():
            variant_to_canonical[canonical.lower()] = canonical
            for variant in variants:
                variant_to_canonical[variant.lower()] = canonical

        member_of: dict[str, set[str]] = {}
        for parent, members in raw.get("_members", {}).items():
            for member in members:
                member_of.setdefault(member, set()).add(parent)

        object.__setattr__(self, "_canonical_to_variants", canonical_to_variants)
        object.__setattr__(self, "_variant_to_canonical", variant_to_canonical)
        object.__setattr__(self, "_member_of", member_of)

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def normalize(self, name: str) -> str | None:
        """Return the canonical form for ``name``, or ``None`` if unknown.

        Lookup is case-insensitive against the alias table. If the
        name is not in the table and not within
        :attr:`_fuzzy_threshold` of any canonical, returns ``None``
        (the caller decides what to do — e.g. emit a discrepancy).
        """
        if not name:
            return None
        canonical = self._variant_to_canonical.get(name.lower())
        if canonical is not None:
            return canonical
        # Fuzzy fallback.
        best_ratio = 0.0
        best_canonical: str | None = None
        for candidate in self._canonical_to_variants:
            ratio = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_canonical = candidate
        if best_canonical is not None and best_ratio >= self._fuzzy_threshold:
            return best_canonical
        return None

    def extract_names(self, text: str) -> set[str]:
        """Return the set of canonical names mentioned in ``text``.

        Scans for each variant as a whole word (case-insensitive),
        collects the canonicals. Returns an empty set if the text
        mentions no known names.
        """
        if not text:
            return set()
        found: set[str] = set()
        for variant, canonical in self._variant_to_canonical.items():
            # Escape regex meta-chars in the variant (dots, slashes,
            # hyphens). ``re.escape`` handles all of them.
            pattern = _VARIANT_PATTERN_TEMPLATE.format(variant=re.escape(variant))
            if re.search(pattern, text, re.IGNORECASE):
                found.add(canonical)
        return found

    def compare(
        self,
        *,
        claim_text: str,
        candidates: list[RetrieveMatch],
    ) -> list[Discrepancy]:
        """Cross-check names mentioned in claim vs. in candidates.

        A paper name is covered when any candidate names it, or names
        one of its member tasks (code on ``mnli`` covers a claim about
        ``glue``). Each uncovered paper name emits one
        :class:`Discrepancy` of this normalizer's
        :attr:`_discrepancy_type`, listing what the code names instead.
        Candidates are a top-k mix, so a candidate naming some *other*
        model alongside one that names the claim's is not a mismatch.
        """
        paper_names = self.extract_names(claim_text)
        if not paper_names:
            return []

        code_text_parts: list[str] = []
        for cand in candidates:
            if cand.signature:
                code_text_parts.append(cand.signature)
            if cand.docstring:
                code_text_parts.append(cand.docstring)
        code_names = self.extract_names("\n".join(code_text_parts))

        # No comparison possible if the code side mentions nothing.
        if not code_names:
            return []

        covered = set(code_names)
        for name in code_names:
            covered |= self._member_of.get(name, set())

        out: list[Discrepancy] = []
        for paper_name in sorted(paper_names - covered):
            detail = (
                f"paper mentions '{paper_name}', "
                f"code references only {', '.join(repr(n) for n in sorted(code_names))}"
            )
            out.append(
                Discrepancy(
                    type=self._discrepancy_type,
                    severity=self._default_severity,
                    detail=detail,
                )
            )
        return out


# ============================================================
# Public normalizers
# ============================================================


@dataclass(frozen=True)
class ModelNameNormalizer(_AliasNormalizer):
    """Canonicalises pretrained-model names.

    Default severity is ``LOW`` because a paper / code mismatch
    on a known model is usually a labelling typo, not a true
    implementation gap. A future phase may tune this if real
    eval data shows misclassifications.
    """

    _alias_filename: ClassVar[str] = "models.json"
    _discrepancy_type: ClassVar[DiscrepancyType] = DiscrepancyType.CONFIGURATION_MISMATCH
    _default_severity: ClassVar[DiscrepancySeverity] = DiscrepancySeverity.LOW


@dataclass(frozen=True)
class DatasetNormalizer(_AliasNormalizer):
    """Canonicalises dataset names.

    Default severity is ``MEDIUM`` — a paper / code mismatch on
    the evaluation dataset is a more serious claim than a model
    naming typo, because it changes the comparison's validity.
    """

    _alias_filename: ClassVar[str] = "datasets.json"
    _discrepancy_type: ClassVar[DiscrepancyType] = DiscrepancyType.DATASET_MISMATCH
    _default_severity: ClassVar[DiscrepancySeverity] = DiscrepancySeverity.MEDIUM


@dataclass(frozen=True)
class MetricNameComparator(_AliasNormalizer):
    """Canonicalises metric names.

    Default severity is ``LOW`` — most metric-name mismatches
    (e.g. ``F1`` vs ``f1-score``) are aliases, not disagreements.
    """

    _alias_filename: ClassVar[str] = "metrics.json"
    _discrepancy_type: ClassVar[DiscrepancyType] = DiscrepancyType.METRIC_MISMATCH
    _default_severity: ClassVar[DiscrepancySeverity] = DiscrepancySeverity.LOW