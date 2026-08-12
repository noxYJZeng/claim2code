"""Deterministic ``HyperparameterChecker``.

Cross-checks hyperparameter values mentioned in the paper against
values referenced in the code. The check is purely regex + numeric
comparison — no LLM involvement — so its output is reproducible
and easy to unit-test.

Taxonomy
--------

Seven hyperparameter names are recognised (case-insensitive,
each with a small set of aliases):

  Multiplicative regime (``ratio > 10× → mismatch``):

    - ``learning_rate | lr | η``
    - ``temperature | τ | tau``
    - ``beta | β``
    - ``gamma | γ``

  Additive regime (``relative diff > 5% → mismatch``):

    - ``batch_size``
    - ``num_epochs | epochs``
    - ``hidden_size | d_model | d``

Why two regimes? Learning rates / temperatures / mixing
coefficients vary by orders of magnitude across methods
(``1e-3`` Adam vs ``1e-5`` fine-tuning); the *log ratio*
matters more than the absolute diff. Batch sizes / epochs /
hidden dims are integer-ish counts and a 5% relative diff is
the right boundary.

Why these 7? They are the names enumerated in
``docs/architecture/phase-roadmap.md:122``. Adding a new name
is a one-line edit to ``_NAME_ALIASES`` + a regime classification.

Extraction
----------

A single regex over each input text (paper quote + claim text +
each candidate's signature / docstring) extracts
``name = value`` pairs. The regex permits scientific notation
(``5e-5``, ``1.0E-3``) plus optional sign and exponent.

Comparison
----------

For every canonical name present on *both* sides, the numeric
values are compared under the regime's tolerance. A mismatch
emits one :class:`Discrepancy` with type
:attr:`DiscrepancyType.HYPERPARAMETER_MISMATCH` and severity
:attr:`DiscrepancySeverity.HIGH` (per
``docs/architecture/phase-6-design.md`` §2.7).

Asymmetric cases (paper mentions ``lr=5e-5`` but code has no
hyperparameter values; vice versa) are *not* emitted as
discrepancies — the LLM residual layer handles them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from claim2code_api.verification.schemas import (
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
)

# Multiplicative regime: ratios > 10× are considered mismatches.
# Used for hyperparameters whose values vary by orders of magnitude
# across methods (learning rates, temperatures, mixing coefficients).
_MULTIPLICATIVE_NAMES: frozenset[str] = frozenset(
    {
        "learning_rate", "lr", "η",
        "temperature", "τ", "tau",
        "beta", "β",
        "gamma", "γ",
    }
)

# Additive regime: relative diff > 5% is considered a mismatch.
# Used for count-like hyperparameters (batch sizes, epochs, dims).
_ADDITIVE_NAMES: frozenset[str] = frozenset(
    {
        "batch_size",
        "num_epochs", "epochs",
        "hidden_size", "d_model", "d",
    }
)

# Aliases → canonical name. Lowercased keys. Built at module import
# from the two regime sets above.
_NAME_ALIASES: dict[str, str] = {
    # Multiplicative
    "learning_rate": "learning_rate",
    "lr": "learning_rate",
    "η": "learning_rate",
    "temperature": "temperature",
    "τ": "temperature",
    "tau": "temperature",
    "beta": "beta",
    "β": "beta",
    "gamma": "gamma",
    "γ": "gamma",
    # Additive
    "batch_size": "batch_size",
    "num_epochs": "num_epochs",
    "epochs": "num_epochs",
    "hidden_size": "hidden_size",
    "d_model": "hidden_size",
    "d": "hidden_size",
}

# Build the master regex. Capture group 1 = alias, group 2 = value.
# We match ``<alias>``= ``<value>`` or ``<alias>``: ``<value>``, and
# also accept the Python signature form ``<alias>: <type> = <value>``
# (e.g. ``lr: float = 5e-5``). The optional ``\\w+\\s*=\\s*`` skip
# absorbs the type annotation that often appears between the colon
# and the default value.
_PATTERN: re.Pattern[str] = re.compile(
    r"(?P<name>"
    + "|".join(re.escape(a) for a in sorted(_NAME_ALIASES, key=len, reverse=True))
    + r")\s*[:=]\s*(?:\w+\s*=\s*)?(?P<value>[0-9.eE+-]+)",
    re.IGNORECASE,
)

# Tolerance constants — exposed at class level for unit tests to
# pin against if we ever want to tune.
_ADDITIVE_RELATIVE_TOLERANCE: float = 0.05
_MULTIPLICATIVE_RATIO_TOLERANCE: float = 10.0


@dataclass(frozen=True)
class HyperparameterChecker:
    """Deterministic hyperparameter comparison.

    Stateless. All methods are pure functions of the inputs.
    Construct once at module import and reuse — there is no
    per-instance state.
    """

    # Tolerance overrides (mainly for testing). Default to the
    # module-level constants.
    additive_relative_tolerance: float = _ADDITIVE_RELATIVE_TOLERANCE
    multiplicative_ratio_tolerance: float = _MULTIPLICATIVE_RATIO_TOLERANCE

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def extract(self, text: str) -> dict[str, float]:
        """Return ``{canonical_name: numeric_value}`` extracted from ``text``.

        If the same name appears multiple times, the *last*
        occurrence wins (matches the Python convention for
        keyword-argument overrides). Values that fail to parse
        as ``float`` are silently dropped — they will never
        trigger a numeric mismatch (the LLM can flag them in
        its own verdict if needed).
        """
        if not text:
            return {}

        out: dict[str, float] = {}
        for match in _PATTERN.finditer(text):
            alias = match.group("name").lower()
            value_str = match.group("value")
            try:
                value = float(value_str)
            except ValueError:
                continue
            canonical = _NAME_ALIASES[alias]
            out[canonical] = value
        return out

    def compare(
        self,
        *,
        claim_text: str,
        paper_evidence_quote: str,
        candidates: list,  # list[RetrieveMatch] (forward-ref to avoid import cycle)
    ) -> list[Discrepancy]:
        """Cross-check paper-side vs. code-side hyperparameter values.

        Paper-side extraction concatenates ``paper_evidence_quote``
        with ``claim_text`` (the quote is preferred, but claim
        text may contain values the quote omits).

        Code-side extraction concatenates each candidate's
        ``signature`` and ``docstring`` — the verifier does NOT
        have access to raw source code (``CodeSymbol.source`` is
        not populated today; see phase-6-design.md §"Hard
        constraints discovered in recon").

        For each canonical name present on BOTH sides, compare
        values under the regime's tolerance. Mismatches emit
        ``HYPERPARAMETER_MISMATCH`` (HIGH).
        """
        paper_text = "\n".join(
            t for t in (paper_evidence_quote, claim_text) if t
        )
        paper_values = self.extract(paper_text)

        code_text_parts: list[str] = []
        for cand in candidates:
            if cand.signature:
                code_text_parts.append(cand.signature)
            if cand.docstring:
                code_text_parts.append(cand.docstring)
        code_values = self.extract("\n".join(code_text_parts))

        # Intersection: only emit when both sides have the name.
        common_names = set(paper_values) & set(code_values)
        if not common_names:
            return []

        out: list[Discrepancy] = []
        for name in sorted(common_names):
            p_val = paper_values[name]
            c_val = code_values[name]
            if self._values_match(name=name, paper_value=p_val, code_value=c_val):
                continue
            detail = (
                f"paper states {name}={self._format_value(name, p_val)}, "
                f"code references {name}={self._format_value(name, c_val)}"
            )
            out.append(
                Discrepancy(
                    type=DiscrepancyType.HYPERPARAMETER_MISMATCH,
                    severity=DiscrepancySeverity.HIGH,
                    detail=detail,
                )
            )
        return out

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    def _values_match(self, *, name: str, paper_value: float, code_value: float) -> bool:
        """Return True if the two values agree within tolerance."""
        if paper_value == code_value:
            return True
        canonical = self._canonical_of(name)
        if canonical in _MULTIPLICATIVE_NAMES:
            # Compare absolute values for ratios. Avoid div-by-zero.
            if code_value == 0 or paper_value == 0:
                return False
            ratio = max(abs(paper_value), abs(code_value)) / min(
                abs(paper_value), abs(code_value)
            )
            return ratio <= self.multiplicative_ratio_tolerance
        # Additive regime: relative diff.
        denom = max(abs(paper_value), abs(code_value))
        if denom == 0:
            return paper_value == code_value
        rel_diff = abs(paper_value - code_value) / denom
        return rel_diff <= self.additive_relative_tolerance

    @staticmethod
    def _canonical_of(alias_or_canonical: str) -> str:
        """Map any alias (or canonical) to its canonical name."""
        return _NAME_ALIASES.get(alias_or_canonical.lower(), alias_or_canonical)

    @staticmethod
    def _format_value(name: str, value: float) -> str:
        """Format a hyperparameter value for human-readable detail strings.

        Integer-ish counts (``batch_size`` / ``num_epochs`` /
        ``hidden_size``) render without trailing decimals; other
        values render in fixed / scientific notation as
        appropriate to preserve precision (``5e-5`` doesn't lose
        its precision via ``0.00005``).
        """
        canonical = HyperparameterChecker._canonical_of(name)
        if canonical in {"batch_size", "num_epochs", "hidden_size"}:
            if value == int(value):
                return str(int(value))
        return f"{value:g}"