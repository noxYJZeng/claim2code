import { useEffect, useState } from "react";
import { getClaims, type Claim, type ClaimsResponse } from "./api";

type ClaimsListProps = {
  analysisId: string;
};

// ClaimsList — Phase 2 claims view.
//
// Renders the extracted claims as a numbered list. Each entry shows
// the claim text, the paper passage it was taken from, its type,
// section, page and confidence, so the user can check the extraction
// against the paper. Verdicts live in <VerificationPanel/>.
export default function ClaimsList({ analysisId }: ClaimsListProps) {
  const [data, setData] = useState<ClaimsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getClaims(analysisId, controller.signal).then((result) => {
      if (result.ok) {
        setData(result.claims);
        setError(null);
      } else {
        setData(null);
        setError(result.error);
      }
    });
    return () => controller.abort();
  }, [analysisId]);

  if (error) {
    return (
      <p className="error-note" data-testid="claims-error">
        Failed to load claims: {error}
      </p>
    );
  }

  if (!data) {
    return (
      <p className="text-sm text-ink-3" data-testid="claims-loading">
        Loading claims…
      </p>
    );
  }

  if (data.claims.length === 0) {
    return (
      <p className="text-sm text-ink-2" data-testid="claims-empty">
        The paper parsed but no testable claims were extracted.
      </p>
    );
  }

  return (
    <section data-testid="claims-list">
      <p className="text-sm text-ink-2">
        {data.claims.length} claim{data.claims.length === 1 ? "" : "s"} extracted, each
        with the passage it was taken from.
      </p>
      <ol className="mt-6 border-t border-rule">
        {data.claims.map((claim) => (
          <ClaimCard key={claim.claim_id} claim={claim} />
        ))}
      </ol>
    </section>
  );
}

function ClaimCard({ claim }: { claim: Claim }) {
  const pct = Math.round(claim.confidence * 100);
  return (
    <li
      data-testid="claim-card"
      className="grid grid-cols-[3rem_minmax(0,1fr)] gap-x-4 border-b border-rule py-5 sm:grid-cols-[3.5rem_minmax(0,1fr)]"
    >
      <span className="pt-0.5 font-mono text-xs text-ink-3">{claim.claim_id}</span>
      <div className="min-w-0">
        <p className="font-serif text-[15px] leading-relaxed">{claim.text}</p>
        <blockquote className="mt-3 border-l-2 border-rule-strong pl-3 font-serif text-sm italic leading-relaxed text-ink-2">
          {claim.paper_evidence.quote}
        </blockquote>
        <p className="mt-2 text-xs text-ink-3">
          {claim.claim_type.toLowerCase()} · {claim.paper_evidence.section_title || "preamble"},
          p. {claim.paper_evidence.page} · confidence{" "}
          <span data-testid="claim-confidence">{pct}%</span>
        </p>
      </div>
    </li>
  );
}
