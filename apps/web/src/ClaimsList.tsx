import { useEffect, useState } from "react";
import { getClaims, type Claim, type ClaimsResponse } from "./api";

type ClaimsListProps = {
  analysisId: string;
};

// ClaimsList — Phase 2 claims view.
//
// Renders the extracted claims as cards. Each card shows the claim
// text, type, confidence, and the paper evidence (section title,
// page, quote) so the user can verify the LLM's citation matches
// the paper. Phase 6 will add a code-evidence side panel and a
// verdict badge.
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
      <section
        className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800"
        data-testid="claims-error"
      >
        Failed to load claims: {error}
      </section>
    );
  }

  if (!data) {
    return (
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-500"
        data-testid="claims-loading"
      >
        Loading claims…
      </section>
    );
  }

  if (data.claims.length === 0) {
    return (
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600"
        data-testid="claims-empty"
      >
        The paper parsed but no testable claims were extracted.
      </section>
    );
  }

  return (
    <section
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="claims-list"
    >
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-lg font-medium">Extracted claims</h2>
        <span className="text-sm text-slate-500">
          {data.claims.length} claim{data.claims.length === 1 ? "" : "s"}
        </span>
      </div>

      <ul className="space-y-3">
        {data.claims.map((claim) => (
          <ClaimCard key={claim.claim_id} claim={claim} />
        ))}
      </ul>
    </section>
  );
}

function ClaimCard({ claim }: { claim: Claim }) {
  const pct = Math.round(claim.confidence * 100);
  return (
    <li
      data-testid="claim-card"
      className="rounded-xl border border-slate-200 bg-slate-50 p-4"
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-medium text-slate-900">{claim.text}</p>
        <span className="shrink-0 rounded-full bg-slate-900 px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-white">
          {claim.claim_type}
        </span>
      </div>
      <dl className="mt-3 grid grid-cols-1 gap-1 text-xs text-slate-600 sm:grid-cols-3">
        <div>
          <dt className="font-medium uppercase tracking-wide text-slate-500">
            Confidence
          </dt>
          <dd data-testid="claim-confidence">{pct}%</dd>
        </div>
        <div>
          <dt className="font-medium uppercase tracking-wide text-slate-500">
            Section
          </dt>
          <dd>{claim.paper_evidence.section_title || "(preamble)"}</dd>
        </div>
        <div>
          <dt className="font-medium uppercase tracking-wide text-slate-500">
            Page
          </dt>
          <dd>{claim.paper_evidence.page}</dd>
        </div>
      </dl>
      <blockquote className="mt-3 border-l-2 border-slate-300 pl-3 text-xs italic text-slate-600">
        "{claim.paper_evidence.quote}"
      </blockquote>
    </li>
  );
}