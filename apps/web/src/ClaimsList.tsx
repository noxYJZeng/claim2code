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
      <p className="error-block" data-testid="claims-error">
        Couldn't load the claims: {error}
      </p>
    );
  }

  if (!data) {
    return (
      <p className="text-ink-muted" data-testid="claims-loading">
        Loading claims…
      </p>
    );
  }

  return (
    <section data-testid={data.claims.length ? "claims-list" : "claims-empty"}>
      <h1 className="text-[22px] font-semibold leading-8">Claims</h1>
      <p className="mt-1 text-ink-muted">
        {data.claims.length === 0
          ? "The paper parsed, but no testable claims were extracted."
          : `${data.claims.length} claim${data.claims.length === 1 ? "" : "s"} extracted from the paper, each with the passage it comes from.`}
      </p>

      {data.claims.length > 0 && (
        <div className="card mt-6 overflow-x-auto">
          <table className="w-full min-w-[640px] text-left">
            <thead>
              <tr className="border-b border-line text-[12px] text-ink-muted">
                <th className="w-14 py-2.5 pl-4 font-normal">Id</th>
                <th className="py-2.5 pr-4 font-normal">Claim and passage</th>
                <th className="w-32 py-2.5 pr-4 font-normal">Type</th>
                <th className="w-16 py-2.5 pr-4 text-right font-normal">Page</th>
                <th className="w-24 py-2.5 pr-4 text-right font-normal">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {data.claims.map((claim) => (
                <ClaimRow key={claim.claim_id} claim={claim} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function ClaimRow({ claim }: { claim: Claim }) {
  const pct = Math.round(claim.confidence * 100);
  const type = claim.claim_type.toLowerCase();
  return (
    <tr data-testid="claim-card" className="border-b border-line-rule align-top last:border-b-0">
      <td className="py-3 pl-4 font-mono text-[12px] text-ink-muted">{claim.claim_id}</td>
      <td className="py-3 pr-4">
        <p>{claim.text}</p>
        <p className="mt-1 text-[13px] text-ink-muted">
          “{claim.paper_evidence.quote}” — {claim.paper_evidence.section_title || "Preamble"}
        </p>
      </td>
      <td className="py-3 pr-4 text-ink-muted">{type.charAt(0).toUpperCase() + type.slice(1)}</td>
      <td className="num py-3 pr-4 text-right">{claim.paper_evidence.page}</td>
      <td className="num py-3 pr-4 text-right" data-testid="claim-confidence">
        {pct}%
      </td>
    </tr>
  );
}
