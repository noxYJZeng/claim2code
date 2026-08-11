import { useEffect, useState } from "react";
import {
  getClaims,
  retrieveSymbols,
  type Claim,
  type RetrieveMatch,
} from "./api";

type ClaimRetrievalPanelProps = {
  repositoryId: string;
  analysisId: string;
};

// ClaimRetrievalPanel — Phase 4 semantic retrieval view.
//
// Given a repository that's been embedded (Phase 4 wires the
// indexer into run_repo_workflow, so any repo the user just
// cloned is already retrievable), this panel lets the user pick
// an extracted claim and POST it to
// /repositories/{id}/retrieve. The endpoint embeds the claim,
// runs the pgvector cosine query, and returns the top-k matching
// CodeSymbol rows with similarity scores.
//
// Why an embedded claim picker instead of free-form input
// --------------------------------------------------------
// Every claim we have is already a labelled, structured row from
// Phase 2; letting the user re-type one is error-prone and
// defeats the point of having structured claims in the first
// place. v1 exposes a dropdown; v2 (when Phase 6 adds verdicts)
// will auto-trigger retrieval on claim click.
//
// The panel fetches its own claim list (rather than receiving it
// from the parent) so it stays decoupled from ClaimsList — the
// two are read-only views onto the same analysis and don't
// need to share state.

export default function ClaimRetrievalPanel({
  repositoryId,
  analysisId,
}: ClaimRetrievalPanelProps) {
  const [claims, setClaims] = useState<Claim[] | null>(null);
  const [claimsError, setClaimsError] = useState<string | null>(null);
  const [selectedClaimId, setSelectedClaimId] = useState<string>("");
  const [matches, setMatches] = useState<RetrieveMatch[] | null>(null);
  const [scores, setScores] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Fetch the analysis's claims (shared with ClaimsList but
  // re-fetched here so the panel stays self-contained).
  useEffect(() => {
    const controller = new AbortController();
    getClaims(analysisId, controller.signal).then((result) => {
      if (result.ok) {
        setClaims(result.claims.claims);
        setClaimsError(null);
      } else {
        setClaims(null);
        setClaimsError(result.error);
      }
    });
    return () => controller.abort();
  }, [analysisId]);

  // When the claim list arrives, default the picker to the first
  // claim. If the user already picked one (and it's still in the
  // list), keep it.
  useEffect(() => {
    if (!claims || claims.length === 0) {
      if (selectedClaimId !== "") setSelectedClaimId("");
      return;
    }
    const stillValid = claims.some((c) => c.claim_id === selectedClaimId);
    if (!stillValid) {
      setSelectedClaimId(claims[0].claim_id);
    }
  }, [claims, selectedClaimId]);

  // Re-run retrieval whenever the user picks a different claim.
  useEffect(() => {
    if (!selectedClaimId || !claims) {
      setMatches(null);
      setScores(null);
      setError(null);
      return;
    }
    const claim = claims.find((c) => c.claim_id === selectedClaimId);
    if (!claim) {
      setMatches(null);
      setScores(null);
      setError("selected claim not found");
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    retrieveSymbols(repositoryId, claim.text, 5, controller.signal).then(
      (result) => {
        setLoading(false);
        if (result.ok) {
          setMatches(result.response.symbols);
          setScores(result.response.scores);
        } else {
          setMatches(null);
          setScores(null);
          setError(result.error);
        }
      },
    );
    return () => controller.abort();
  }, [selectedClaimId, claims, repositoryId]);

  if (claimsError) {
    return (
      <section
        className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800"
        data-testid="retrieval-claims-error"
      >
        Failed to load claims: {claimsError}
      </section>
    );
  }

  if (!claims) {
    return (
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-500"
        data-testid="retrieval-loading-claims"
      >
        Loading claims…
      </section>
    );
  }

  if (claims.length === 0) {
    return (
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600"
        data-testid="retrieval-empty"
      >
        No claims available to retrieve against — upload a paper with extracted
        claims first.
      </section>
    );
  }

  const selectedClaim = claims.find((c) => c.claim_id === selectedClaimId);

  return (
    <section
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="retrieval-panel"
    >
      <div className="mb-4 flex items-baseline justify-between gap-2">
        <h2 className="text-lg font-medium">Claim ↔ code retrieval</h2>
        <span className="text-xs text-slate-500">
          pgvector cosine over embedded symbols
        </span>
      </div>

      <label
        htmlFor="claim-picker"
        className="mb-2 block text-xs font-medium uppercase tracking-wide text-slate-500"
      >
        Pick a claim
      </label>
      <select
        id="claim-picker"
        data-testid="claim-picker"
        value={selectedClaimId}
        onChange={(e) => setSelectedClaimId(e.target.value)}
        className="mb-4 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-slate-500 focus:outline-none"
      >
        {claims.map((c) => (
          <option key={c.claim_id} value={c.claim_id}>
            [{c.claim_type}] {c.text.slice(0, 80)}
            {c.text.length > 80 ? "…" : ""}
          </option>
        ))}
      </select>

      {selectedClaim && (
        <blockquote
          className="mb-4 rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm italic text-slate-700"
          data-testid="retrieval-claim-quote"
        >
          {selectedClaim.text}
        </blockquote>
      )}

      {error && (
        <div
          className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800"
          data-testid="retrieval-error"
        >
          Retrieval failed: {error}
        </div>
      )}

      {loading && (
        <div
          className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-500"
          data-testid="retrieval-loading"
        >
          Searching the index…
        </div>
      )}

      {!loading && !error && matches && matches.length === 0 && (
        <div
          className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600"
          data-testid="retrieval-empty-results"
        >
          No code symbols matched this claim — the repo may not yet be indexed
          (check the <code>index_status</code> above) or the embedding model
          returned no close hits.
        </div>
      )}

      {!loading && !error && matches && matches.length > 0 && (
        <ul className="space-y-2" data-testid="retrieval-matches">
          {matches.map((m, idx) => (
            <li
              key={m.symbol_id}
              data-testid="retrieval-match"
              className="rounded-xl border border-slate-200 bg-slate-50 p-3"
            >
              <div className="flex items-baseline justify-between gap-3">
                <div className="flex flex-wrap items-baseline gap-2">
                  <span className="shrink-0 rounded-full bg-slate-900 px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-white">
                    {m.kind}
                  </span>
                  <span className="shrink-0 rounded-full bg-slate-200 px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-slate-700">
                    {m.language}
                  </span>
                  <code
                    data-testid="retrieval-match-name"
                    className="font-mono text-sm text-slate-900"
                  >
                    {m.name}
                    {m.signature && (
                      <span className="text-slate-500">{m.signature}</span>
                    )}
                  </code>
                </div>
                <span
                  className="shrink-0 rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800"
                  data-testid="retrieval-match-score"
                >
                  {(scores?.[idx] ?? m.similarity).toFixed(3)}
                </span>
              </div>
              <div className="mt-1 flex flex-wrap items-baseline gap-3 text-xs text-slate-500">
                <code data-testid="retrieval-match-path">
                  {m.file_path}:{m.start_line}:{m.start_col}
                </code>
                {m.docstring && (
                  <span
                    className="line-clamp-2 italic text-slate-600"
                    data-testid="retrieval-match-docstring"
                  >
                    {m.docstring}
                  </span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
