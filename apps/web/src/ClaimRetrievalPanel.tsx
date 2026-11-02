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
      <p className="error-note" data-testid="retrieval-claims-error">
        Failed to load claims: {claimsError}
      </p>
    );
  }

  if (!claims) {
    return (
      <p className="text-sm text-ink-3" data-testid="retrieval-loading-claims">
        Loading claims…
      </p>
    );
  }

  if (claims.length === 0) {
    return (
      <p className="text-sm text-ink-2" data-testid="retrieval-empty">
        No claims available to retrieve against — upload a paper with extracted
        claims first.
      </p>
    );
  }

  const selectedClaim = claims.find((c) => c.claim_id === selectedClaimId);

  return (
    <section data-testid="retrieval-panel">
      <p className="text-sm text-ink-2">
        The code symbols nearest to a claim in the embedding index (pgvector cosine).
        Verification starts from a wider, reranked candidate set.
      </p>

      <label htmlFor="claim-picker" className="eyebrow mb-2 mt-6 block">
        Claim
      </label>
      <select
        id="claim-picker"
        data-testid="claim-picker"
        value={selectedClaimId}
        onChange={(e) => setSelectedClaimId(e.target.value)}
        className="field"
      >
        {claims.map((c) => (
          <option key={c.claim_id} value={c.claim_id}>
            {c.claim_id} · {c.text.slice(0, 90)}
            {c.text.length > 90 ? "…" : ""}
          </option>
        ))}
      </select>

      {selectedClaim && (
        <blockquote
          className="mt-4 border-l-2 border-ink pl-3 font-serif text-[15px] leading-relaxed"
          data-testid="retrieval-claim-quote"
        >
          {selectedClaim.text}
        </blockquote>
      )}

      <div className="mt-6">
        {error && (
          <p className="error-note" data-testid="retrieval-error">
            Retrieval failed: {error}
          </p>
        )}

        {loading && (
          <p className="text-sm text-ink-3" data-testid="retrieval-loading">
            Searching the index…
          </p>
        )}

        {!loading && !error && matches && matches.length === 0 && (
          <p className="text-sm text-ink-2" data-testid="retrieval-empty-results">
            No code symbols matched this claim — the repository may not be indexed yet,
            or the embedding model returned no close hits.
          </p>
        )}

        {!loading && !error && matches && matches.length > 0 && (
          <ol className="border-t border-rule" data-testid="retrieval-matches">
            {matches.map((m, idx) => (
              <li
                key={m.symbol_id}
                data-testid="retrieval-match"
                className="grid grid-cols-[2rem_minmax(0,1fr)_4rem] gap-x-3 border-b border-rule py-3"
              >
                <span className="font-mono text-xs text-ink-3">{idx + 1}</span>
                <div className="min-w-0">
                  <code
                    data-testid="retrieval-match-name"
                    className="block truncate font-mono text-[13px]"
                  >
                    {m.name}
                    {m.signature && <span className="text-ink-3">{m.signature}</span>}
                  </code>
                  <p className="mt-0.5 text-xs text-ink-3">
                    {m.kind} · <code data-testid="retrieval-match-path" className="font-mono">
                      {m.file_path}:{m.start_line}:{m.start_col}
                    </code>
                  </p>
                  {m.docstring && (
                    <p
                      className="mt-1 line-clamp-2 text-xs italic text-ink-2"
                      data-testid="retrieval-match-docstring"
                    >
                      {m.docstring}
                    </p>
                  )}
                </div>
                <span
                  className="text-right font-mono text-xs text-ink-2"
                  data-testid="retrieval-match-score"
                >
                  {(scores?.[idx] ?? m.similarity).toFixed(3)}
                </span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}
