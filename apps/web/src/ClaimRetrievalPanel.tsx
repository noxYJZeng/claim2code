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
      <p className="error-block" data-testid="retrieval-claims-error">
        Couldn't load the claims: {claimsError}
      </p>
    );
  }

  if (!claims) {
    return (
      <p className="text-ink-muted" data-testid="retrieval-loading-claims">
        Loading claims…
      </p>
    );
  }

  if (claims.length === 0) {
    return (
      <p className="text-ink-muted" data-testid="retrieval-empty">
        No claims to retrieve against: the paper has no extracted claims.
      </p>
    );
  }

  const selectedClaim = claims.find((c) => c.claim_id === selectedClaimId);

  return (
    <section data-testid="retrieval-panel">
      <h1 className="text-[22px] font-semibold leading-8">Retrieval</h1>
      <p className="mt-1 text-ink-muted">
        The code nearest to a claim in the embedding index. Verification starts from a wider
        candidate set, reranked by the chat model.
      </p>

      <div className="mt-6 max-w-3xl">
        <label htmlFor="claim-picker" className="mb-2 block font-medium">
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
          <p className="mt-3 leading-[22px]" data-testid="retrieval-claim-quote">
            {selectedClaim.text}
          </p>
        )}
      </div>

      <div className="mt-6 max-w-3xl">
        {error && (
          <p className="error-block" data-testid="retrieval-error">
            Retrieval failed: {error}
          </p>
        )}
        {loading && (
          <p className="text-ink-muted" data-testid="retrieval-loading">
            Searching the index…
          </p>
        )}
        {!loading && !error && matches && matches.length === 0 && (
          <p className="text-ink-muted" data-testid="retrieval-empty-results">
            No code matched this claim. The repository may not be indexed yet, or nothing in it
            is close.
          </p>
        )}
        {!loading && !error && matches && matches.length > 0 && (
          <div className="card">
            <div className="grid grid-cols-[2rem_minmax(0,1fr)_4.5rem] gap-3 border-b border-line px-4 py-2.5 text-[12px] text-ink-muted">
              <span>Rank</span>
              <span>Symbol</span>
              <span className="text-right">Similarity</span>
            </div>
            <ol data-testid="retrieval-matches">
              {matches.map((m, idx) => (
                <li
                  key={m.symbol_id}
                  data-testid="retrieval-match"
                  className="grid grid-cols-[2rem_minmax(0,1fr)_4.5rem] gap-3 border-b border-line-rule px-4 py-3 last:border-b-0"
                >
                  <span className="num text-ink-muted">{idx + 1}.</span>
                  <div className="min-w-0">
                    <code data-testid="retrieval-match-name" className="block truncate font-mono text-[13px]">
                      {m.name}
                      {m.signature && <span className="text-ink-muted">{m.signature}</span>}
                    </code>
                    <p className="mt-0.5 truncate text-[12px] text-ink-muted">
                      {m.kind} in{" "}
                      <code data-testid="retrieval-match-path" className="font-mono">
                        {m.file_path}:{m.start_line}:{m.start_col}
                      </code>
                    </p>
                    {m.docstring && (
                      <p className="mt-1 line-clamp-2 text-[13px] text-ink-muted" data-testid="retrieval-match-docstring">
                        {m.docstring}
                      </p>
                    )}
                  </div>
                  <span className="num text-right" data-testid="retrieval-match-score">
                    {(scores?.[idx] ?? m.similarity).toFixed(3)}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </section>
  );
}
