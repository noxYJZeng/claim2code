import { useEffect, useState } from "react";
import { getClaims, getHealth, type Health, type RepoCreate } from "./api";
import UploadSection from "./UploadSection";
import ClaimsList from "./ClaimsList";
import RepositorySection from "./RepositorySection";
import SymbolsList from "./SymbolsList";
import ClaimRetrievalPanel from "./ClaimRetrievalPanel";
import VerificationPanel from "./VerificationPanel";
import OpenAnalysisForm from "./OpenAnalysisForm";
import ModelProviderPanel from "./ModelProviderPanel";

// Claim2Code — Phase 3 single-page UI.
//
// Three sections stacked: <UploadSection/> → <ClaimsList/> →
// <RepositorySection/> (+ <SymbolsList/> once a repo is bound).
// After a successful paper upload the analysis_id flows into both
// the claims list and the repository section; both sections are
// read-only views onto the same analysis.
//
// Phase 4 appends <ClaimRetrievalPanel/> below SymbolsList: once
// a repo is bound, the user can pick one of the extracted claims
// and the panel POSTs it to /repositories/{id}/retrieve, rendering
// the top-k matching symbols with similarity scores.
//
// Phase 6 appends <VerificationPanel/>: per-claim verdicts against
// the bound repository. An earlier analysis can be reopened by id
// (<OpenAnalysisForm/>), since upload and indexing take minutes. The
// open analysis is mirrored in the URL (?analysis=<id>), so a refresh
// or a shared link lands on the same analysis.
//
// <ModelProviderPanel/> in the header picks the chat model the
// backend uses (llm/providers.py).
//
// Phase 5+ will introduce react-router-dom when the repository page
// becomes its own route; for now, the whole interaction is still
// one screen.
export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("analysis"),
  );
  const [nClaims, setNClaims] = useState<number>(0);
  const [repo, setRepo] = useState<RepoCreate | null>(null);

  // An analysis opened from the URL: count its claims for the summary,
  // and fall back to the upload screen if the id is unknown.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("analysis");
    if (!id) return;
    const controller = new AbortController();
    getClaims(id, controller.signal).then((result) => {
      if (result.ok) setNClaims(result.claims.claims.length);
      else if (/^HTTP 4/.test(result.error)) setAnalysisId(null);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (analysisId) url.searchParams.set("analysis", analysisId);
    else url.searchParams.delete("analysis");
    window.history.replaceState(null, "", url);
  }, [analysisId]);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal).then((result) => {
      if (result.ok) {
        setHealth(result.health);
        setError(null);
      } else {
        setHealth(null);
        setError(result.error);
      }
    });
    return () => controller.abort();
  }, []);

  const online = health?.status === "ok";

  return (
    <main className="min-h-screen bg-slate-50 font-sans text-slate-900">
      <div className="mx-auto max-w-3xl px-6 py-16">
        <header className="mb-10">
          <h1 className="text-4xl font-semibold tracking-tight">Claim2Code</h1>
          <p className="mt-2 text-slate-600">
            AI Research Agent for Paper–Code Verification and Reproducibility.
          </p>
          <div className="mt-4 flex items-center gap-3">
            <span
              data-testid="status-badge"
              className={
                online
                  ? "inline-flex items-center gap-2 rounded-full bg-emerald-100 px-3 py-1 text-sm font-medium text-emerald-800"
                  : "inline-flex items-center gap-2 rounded-full bg-rose-100 px-3 py-1 text-sm font-medium text-rose-800"
              }
            >
              <span
                className={
                  online
                    ? "h-2 w-2 rounded-full bg-emerald-500"
                    : "h-2 w-2 rounded-full bg-rose-500"
                }
              />
              {online ? "online" : "offline"}
            </span>
            {health?.version && (
              <span className="text-sm text-slate-500">
                API v{health.version}
              </span>
            )}
          </div>
          {error && (
            <p
              className="mt-3 text-sm text-rose-700"
              data-testid="status-error"
            >
              {error}
            </p>
          )}
          {online && <ModelProviderPanel />}
        </header>

        {!analysisId ? (
          <div className="space-y-6">
            <UploadSection
              onUploaded={(result) => {
                setAnalysisId(result.analysisId);
                setNClaims(result.nClaims);
              }}
            />
            <OpenAnalysisForm
              onOpened={(result) => {
                setAnalysisId(result.analysisId);
                setNClaims(result.nClaims);
              }}
            />
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex items-baseline justify-between rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
              <span data-testid="upload-summary">
                Parsed and extracted {nClaims} claim
                {nClaims === 1 ? "" : "s"} from your paper.
              </span>
              <button
                type="button"
                onClick={() => {
                  setAnalysisId(null);
                  setRepo(null);
                }}
                className="text-xs font-medium uppercase tracking-wide text-emerald-700 hover:underline"
              >
                Upload another
              </button>
            </div>
            <ClaimsList analysisId={analysisId} />
            <RepositorySection
              analysisId={analysisId}
              onCloned={(created) => setRepo(created)}
            />
            {repo && (
              <div
                className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800"
                data-testid="repo-summary"
              >
                Cloned and indexed {repo.name} — {repo.n_files} file
                {repo.n_files === 1 ? "" : "s"}, {repo.n_symbols} symbol
                {repo.n_symbols === 1 ? "" : "s"}.
              </div>
            )}
            {repo && <SymbolsList repositoryId={repo.repository_id} />}
            <VerificationPanel analysisId={analysisId} />
            {repo && (
              <ClaimRetrievalPanel
                repositoryId={repo.repository_id}
                analysisId={analysisId}
              />
            )}
          </div>
        )}
      </div>
    </main>
  );
}
