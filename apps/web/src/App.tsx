import { useEffect, useState } from "react";
import { getHealth, type Health } from "./api";
import UploadSection from "./UploadSection";
import ClaimsList from "./ClaimsList";

// Claim2Code — Phase 2 single-page UI.
//
// Stacks <UploadSection /> + <ClaimsList /> vertically. After a
// successful upload, the analysis_id flows into ClaimsList which
// fetches the extracted claims via GET /analyses/{id}/claims.
//
// Phase 3 will introduce react-router-dom when the repository page
// becomes its own route. For Phase 2, the whole interaction is one
// screen, so no router is needed.
export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [nClaims, setNClaims] = useState<number>(0);

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
        </header>

        {!analysisId ? (
          <UploadSection
            onUploaded={(result) => {
              setAnalysisId(result.analysisId);
              setNClaims(result.nClaims);
            }}
          />
        ) : (
          <>
            <div className="mb-6 flex items-baseline justify-between rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
              <span data-testid="upload-summary">
                Parsed and extracted {nClaims} claim
                {nClaims === 1 ? "" : "s"} from your paper.
              </span>
              <button
                type="button"
                onClick={() => setAnalysisId(null)}
                className="text-xs font-medium uppercase tracking-wide text-emerald-700 hover:underline"
              >
                Upload another
              </button>
            </div>
            <ClaimsList analysisId={analysisId} />
          </>
        )}
      </div>
    </main>
  );
}