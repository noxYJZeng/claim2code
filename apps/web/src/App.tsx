import { useEffect, useState } from "react";
import {
  getAnalysisRepository,
  getClaims,
  getHealth,
  type Health,
} from "./api";
import UploadSection from "./UploadSection";
import ClaimsList from "./ClaimsList";
import RepositorySection from "./RepositorySection";
import SymbolsList from "./SymbolsList";
import ClaimRetrievalPanel from "./ClaimRetrievalPanel";
import VerificationPanel from "./VerificationPanel";
import OpenAnalysisForm from "./OpenAnalysisForm";
import ModelProviderPanel from "./ModelProviderPanel";
import DemoAnalysis from "./DemoAnalysis";

// Claim2Code — single-page workbench.
//
// The flow has three steps, shown as a stepper under the top bar:
// 1. upload a paper (claims are extracted), 2. bind its repository
// (cloned and indexed), 3. verify claims against that repository.
// Once a paper is open, the analysis view shows the repository card
// and tabs for verdicts, extracted claims, indexed symbols and the
// retrieval explorer.
//
// The open analysis is mirrored in the URL (?analysis=<id>), so a
// refresh or a shared link lands on the same analysis; its bound
// repository is looked up from the API rather than asked for again.
// ?demo=1 shows a static, read-only sample with no backend.

// The bound repository as the workbench needs it, from either the
// clone response or the analysis's stored repository.
export type BoundRepo = {
  repository_id: string;
  url: string;
  owner: string;
  name: string;
  n_files: number | null;
  n_symbols: number | null;
};

type Tab = "verify" | "claims" | "symbols" | "retrieval";

export default function App() {
  const [demo, setDemo] = useState(() =>
    new URLSearchParams(window.location.search).get("demo") === "1",
  );

  function changeMode(showDemo: boolean) {
    const url = new URL(window.location.href);
    if (showDemo) url.searchParams.set("demo", "1");
    else url.searchParams.delete("demo");
    window.history.replaceState(null, "", url);
    setDemo(showDemo);
  }

  // The static demo mounts no workbench effects or API clients. A
  // portfolio can link directly to ?demo=1 without a backend or key.
  return demo ? (
    <DemoAnalysis onOpenWorkbench={() => changeMode(false)} />
  ) : (
    <Workbench onOpenDemo={() => changeMode(true)} />
  );
}

function Workbench({ onOpenDemo }: { onOpenDemo: () => void }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("analysis"),
  );
  const [nClaims, setNClaims] = useState<number>(0);
  const [repo, setRepo] = useState<BoundRepo | null>(null);
  const [tab, setTab] = useState<Tab>("verify");
  const [showModel, setShowModel] = useState(false);

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

  // A reopened analysis keeps the repository bound to it.
  useEffect(() => {
    if (!analysisId) return;
    const controller = new AbortController();
    getAnalysisRepository(analysisId, controller.signal).then((result) => {
      if (result.ok && result.metadata.index_status === "done") {
        setRepo((current) => current ?? result.metadata);
      }
    });
    return () => controller.abort();
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

  function startOver() {
    setAnalysisId(null);
    setRepo(null);
    setNClaims(0);
    setTab("verify");
  }

  function opened(result: { analysisId: string; nClaims: number }) {
    setAnalysisId(result.analysisId);
    setNClaims(result.nClaims);
  }

  const step = !analysisId ? 1 : !repo ? 2 : 3;

  return (
    <main className="min-h-screen bg-slate-50 font-sans text-slate-900">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <button
            type="button"
            onClick={startOver}
            className="flex items-baseline gap-2 text-left"
          >
            <h1 className="text-xl font-semibold tracking-tight">Claim2Code</h1>
            <span className="hidden text-sm text-slate-500 sm:inline">
              paper ↔ code verification
            </span>
          </button>
          <div className="flex items-center gap-2">
            <span
              data-testid="status-badge"
              title={health?.version ? `API v${health.version}` : undefined}
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
                online ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${online ? "bg-emerald-500" : "bg-rose-500"}`}
              />
              {online ? "online" : "offline"}
            </span>
            {online && (
              <button
                type="button"
                onClick={() => setShowModel((s) => !s)}
                aria-expanded={showModel}
                className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100"
              >
                Model
              </button>
            )}
            <button
              type="button"
              onClick={onOpenDemo}
              className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100"
            >
              View sample analysis
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-5xl space-y-6 px-4 py-8 sm:px-6">
        {error && (
          <p
            className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800"
            data-testid="status-error"
          >
            The API is not reachable ({error}). Start it with{" "}
            <code className="font-mono text-xs">make dev</code> or uvicorn, then reload.
          </p>
        )}
        {online && showModel && <ModelProviderPanel />}

        <Stepper step={step} />

        {!analysisId ? (
          <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_18rem]">
            <UploadSection onUploaded={opened} />
            <OpenAnalysisForm onOpened={opened} />
          </div>
        ) : (
          <>
            <div className="divide-y divide-slate-100 rounded-2xl border border-slate-200 bg-white text-sm shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3">
                <span data-testid="upload-summary" className="text-slate-700">
                  <span className="mr-2 font-medium text-slate-900">Paper</span>
                  extracted {nClaims} claim{nClaims === 1 ? "" : "s"}
                  <CopyId id={analysisId} />
                </span>
                <button
                  type="button"
                  onClick={startOver}
                  className="text-xs font-medium text-slate-600 hover:text-slate-900 hover:underline"
                >
                  Upload another
                </button>
              </div>
              {repo && (
                <div className="px-5 py-3 text-slate-700" data-testid="repo-summary">
                  <span className="mr-2 font-medium text-slate-900">Repository</span>
                  indexed{" "}
                  <a
                    href={repo.url}
                    target="_blank"
                    rel="noreferrer"
                    className="font-mono text-slate-900 underline decoration-slate-300 hover:decoration-slate-900"
                  >
                    {repo.owner}/{repo.name}
                  </a>{" "}
                  — {count(repo.n_files, "file")}, {count(repo.n_symbols, "symbol")}
                </div>
              )}
            </div>

            {!repo && (
              <RepositorySection
                analysisId={analysisId}
                onCloned={(created) => setRepo(created)}
              />
            )}

            <nav
              className="flex gap-1 overflow-x-auto border-b border-slate-200"
              aria-label="Analysis views"
            >
              <TabButton current={tab} value="verify" onSelect={setTab}>
                Verification
              </TabButton>
              <TabButton current={tab} value="claims" onSelect={setTab}>
                Claims
              </TabButton>
              {repo && (
                <TabButton current={tab} value="symbols" onSelect={setTab}>
                  Code symbols
                </TabButton>
              )}
              {repo && (
                <TabButton current={tab} value="retrieval" onSelect={setTab}>
                  Retrieval
                </TabButton>
              )}
            </nav>

            {tab === "verify" && (
              <VerificationPanel analysisId={analysisId} repositoryBound={repo !== null} />
            )}
            {tab === "claims" && <ClaimsList analysisId={analysisId} />}
            {tab === "symbols" && repo && <SymbolsList repositoryId={repo.repository_id} />}
            {tab === "retrieval" && repo && (
              <ClaimRetrievalPanel
                repositoryId={repo.repository_id}
                analysisId={analysisId}
              />
            )}
          </>
        )}
      </div>
    </main>
  );
}

function count(n: number | null, noun: string): string {
  if (n === null) return `? ${noun}s`;
  return `${n.toLocaleString()} ${noun}${n === 1 ? "" : "s"}`;
}

const STEPS = ["Upload paper", "Bind repository", "Verify claims"];

function Stepper({ step }: { step: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm" aria-label="Progress">
      {STEPS.map((label, i) => {
        const n = i + 1;
        const done = n < step;
        const current = n === step;
        return (
          <li key={label} className="flex items-center gap-3">
            <span
              aria-current={current ? "step" : undefined}
              className={`flex items-center gap-2 ${
                current ? "font-medium text-slate-900" : done ? "text-slate-600" : "text-slate-400"
              }`}
            >
              <span
                className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-semibold ${
                  done
                    ? "bg-emerald-500 text-white"
                    : current
                      ? "bg-slate-900 text-white"
                      : "bg-slate-200 text-slate-500"
                }`}
              >
                {done ? "✓" : n}
              </span>
              {label}
            </span>
            {n < STEPS.length && <span className="h-px w-6 bg-slate-300 sm:w-10" />}
          </li>
        );
      })}
    </ol>
  );
}

function TabButton({
  current,
  value,
  onSelect,
  children,
}: {
  current: Tab;
  value: Tab;
  onSelect: (tab: Tab) => void;
  children: React.ReactNode;
}) {
  const active = current === value;
  return (
    <button
      type="button"
      onClick={() => onSelect(value)}
      aria-pressed={active}
      className={`-mb-px whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium ${
        active
          ? "border-slate-900 text-slate-900"
          : "border-transparent text-slate-500 hover:text-slate-800"
      }`}
    >
      {children}
    </button>
  );
}

function CopyId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      title="Copy the analysis id to reopen it later"
      onClick={() => {
        navigator.clipboard?.writeText(id).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        });
      }}
      className="ml-3 rounded-md bg-slate-100 px-2 py-0.5 font-mono text-xs text-slate-500 hover:bg-slate-200"
    >
      {copied ? "copied" : `id ${id.slice(0, 8)}…`}
    </button>
  );
}
