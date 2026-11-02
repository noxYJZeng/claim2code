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
    <main className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-rule bg-paper/95 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <button type="button" onClick={startOver} className="flex items-baseline gap-3 text-left">
            <h1 className="font-serif text-xl font-semibold tracking-tight">Claim2Code</h1>
            <span className="hidden font-mono text-xs text-ink-3 sm:inline">paper ⇄ code</span>
          </button>
          <div className="flex items-center gap-5 text-sm">
            <span
              data-testid="status-badge"
              title={health?.version ? `API v${health.version}` : undefined}
              className="inline-flex items-center gap-2 text-ink-2"
            >
              <span
                className={`h-2 w-2 rounded-full ${online ? "bg-verdict-ok" : "bg-verdict-bad"}`}
              />
              API {online ? "online" : "offline"}
            </span>
            {online && (
              <button
                type="button"
                onClick={() => setShowModel((s) => !s)}
                aria-expanded={showModel}
                className="text-ink-2 hover:text-ink"
              >
                Model {showModel ? "▴" : "▾"}
              </button>
            )}
            <button type="button" onClick={onOpenDemo} className="link">
              View sample analysis
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-4 pb-24 pt-8 sm:px-6">
        {error && (
          <p className="error-note mb-6" data-testid="status-error">
            The API is not reachable ({error}). Start it with{" "}
            <code className="font-mono text-xs">make dev</code> or uvicorn, then reload.
          </p>
        )}
        {online && showModel && (
          <div className="mb-8">
            <ModelProviderPanel />
          </div>
        )}

        {!analysisId ? (
          <>
            <section className="max-w-3xl pb-8 pt-6">
              <h2 className="font-serif text-3xl leading-tight tracking-tight sm:text-4xl">
                Does the code do what the paper says?
              </h2>
              <p className="mt-4 text-[15px] leading-relaxed text-ink-2">
                Give Claim2Code a paper and the repository that implements it. Each
                claim comes back verified, partially verified, contradicted or
                unverified, with the paper passage and the lines of code it rests on.
              </p>
            </section>
            <Stepper step={step} />
            <div className="mt-6 grid gap-6 md:grid-cols-[minmax(0,1fr)_17rem]">
              <UploadSection onUploaded={opened} />
              <OpenAnalysisForm onOpened={opened} />
            </div>
          </>
        ) : (
          <>
            <section className="border-b border-rule pb-6">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <p className="eyebrow">
                  Analysis <CopyId id={analysisId} />
                </p>
                <button type="button" onClick={startOver} className="link text-sm">
                  Upload another paper
                </button>
              </div>
              <dl className="mt-3 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[7rem_minmax(0,1fr)]">
                <dt className="text-ink-3">Paper</dt>
                <dd data-testid="upload-summary">
                  extracted {nClaims} claim{nClaims === 1 ? "" : "s"}
                </dd>
                <dt className="text-ink-3">Repository</dt>
                <dd data-testid={repo ? "repo-summary" : undefined}>
                  {repo ? (
                    <>
                      indexed{" "}
                      <a href={repo.url} target="_blank" rel="noreferrer" className="link font-mono">
                        {repo.owner}/{repo.name}
                      </a>{" "}
                      — {count(repo.n_files, "file")}, {count(repo.n_symbols, "symbol")}
                    </>
                  ) : (
                    <span className="text-ink-3">not bound yet</span>
                  )}
                </dd>
              </dl>
            </section>

            {!repo && (
              <div className="mt-8">
                <Stepper step={step} />
                <div className="mt-6">
                  <RepositorySection
                    analysisId={analysisId}
                    onCloned={(created) => setRepo(created)}
                  />
                </div>
              </div>
            )}

            <nav
              className="mt-8 flex flex-wrap gap-x-6 border-b border-rule"
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

            <div className="mt-6">
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
            </div>
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

const STEPS = ["Paper", "Repository", "Verification"];

function Stepper({ step }: { step: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm" aria-label="Progress">
      {STEPS.map((label, i) => {
        const n = i + 1;
        const done = n < step;
        const current = n === step;
        return (
          <li key={label} className="flex items-center gap-4">
            <span
              aria-current={current ? "step" : undefined}
              className={`flex items-baseline gap-2 ${
                current ? "text-ink" : done ? "text-verdict-ok" : "text-ink-3"
              }`}
            >
              <span className="font-mono text-xs">{done ? "✓" : `0${n}`}</span>
              <span className={current ? "font-medium" : ""}>{label}</span>
            </span>
            {n < STEPS.length && <span className="h-px w-8 bg-rule-strong sm:w-14" />}
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
      className={`-mb-px whitespace-nowrap border-b-2 pb-2 text-sm ${
        active
          ? "border-ink font-medium text-ink"
          : "border-transparent text-ink-3 hover:text-ink"
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
      className="ml-2 font-mono normal-case tracking-normal text-ink-2 hover:text-ink"
    >
      {copied ? "copied" : `${id.slice(0, 8)} ⧉`}
    </button>
  );
}
