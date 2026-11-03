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

// Claim2Code — the workbench.
//
// A framed-sheet shell: the sidebar and top bar sit on one tinted frame
// and the current view on a white sheet. Without an open analysis the
// sheet is "New analysis" (upload a paper, or open an existing one).
// With one, the sidebar names it and its repository and switches
// between its views: verification (the default), claims, code symbols
// and retrieval. The chat model has its own view.
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

type View = "verify" | "claims" | "symbols" | "retrieval" | "model";

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
  const [view, setView] = useState<View>("verify");
  const [menuOpen, setMenuOpen] = useState(false);

  // An analysis opened from the URL: count its claims, and fall back
  // to the new-analysis screen if the id is unknown.
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
    setView("verify");
    setMenuOpen(false);
  }

  function opened(result: { analysisId: string; nClaims: number }) {
    setAnalysisId(result.analysisId);
    setNClaims(result.nClaims);
    setView("verify");
  }

  function go(next: View) {
    setView(next);
    setMenuOpen(false);
  }

  const showingModel = view === "model";
  const crumb = showingModel
    ? "Model"
    : !analysisId
      ? "New analysis"
      : repo
        ? `${repo.owner}/${repo.name}`
        : `Analysis ${analysisId.slice(0, 8)}`;

  const nav = (
    <SidebarNav
      analysisId={analysisId}
      nClaims={nClaims}
      repo={repo}
      view={view}
      online={online}
      version={health?.version ?? null}
      onGo={go}
      onNew={startOver}
      onOpenDemo={onOpenDemo}
    />
  );

  return (
    <div className="min-h-screen md:grid md:grid-cols-[228px_minmax(0,1fr)]">
      <aside className="hidden md:block">
        <div className="sticky top-0 flex h-screen flex-col px-3 pb-4 pt-3">{nav}</div>
      </aside>

      <div className="flex min-h-screen min-w-0 flex-col">
        <header className="flex h-12 items-center justify-between gap-3 px-4 md:px-2">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              className="btn-secondary px-2 md:hidden"
              aria-label="Menu"
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen((o) => !o)}
            >
              <MenuIcon />
            </button>
            <span className="md:hidden">
              <Mark />
            </span>
            <p className="truncate text-ink-muted">
              <span className="hidden sm:inline">Analyses&nbsp;&nbsp;/&nbsp;&nbsp;</span>
              <span className="text-ink">{crumb}</span>
            </p>
          </div>
          {analysisId && (
            <button type="button" onClick={startOver} className="btn-secondary">
              New analysis
            </button>
          )}
        </header>

        {menuOpen && (
          <div className="fixed inset-0 z-20 flex flex-col bg-frame px-4 pb-6 pt-3 md:hidden">
            <div className="mb-4 flex justify-end">
              <button type="button" className="btn-secondary" onClick={() => setMenuOpen(false)}>
                Close
              </button>
            </div>
            {nav}
          </div>
        )}

        <main className="mx-2 mb-2 flex-1 rounded-sheet border border-line bg-sheet px-4 py-6 sm:px-8 md:ml-0">
          {error && (
            <p className="error-block mb-6" data-testid="status-error">
              <span className="font-medium text-negative">The API is not reachable</span> ({error}).
              Start it with <code className="font-mono text-[13px]">make dev</code>, then reload.
            </p>
          )}

          {showingModel ? (
            <>
              <h1 className="text-[22px] font-semibold leading-8">Model</h1>
              <p className="mt-1 text-ink-muted">
                The chat model extracts claims, writes verdicts and compares related work.
              </p>
              <div className="mt-6 max-w-3xl">
                {online ? <ModelProviderPanel /> : <p className="text-ink-muted">—</p>}
              </div>
            </>
          ) : !analysisId ? (
            <NewAnalysis onOpened={opened} />
          ) : (
            <>
              {!repo && (
                <div className="mb-8">
                  <RepositorySection
                    analysisId={analysisId}
                    onCloned={(created) => setRepo(created)}
                  />
                </div>
              )}
              {view === "verify" && (
                <VerificationPanel analysisId={analysisId} repositoryBound={repo !== null} />
              )}
              {view === "claims" && <ClaimsList analysisId={analysisId} />}
              {view === "symbols" && repo && <SymbolsList repositoryId={repo.repository_id} />}
              {view === "retrieval" && repo && (
                <ClaimRetrievalPanel repositoryId={repo.repository_id} analysisId={analysisId} />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function NewAnalysis({
  onOpened,
}: {
  onOpened: (result: { analysisId: string; nClaims: number }) => void;
}) {
  return (
    <>
      <h1 className="text-[22px] font-semibold leading-8">New analysis</h1>
      <p className="mt-1 text-ink-muted">
        Upload a paper; its claims are extracted. Then bind the repository that implements it
        and verify each claim against the code.
      </p>

      <div className="mt-8 grid max-w-5xl gap-x-10 gap-y-3 md:grid-cols-[240px_minmax(0,1fr)]">
        <div>
          <h2 className="text-[16px] font-medium">Paper</h2>
          <p className="mt-1 text-ink-muted">
            A digitally generated PDF, up to 50 MB. Scanned pages are not read.
          </p>
        </div>
        <UploadSection onUploaded={onOpened} />
      </div>

      <div className="my-8 max-w-5xl border-t border-line" />

      <div className="grid max-w-5xl gap-x-10 gap-y-3 md:grid-cols-[240px_minmax(0,1fr)]">
        <div>
          <h2 className="text-[16px] font-medium">Existing analysis</h2>
          <p className="mt-1 text-ink-muted">
            Its claims, repository and verdicts are kept, so it opens where you left it.
          </p>
        </div>
        <OpenAnalysisForm onOpened={onOpened} />
      </div>
    </>
  );
}

type SidebarNavProps = {
  analysisId: string | null;
  nClaims: number;
  repo: BoundRepo | null;
  view: View;
  online: boolean;
  version: string | null;
  onGo: (view: View) => void;
  onNew: () => void;
  onOpenDemo: () => void;
};

function SidebarNav({
  analysisId,
  nClaims,
  repo,
  view,
  online,
  version,
  onGo,
  onNew,
  onOpenDemo,
}: SidebarNavProps) {
  return (
    <nav className="flex flex-1 flex-col" aria-label="Workbench">
      <button type="button" onClick={onNew} className="hidden items-center gap-2 px-2 md:flex">
        <Mark />
        <span className="text-[15px] font-semibold">Claim2Code</span>
      </button>

      {analysisId ? (
        <>
          <div className="mt-6 px-2">
            <p className="text-[12px] text-ink-muted">Analysis</p>
            <CopyId id={analysisId} />
            <p className="mt-1 text-[13px] text-ink-muted" data-testid={repo ? "repo-summary" : undefined}>
              {repo ? (
                <>
                  <a href={repo.url} target="_blank" rel="noreferrer" className="text-ink hover:underline">
                    {repo.owner}/{repo.name}
                  </a>
                  <br />
                  {count(repo.n_files, "file")}, {count(repo.n_symbols, "symbol")}
                </>
              ) : (
                "No repository yet"
              )}
            </p>
            <p className="sr-only" data-testid="upload-summary">
              extracted {nClaims} claim{nClaims === 1 ? "" : "s"}
            </p>
          </div>

          <ul className="mt-4 space-y-0.5">
            <NavItem active={view === "verify"} onClick={() => onGo("verify")}>
              Verification
            </NavItem>
            <NavItem active={view === "claims"} onClick={() => onGo("claims")} count={nClaims}>
              Claims
            </NavItem>
            <NavItem active={view === "symbols"} onClick={() => onGo("symbols")} disabled={!repo}>
              Code symbols
            </NavItem>
            <NavItem active={view === "retrieval"} onClick={() => onGo("retrieval")} disabled={!repo}>
              Retrieval
            </NavItem>
          </ul>
        </>
      ) : (
        <ul className="mt-6 space-y-0.5">
          <NavItem active={view !== "model"} onClick={() => onGo("verify")}>
            New analysis
          </NavItem>
        </ul>
      )}

      <div className="mt-auto pt-6">
        <ul className="space-y-0.5">
          <NavItem active={view === "model"} onClick={() => onGo("model")} disabled={!online}>
            Model
          </NavItem>
          <NavItem active={false} onClick={onOpenDemo}>
            Sample analysis
          </NavItem>
        </ul>
        <p
          className="flex items-center gap-2 px-2 pt-3 text-[12px] text-ink-muted"
          data-testid="status-badge"
        >
          <span className={`h-1.5 w-1.5 rounded-full ${online ? "bg-positive" : "bg-negative"}`} />
          API {online ? "online" : "offline"}
          {version && <span className="num">· v{version}</span>}
        </p>
      </div>
    </nav>
  );
}

function NavItem({
  active,
  onClick,
  disabled = false,
  count,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  count?: number;
  children: React.ReactNode;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        aria-current={active ? "page" : undefined}
        className={`flex h-8 w-full items-center justify-between rounded-control px-2 text-left ${
          active ? "bg-active text-ink" : "text-ink-muted hover:text-ink"
        } disabled:cursor-not-allowed disabled:opacity-50`}
      >
        {children}
        {count !== undefined && count > 0 && (
          <span className="num text-[12px] text-ink-muted" aria-hidden>
            {count}
          </span>
        )}
      </button>
    </li>
  );
}

// The mark: a page and a bracket, the paper and its code.
function Mark() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden className="flex-none text-accent">
      <path d="M3 2.5h7l3 3v12H3z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M6 9h4M6 12h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M15 8l2.5 3.5L15 15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MenuIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
      <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function count(n: number | null, noun: string): string {
  if (n === null) return "—";
  return `${n.toLocaleString()} ${noun}${n === 1 ? "" : "s"}`;
}

function CopyId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      title="Copy the analysis id"
      onClick={() => {
        navigator.clipboard?.writeText(id).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        });
      }}
      className="mt-0.5 flex items-center gap-1.5 font-mono text-[13px] text-ink hover:text-accent"
    >
      {id.slice(0, 8)}…
      <span className="font-sans text-[12px] text-ink-muted">{copied ? "copied" : "copy"}</span>
    </button>
  );
}
