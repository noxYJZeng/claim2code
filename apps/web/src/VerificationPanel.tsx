import { useEffect, useRef, useState } from "react";
import {
  getClaims,
  getVerifications,
  runVerification,
  type Claim,
  type Verification,
  type VerificationStatus,
} from "./api";
import CitedCode from "./CitedCode";
import CitedPaper from "./CitedPaper";
import Spinner from "./Spinner";
import { STATUSES, STATUS_EDGE, STATUS_INK, STATUS_LABEL, STATUS_MARK } from "./verdicts";

type VerificationPanelProps = {
  analysisId: string;
  // Verification needs the analysis's repository; until one is bound
  // the stored verdicts are shown but no new run can start.
  repositoryBound?: boolean;
};

// VerificationPanel — Phase 6 verdict view.
//
// Lists every extracted claim with its stored verdict (GET
// /analyses/{id}/verifications) and lets the user verify one claim
// or all of them against the repository bound to the analysis.
//
// "Verify all" sends one POST per claim, in order, rather than one
// POST for the whole list: each claim costs an LLM call (minutes on
// a local model), so verdicts render as they land and the user can
// stop between claims. The server keeps the stored verdicts of the
// claims a request does not name.

const SEVERITY_INK: Record<string, string> = {
  HIGH: "text-verdict-bad",
  MEDIUM: "text-verdict-partial",
  LOW: "text-ink-3",
};

export default function VerificationPanel({
  analysisId,
  repositoryBound = true,
}: VerificationPanelProps) {
  const [claims, setClaims] = useState<Claim[] | null>(null);
  const [verdicts, setVerdicts] = useState<Record<string, Verification>>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  // Claim currently being verified, and the queue still waiting.
  const [running, setRunning] = useState<string | null>(null);
  const [queued, setQueued] = useState<string[]>([]);
  const [batchSize, setBatchSize] = useState(0);
  const [filter, setFilter] = useState<VerificationStatus | "PENDING" | null>(null);
  const stopRequested = useRef(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    Promise.all([
      getClaims(analysisId, controller.signal),
      getVerifications(analysisId, controller.signal),
    ]).then(([claimsResult, verdictsResult]) => {
      if (!claimsResult.ok) {
        setLoadError(claimsResult.error);
        return;
      }
      if (!verdictsResult.ok) {
        setLoadError(verdictsResult.error);
        return;
      }
      setClaims(claimsResult.claims.claims);
      setVerdicts(byClaim(verdictsResult.verifications));
      setLoadError(null);
    });
    return () => {
      mounted.current = false;
      stopRequested.current = true;
      controller.abort();
    };
  }, [analysisId]);

  async function verify(claimIds: string[]) {
    stopRequested.current = false;
    setRunError(null);
    setBatchSize(claimIds.length);
    for (let i = 0; i < claimIds.length; i++) {
      if (stopRequested.current || !mounted.current) break;
      const claimId = claimIds[i];
      setRunning(claimId);
      setQueued(claimIds.slice(i + 1));
      const result = await runVerification(analysisId, [claimId]);
      if (!mounted.current) return;
      if (!result.ok) {
        setRunError(`${claimId}: ${result.error}`);
        break;
      }
      setVerdicts((prev) => ({ ...prev, ...byClaim(result.verifications) }));
    }
    setRunning(null);
    setQueued([]);
  }

  if (loadError) {
    return (
      <p className="error-note" data-testid="verification-error">
        Failed to load verifications: {loadError}
      </p>
    );
  }

  if (!claims) {
    return (
      <p className="text-sm text-ink-3" data-testid="verification-loading">
        Loading verifications…
      </p>
    );
  }

  const busy = running !== null;
  const pending = claims.filter((c) => !verdicts[c.claim_id]).map((c) => c.claim_id);
  const shown = claims.filter((c) => {
    if (!filter) return true;
    const v = verdicts[c.claim_id];
    return filter === "PENDING" ? !v : v?.status === filter;
  });
  const counts = STATUSES.map(
    (s) => [s, Object.values(verdicts).filter((v) => v.status === s).length] as const,
  );
  const done = batchSize - queued.length - 1;

  return (
    <section data-testid="verification-panel">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <p className="max-w-2xl text-sm leading-relaxed text-ink-2">
          Each claim is checked against the bound repository with one model call (minutes
          on a local model). Every verdict cites the code it rests on. <em>Unverified</em>{" "}
          means no supporting code was found, not that the code lacks it.
        </p>
        {busy ? (
          <button
            type="button"
            onClick={() => {
              stopRequested.current = true;
            }}
            className="btn-quiet"
            data-testid="verify-stop"
          >
            Stop after this claim
          </button>
        ) : (
          <button
            type="button"
            disabled={pending.length === 0 || !repositoryBound}
            onClick={() => verify(pending)}
            className="btn-primary"
            data-testid="verify-all"
          >
            Verify {pending.length} unverified claim{pending.length === 1 ? "" : "s"}
          </button>
        )}
      </div>

      {!repositoryBound && (
        <p
          className="mt-4 border-l-2 border-verdict-partial pl-3 text-sm text-verdict-partial"
          data-testid="verification-needs-repo"
        >
          Bind the paper's repository above to verify claims.
        </p>
      )}

      <div
        className="mt-6 flex flex-wrap items-baseline gap-x-5 gap-y-2 text-sm"
        data-testid="verification-counts"
      >
        {counts.map(([status, n]) => (
          <button
            type="button"
            key={status}
            onClick={() => setFilter((f) => (f === status ? null : status))}
            aria-pressed={filter === status}
            className={`${STATUS_INK[status]} ${
              filter === status ? "underline decoration-2 underline-offset-4" : "hover:underline"
            } ${n === 0 ? "opacity-50" : ""}`}
          >
            <span aria-hidden className="mr-1.5 font-mono">{STATUS_MARK[status]}</span>
            {STATUS_LABEL[status]}: {n}
          </button>
        ))}
        <button
          type="button"
          onClick={() => setFilter((f) => (f === "PENDING" ? null : "PENDING"))}
          aria-pressed={filter === "PENDING"}
          className={`text-ink-3 ${
            filter === "PENDING" ? "underline decoration-2 underline-offset-4" : "hover:underline"
          }`}
        >
          Not checked: {pending.length}
        </button>
        {filter && (
          <button type="button" onClick={() => setFilter(null)} className="link">
            Show all
          </button>
        )}
      </div>

      {busy && (
        <div className="mt-5" data-testid="verification-progress">
          <div className="mb-1.5 flex justify-between text-xs text-ink-2">
            <span className="inline-flex items-center gap-2">
              <Spinner /> Verifying {running}…
            </span>
            <span className="font-mono">
              {done}/{batchSize}
              {queued.length > 0 && ` · ${queued.length} queued`}
            </span>
          </div>
          <div className="h-[3px] bg-rule">
            <div
              className="h-full bg-ink transition-all"
              style={{ width: `${(done / batchSize) * 100}%` }}
            />
          </div>
        </div>
      )}
      {runError && (
        <p className="error-note mt-4" data-testid="verification-run-error">
          Verification failed for {runError}
        </p>
      )}

      <ol className="mt-6 border-t border-rule">
        {shown.map((claim) => (
          <VerdictCard
            key={claim.claim_id}
            analysisId={analysisId}
            claim={claim}
            verdict={verdicts[claim.claim_id]}
            state={
              running === claim.claim_id
                ? "running"
                : queued.includes(claim.claim_id)
                  ? "queued"
                  : "idle"
            }
            disabled={busy || !repositoryBound}
            onVerify={() => verify([claim.claim_id])}
          />
        ))}
      </ol>
    </section>
  );
}

function byClaim(list: Verification[]): Record<string, Verification> {
  return Object.fromEntries(list.map((v) => [v.claim_id, v]));
}

type VerdictCardProps = {
  analysisId: string;
  claim: Claim;
  verdict: Verification | undefined;
  state: "idle" | "running" | "queued";
  disabled: boolean;
  onVerify: () => void;
};

function VerdictCard({
  analysisId,
  claim,
  verdict,
  state,
  disabled,
  onVerify,
}: VerdictCardProps) {
  const [open, setOpen] = useState(false);
  const shownVerdict = state === "idle" ? verdict : undefined;
  return (
    <li
      data-testid="verdict-card"
      className="grid grid-cols-[3rem_minmax(0,1fr)] gap-x-4 border-b border-rule py-5 sm:grid-cols-[3.5rem_minmax(0,1fr)_10rem]"
    >
      <span className="pt-0.5 font-mono text-xs text-ink-3">{claim.claim_id}</span>

      <div className="min-w-0">
        <p
          className={`border-l-2 pl-3 font-serif text-[15px] leading-relaxed ${
            shownVerdict ? STATUS_EDGE[shownVerdict.status] : "border-transparent"
          }`}
        >
          {claim.text}
        </p>
        <p className="mt-1 pl-3.5 text-xs text-ink-3">
          {claim.claim_type.toLowerCase()} · p. {claim.paper_evidence.page}
        </p>

        {shownVerdict && (
          <div className="mt-3 pl-3.5">
            <button
              type="button"
              onClick={() => setOpen((o) => !o)}
              className="text-xs text-ink-2 hover:text-ink"
              data-testid="verdict-toggle"
            >
              {open ? "▾ Hide details" : "▸ Show details"} · {shownVerdict.code_evidence.length}{" "}
              code citation{shownVerdict.code_evidence.length === 1 ? "" : "s"}
              {shownVerdict.discrepancies.length > 0 &&
                ` · ${shownVerdict.discrepancies.length} discrepanc${shownVerdict.discrepancies.length === 1 ? "y" : "ies"}`}
            </button>
            {open && <VerdictDetails analysisId={analysisId} verdict={shownVerdict} />}
          </div>
        )}
      </div>

      <div className="col-start-2 mt-3 flex items-baseline gap-4 pl-3.5 sm:col-start-3 sm:mt-0 sm:flex-col sm:items-end sm:gap-1.5 sm:pl-0">
        {state !== "idle" ? (
          <span className="inline-flex items-center gap-2 text-sm text-accent">
            {state === "running" && <Spinner />}
            {state === "running" ? "Verifying…" : "Queued"}
          </span>
        ) : verdict ? (
          <span
            data-testid="verdict-status"
            className={`text-sm font-medium ${STATUS_INK[verdict.status]}`}
          >
            <span aria-hidden className="mr-1.5 font-mono">{STATUS_MARK[verdict.status]}</span>
            {STATUS_LABEL[verdict.status]}
          </span>
        ) : (
          <span className="text-sm text-ink-3">Not checked</span>
        )}
        <button
          type="button"
          disabled={disabled}
          onClick={onVerify}
          className="text-xs text-accent hover:underline disabled:cursor-not-allowed disabled:text-ink-3 disabled:no-underline"
        >
          {verdict ? "Re-run" : "Verify"}
        </button>
      </div>
    </li>
  );
}

type VerdictDetailsProps = { analysisId: string; verdict: Verification };

function VerdictDetails({ analysisId, verdict }: VerdictDetailsProps) {
  return (
    <div className="mt-4 space-y-6 text-sm text-ink-2" data-testid="verdict-details">
      <p className="whitespace-pre-line leading-relaxed">{verdict.explanation}</p>

      {verdict.discrepancies.length > 0 && (
        <div>
          <h4 className="eyebrow mb-2">Discrepancies</h4>
          <ul className="space-y-3">
            {verdict.discrepancies.map((d, i) => (
              <li key={i} className="border-l-2 border-rule-strong pl-3">
                <p className="text-xs">
                  <span className={`font-medium ${SEVERITY_INK[d.severity] ?? ""}`}>
                    {d.severity.toLowerCase()}
                  </span>
                  <span className="ml-2 font-mono text-ink-3">{d.type}</span>
                </p>
                <p className="mt-1">{d.detail}</p>
                {d.code_quote && (
                  <pre className="mt-2 overflow-x-auto rounded-sm bg-paper px-3 py-2 font-mono text-xs text-ink">
                    {d.code_quote}
                  </pre>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <h4 className="eyebrow mb-2">Paper</h4>
        <CitedPaper analysisId={analysisId} evidence={verdict.paper_evidence} />
      </div>

      <div>
        <h4 className="eyebrow mb-2">Code</h4>
        {verdict.code_evidence.length === 0 ? (
          <p className="text-ink-3">No code cited.</p>
        ) : (
          <ul className="space-y-5">
            {verdict.code_evidence.map((e, i) => (
              <li key={i}>
                <CitedCode analysisId={analysisId} citation={e} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
