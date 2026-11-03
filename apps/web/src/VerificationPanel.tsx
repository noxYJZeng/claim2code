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
import { STATUSES, STATUS_CELL, STATUS_LABEL, STATUS_PILL } from "./verdicts";

type VerificationPanelProps = {
  analysisId: string;
  // Verification needs the analysis's repository; until one is bound
  // the stored verdicts are shown but no new run can start.
  repositoryBound?: boolean;
};

// VerificationPanel — the screen that answers "which of the paper's
// claims does the code back up, and which need a look?".
//
// The claim strip shows every claim in reading order, coloured by its
// stored verdict (GET /analyses/{id}/verifications); its legend counts
// each verdict and filters the list. The list and the selected claim's
// detail sit side by side: the verdict, its explanation and
// discrepancies, then the paper passage and the code it cites.
//
// "Verify all" sends one POST per claim, in order, rather than one
// POST for the whole list: each claim costs an LLM call (minutes on
// a local model), so verdicts render as they land and the user can
// stop between claims. The server keeps the stored verdicts of the
// claims a request does not name.

const SEVERITY_PILL: Record<string, string> = {
  HIGH: "bg-negative-bg text-negative-ink",
  MEDIUM: "bg-warning-bg text-warning-ink",
  LOW: "bg-neutral-bg text-neutral-ink",
};

type Filter = VerificationStatus | "PENDING" | null;

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
  const [filter, setFilter] = useState<Filter>(null);
  const [selected, setSelected] = useState<string | null>(null);
  // On a narrow screen the list and the detail show one at a time.
  const [showDetail, setShowDetail] = useState(false);
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
      <p className="error-block" data-testid="verification-error">
        Couldn't load the verdicts: {loadError}
      </p>
    );
  }

  if (!claims) {
    return (
      <p className="text-ink-muted" data-testid="verification-loading">
        Loading verdicts…
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
  const current =
    claims.find((c) => c.claim_id === selected) ??
    shown.find((c) => verdicts[c.claim_id]) ??
    shown[0];

  function select(claimId: string) {
    setSelected(claimId);
    setShowDetail(true);
  }

  return (
    <section data-testid="verification-panel">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-[22px] font-semibold leading-8">Verification</h1>
          <p className="mt-1 text-ink-muted" data-testid="verification-finding">
            {finding(claims.length, counts, pending.length)}
          </p>
        </div>
        {busy ? (
          <button
            type="button"
            onClick={() => {
              stopRequested.current = true;
            }}
            className="btn-secondary"
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
        <p className="mt-3 text-[13px] text-warning-ink" data-testid="verification-needs-repo">
          Bind the paper's repository above to verify claims.
        </p>
      )}

      {busy && (
        <div className="mt-5 max-w-xl" data-testid="verification-progress">
          <div className="flex justify-between text-[13px]">
            <span className="inline-flex items-center gap-2">
              <Spinner /> Verifying <span className="font-mono">{running}</span>
            </span>
            <span className="num text-ink-muted">
              {done} of {batchSize} done
            </span>
          </div>
          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-active">
            <div
              className="h-full rounded-full bg-accent transition-[width]"
              style={{ width: `${(done / batchSize) * 100}%` }}
            />
          </div>
        </div>
      )}
      {runError && (
        <p className="mt-3 text-[13px] text-negative" data-testid="verification-run-error">
          Verification failed for {runError}
        </p>
      )}

      {claims.length > 0 && (
        <div className="mt-6">
          <div className="flex flex-wrap gap-[3px]" aria-label="Claims by verdict">
            {claims.map((c) => {
              const v = verdicts[c.claim_id];
              const state = running === c.claim_id ? "running" : undefined;
              return (
                <button
                  key={c.claim_id}
                  type="button"
                  onClick={() => select(c.claim_id)}
                  title={`${c.claim_id}: ${v ? STATUS_LABEL[v.status] : "not checked"}`}
                  aria-label={`${c.claim_id}: ${v ? STATUS_LABEL[v.status] : "not checked"}`}
                  className={`h-5 w-5 rounded-[3px] ${
                    v ? STATUS_CELL[v.status] : "border border-line-control bg-sheet"
                  } ${state ? "animate-pulse border-accent" : ""} ${
                    current?.claim_id === c.claim_id ? "ring-2 ring-accent ring-offset-1" : ""
                  }`}
                />
              );
            })}
          </div>
          <div
            className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[13px]"
            data-testid="verification-counts"
          >
            {counts.map(([status, n]) => (
              <LegendButton
                key={status}
                active={filter === status}
                onClick={() => setFilter((f) => (f === status ? null : status))}
                swatch={STATUS_CELL[status]}
                label={STATUS_LABEL[status]}
                n={n}
              />
            ))}
            <LegendButton
              active={filter === "PENDING"}
              onClick={() => setFilter((f) => (f === "PENDING" ? null : "PENDING"))}
              swatch="border border-line-control bg-sheet"
              label="Not checked"
              n={pending.length}
            />
            {filter && (
              <button type="button" onClick={() => setFilter(null)} className="btn-text text-accent">
                Show all
              </button>
            )}
          </div>
        </div>
      )}

      <div className="mt-6 grid items-start gap-6 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
        <ol className={`card overflow-hidden ${showDetail ? "hidden lg:block" : ""}`}>
          {shown.length === 0 && (
            <li className="px-4 py-3 text-ink-muted">No claims with this verdict.</li>
          )}
          {shown.map((claim) => (
            <ClaimRow
              key={claim.claim_id}
              claim={claim}
              verdict={verdicts[claim.claim_id]}
              state={
                running === claim.claim_id
                  ? "running"
                  : queued.includes(claim.claim_id)
                    ? "queued"
                    : "idle"
              }
              selected={current?.claim_id === claim.claim_id}
              onSelect={() => select(claim.claim_id)}
            />
          ))}
        </ol>

        {current && (
          <div className={`lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)] lg:overflow-y-auto ${showDetail ? "" : "hidden lg:block"}`}>
            <button
              type="button"
              onClick={() => setShowDetail(false)}
              className="btn-text mb-3 lg:hidden"
            >
              ← All claims
            </button>
            <ClaimDetail
              key={current.claim_id}
              analysisId={analysisId}
              claim={current}
              verdict={verdicts[current.claim_id]}
              state={
                running === current.claim_id
                  ? "running"
                  : queued.includes(current.claim_id)
                    ? "queued"
                    : "idle"
              }
              disabled={busy || !repositoryBound}
              onVerify={() => verify([current.claim_id])}
            />
          </div>
        )}
      </div>
    </section>
  );
}

function finding(
  total: number,
  counts: readonly (readonly [VerificationStatus, number])[],
  pending: number,
): string {
  if (total === 0) return "The paper has no extracted claims.";
  if (pending === total) {
    return `None of the ${total} claims has been checked against the code yet.`;
  }
  const n = Object.fromEntries(counts) as Record<VerificationStatus, number>;
  const parts = [
    n.CONTRADICTED && `${n.CONTRADICTED} contradicted`,
    n.PARTIALLY_VERIFIED && `${n.PARTIALLY_VERIFIED} partially verified`,
    n.VERIFIED && `${n.VERIFIED} verified`,
    n.UNVERIFIED && `${n.UNVERIFIED} without supporting code`,
    pending && `${pending} not checked`,
  ].filter(Boolean);
  return `Of ${total} claims: ${parts.join(", ")}.`;
}

function LegendButton({
  active,
  onClick,
  swatch,
  label,
  n,
}: {
  active: boolean;
  onClick: () => void;
  swatch: string;
  label: string;
  n: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex items-center gap-1.5 rounded-control px-1.5 py-0.5 ${
        active ? "bg-active text-ink" : n === 0 ? "text-ink-faint" : "text-ink-muted hover:text-ink"
      }`}
    >
      <span aria-hidden className={`h-2.5 w-2.5 rounded-[2px] ${swatch}`} />
      {label} <span className="num text-ink">{n}</span>
    </button>
  );
}

function byClaim(list: Verification[]): Record<string, Verification> {
  return Object.fromEntries(list.map((v) => [v.claim_id, v]));
}

type RowState = "idle" | "running" | "queued";

function VerdictMark({
  verdict,
  state,
  testId,
}: {
  verdict: Verification | undefined;
  state: RowState;
  testId?: string;
}) {
  if (state === "running") {
    return (
      <span className="pill gap-1.5 bg-accent-wash text-accent">
        <Spinner /> Verifying
      </span>
    );
  }
  if (state === "queued") return <span className="pill bg-accent-wash text-accent">Queued</span>;
  if (!verdict) return <span className="text-[12px] text-ink-muted">Not checked</span>;
  return (
    <span data-testid={testId} className={`pill ${STATUS_PILL[verdict.status]}`}>
      {STATUS_LABEL[verdict.status]}
    </span>
  );
}

function ClaimRow({
  claim,
  verdict,
  state,
  selected,
  onSelect,
}: {
  claim: Claim;
  verdict: Verification | undefined;
  state: RowState;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <li data-testid="verdict-card" className="border-b border-line-rule last:border-b-0">
      <button
        type="button"
        onClick={onSelect}
        aria-current={selected ? "true" : undefined}
        className={`block w-full px-4 py-3 text-left ${selected ? "bg-active" : "hover:bg-raised"}`}
      >
        <span className="flex items-center justify-between gap-3">
          <span className="font-mono text-[12px] text-ink-muted">{claim.claim_id}</span>
          <VerdictMark verdict={verdict} state={state} testId="verdict-status" />
        </span>
        <span className="mt-1 line-clamp-2">{claim.text}</span>
      </button>
    </li>
  );
}

type ClaimDetailProps = {
  analysisId: string;
  claim: Claim;
  verdict: Verification | undefined;
  state: RowState;
  disabled: boolean;
  onVerify: () => void;
};

function ClaimDetail({ analysisId, claim, verdict, state, disabled, onVerify }: ClaimDetailProps) {
  const shownVerdict = state === "idle" ? verdict : undefined;
  return (
    <article className="card p-5 sm:p-6" data-testid="verdict-details">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[13px] text-ink-muted">{claim.claim_id}</span>
          <VerdictMark verdict={verdict} state={state} />
        </div>
        <button type="button" disabled={disabled} onClick={onVerify} className="btn-secondary">
          {verdict ? "Re-run" : "Verify"}
        </button>
      </div>
      <p className="mt-3 text-[16px] font-medium leading-6">{claim.text}</p>
      <p className="mt-1 text-[13px] text-ink-muted">
        {sentence(claim.claim_type)} · {claim.paper_evidence.section_title || "Preamble"}, page{" "}
        {claim.paper_evidence.page}
      </p>

      {shownVerdict ? (
        <p className="mt-4 whitespace-pre-line leading-[22px]">{shownVerdict.explanation}</p>
      ) : (
        <p className="mt-4 text-ink-muted">
          {state === "idle"
            ? "Not checked against the code yet."
            : "The verdict appears here when the model returns."}
        </p>
      )}

      {shownVerdict && shownVerdict.discrepancies.length > 0 && (
        <section className="mt-6">
          <h3 className="font-semibold">
            Discrepancies <span className="num font-normal text-ink-muted">{shownVerdict.discrepancies.length}</span>
          </h3>
          <ul className="mt-2 space-y-3">
            {shownVerdict.discrepancies.map((d, i) => (
              <li key={i}>
                <p className="flex flex-wrap items-center gap-2">
                  <span className={`pill ${SEVERITY_PILL[d.severity] ?? SEVERITY_PILL.LOW}`}>
                    {sentence(d.severity)}
                  </span>
                  <span className="text-[13px] text-ink-muted">{sentence(d.type)}</span>
                </p>
                <p className="mt-1">{d.detail}</p>
                {d.code_quote && (
                  <pre className="mt-2 overflow-x-auto rounded-control bg-raised px-3 py-2 font-mono text-[12px]">
                    {d.code_quote}
                  </pre>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="mt-6">
        <h3 className="font-semibold">In the paper</h3>
        <div className="mt-2">
          <CitedPaper analysisId={analysisId} evidence={shownVerdict?.paper_evidence ?? claim.paper_evidence} />
        </div>
      </section>

      {shownVerdict && (
        <section className="mt-6">
          <h3 className="font-semibold">
            In the code{" "}
            <span className="num font-normal text-ink-muted">
              {shownVerdict.code_evidence.length} citation{shownVerdict.code_evidence.length === 1 ? "" : "s"}
            </span>
          </h3>
          {shownVerdict.code_evidence.length === 0 ? (
            <p className="mt-2 text-ink-muted">No code cited.</p>
          ) : (
            <ul className="mt-2 space-y-4">
              {shownVerdict.code_evidence.map((e, i) => (
                <li key={i}>
                  <CitedCode analysisId={analysisId} citation={e} />
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </article>
  );
}

function sentence(value: string): string {
  const s = value.replace(/_/g, " ").toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}
