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

type VerificationPanelProps = {
  analysisId: string;
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

const STATUSES: VerificationStatus[] = [
  "VERIFIED",
  "PARTIALLY_VERIFIED",
  "CONTRADICTED",
  "UNVERIFIED",
];

const STATUS_LABEL: Record<VerificationStatus, string> = {
  VERIFIED: "Verified",
  PARTIALLY_VERIFIED: "Partially verified",
  CONTRADICTED: "Contradicted",
  UNVERIFIED: "Unverified",
};

const STATUS_STYLE: Record<VerificationStatus, string> = {
  VERIFIED: "bg-emerald-100 text-emerald-800",
  PARTIALLY_VERIFIED: "bg-amber-100 text-amber-800",
  CONTRADICTED: "bg-rose-100 text-rose-800",
  UNVERIFIED: "bg-slate-200 text-slate-700",
};

const SEVERITY_STYLE: Record<string, string> = {
  HIGH: "bg-rose-600 text-white",
  MEDIUM: "bg-amber-500 text-white",
  LOW: "bg-slate-400 text-white",
};

export default function VerificationPanel({ analysisId }: VerificationPanelProps) {
  const [claims, setClaims] = useState<Claim[] | null>(null);
  const [verdicts, setVerdicts] = useState<Record<string, Verification>>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  // Claim currently being verified, and the queue still waiting.
  const [running, setRunning] = useState<string | null>(null);
  const [queued, setQueued] = useState<string[]>([]);
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
      <section
        className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800"
        data-testid="verification-error"
      >
        Failed to load verifications: {loadError}
      </section>
    );
  }

  if (!claims) {
    return (
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-500"
        data-testid="verification-loading"
      >
        Loading verifications…
      </section>
    );
  }

  const busy = running !== null;
  const pending = claims.filter((c) => !verdicts[c.claim_id]).map((c) => c.claim_id);
  const counts = STATUSES.map(
    (s) => [s, Object.values(verdicts).filter((v) => v.status === s).length] as const,
  );

  return (
    <section
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="verification-panel"
    >
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="text-lg font-medium">Verification</h2>
        <div className="flex gap-2">
          {busy ? (
            <button
              type="button"
              onClick={() => {
                stopRequested.current = true;
              }}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
              data-testid="verify-stop"
            >
              Stop after this claim
            </button>
          ) : (
            <button
              type="button"
              disabled={pending.length === 0}
              onClick={() => verify(pending)}
              className="rounded-lg bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
              data-testid="verify-all"
            >
              Verify {pending.length} unverified claim{pending.length === 1 ? "" : "s"}
            </button>
          )}
        </div>
      </div>
      <p className="mb-4 text-xs text-slate-500">
        Each claim is checked against the repository bound to this analysis, one LLM
        call per claim (a few minutes each on a local model). Every verdict cites the
        code it rests on; UNVERIFIED means no supporting code was found, not that the
        code lacks it.
      </p>

      <div className="mb-4 flex flex-wrap gap-2 text-xs" data-testid="verification-counts">
        {counts.map(([status, n]) => (
          <span
            key={status}
            className={`rounded-full px-2.5 py-1 font-medium ${STATUS_STYLE[status]}`}
          >
            {STATUS_LABEL[status]}: {n}
          </span>
        ))}
        <span className="rounded-full bg-white px-2.5 py-1 font-medium text-slate-500 ring-1 ring-slate-200">
          Not checked: {pending.length}
        </span>
      </div>

      {busy && (
        <p className="mb-3 text-sm text-slate-600" data-testid="verification-progress">
          Verifying {running}… {queued.length > 0 && `${queued.length} more queued.`}
        </p>
      )}
      {runError && (
        <p className="mb-3 text-sm text-rose-700" data-testid="verification-run-error">
          Verification failed for {runError}
        </p>
      )}

      <ul className="space-y-3">
        {claims.map((claim) => (
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
            disabled={busy}
            onVerify={() => verify([claim.claim_id])}
          />
        ))}
      </ul>
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
  return (
    <li
      data-testid="verdict-card"
      className="rounded-xl border border-slate-200 bg-slate-50 p-4"
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-medium text-slate-900">
          <span className="mr-2 text-xs text-slate-400">{claim.claim_id}</span>
          {claim.text}
        </p>
        <div className="flex shrink-0 flex-col items-end gap-2">
          {state !== "idle" ? (
            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-800">
              {state === "running" ? "Verifying…" : "Queued"}
            </span>
          ) : verdict ? (
            <span
              data-testid="verdict-status"
              className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLE[verdict.status]}`}
            >
              {STATUS_LABEL[verdict.status]}
            </span>
          ) : (
            <span className="text-xs text-slate-400">Not checked</span>
          )}
          <button
            type="button"
            disabled={disabled}
            onClick={onVerify}
            className="text-xs font-medium text-slate-600 hover:underline disabled:cursor-not-allowed disabled:text-slate-300"
          >
            {verdict ? "Re-run" : "Verify"}
          </button>
        </div>
      </div>

      {verdict && state === "idle" && (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="text-xs font-medium uppercase tracking-wide text-slate-500 hover:underline"
            data-testid="verdict-toggle"
          >
            {open ? "Hide details" : "Show details"} · {verdict.code_evidence.length} code
            citation{verdict.code_evidence.length === 1 ? "" : "s"}
            {verdict.discrepancies.length > 0 &&
              ` · ${verdict.discrepancies.length} discrepanc${verdict.discrepancies.length === 1 ? "y" : "ies"}`}
          </button>
          {open && <VerdictDetails analysisId={analysisId} verdict={verdict} />}
        </div>
      )}
    </li>
  );
}

type VerdictDetailsProps = { analysisId: string; verdict: Verification };

function VerdictDetails({ analysisId, verdict }: VerdictDetailsProps) {
  return (
    <div className="mt-3 space-y-4 text-xs text-slate-700" data-testid="verdict-details">
      <p className="whitespace-pre-line">{verdict.explanation}</p>

      {verdict.discrepancies.length > 0 && (
        <div>
          <h3 className="mb-1 font-medium uppercase tracking-wide text-slate-500">
            Discrepancies
          </h3>
          <ul className="space-y-2">
            {verdict.discrepancies.map((d, i) => (
              <li key={i} className="rounded-lg border border-slate-200 bg-white p-2">
                <div className="mb-1 flex gap-2">
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${SEVERITY_STYLE[d.severity] ?? ""}`}
                  >
                    {d.severity}
                  </span>
                  <span className="font-mono text-[11px] text-slate-500">{d.type}</span>
                </div>
                <p>{d.detail}</p>
                {d.code_quote && (
                  <pre className="mt-1 overflow-x-auto rounded bg-slate-100 p-2 font-mono text-[11px]">
                    {d.code_quote}
                  </pre>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <h3 className="mb-1 font-medium uppercase tracking-wide text-slate-500">Paper</h3>
        <CitedPaper analysisId={analysisId} evidence={verdict.paper_evidence} />
      </div>

      <div>
        <h3 className="mb-1 font-medium uppercase tracking-wide text-slate-500">Code</h3>
        {verdict.code_evidence.length === 0 ? (
          <p className="text-slate-500">No code cited.</p>
        ) : (
          <ul className="space-y-3">
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
