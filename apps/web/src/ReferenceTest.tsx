import { useEffect, useState } from "react";
import {
  getReferenceChecks,
  runReferenceCheck,
  type ReferenceCase,
  type ReferenceCheck,
  type ReferenceCheckStatus,
} from "./api";
import Spinner from "./Spinner";

export type ReferenceChecks = {
  enabled: boolean | null;
  checks: Record<string, ReferenceCheck>;
  running: string | null;
  error: { claimId: string; message: string } | null;
  run: (claimId: string) => void;
};

type ReferenceTestProps = {
  claimId: string;
  repositoryBound: boolean;
  reference: ReferenceChecks;
};

// ReferenceTest — the "does it compute what the paper says?" section
// of a claim's detail.
//
// The server picks the function the claim cites, has the model write
// inputs for it (with the code in view) and a reference implementation
// (from the paper passage and the signature only), runs both in a
// sandboxed process and compares outputs per input. This shows the
// outcome, each input side by side, what the reference had to assume,
// and the isolation that was actually applied. Running is off unless
// the server enables it, because it executes repository code.

const STATUS_TEXT: Record<ReferenceCheckStatus, string> = {
  MATCH: "Same output",
  MISMATCH: "Different output",
  ERROR: "Could not run",
  UNTESTABLE: "Not testable",
};

const STATUS_STYLE: Record<ReferenceCheckStatus, string> = {
  MATCH: "bg-positive-bg text-positive-ink",
  MISMATCH: "bg-negative-bg text-negative-ink",
  ERROR: "bg-warning-bg text-warning-ink",
  UNTESTABLE: "bg-neutral-bg text-neutral-ink",
};

const OUTCOME_TEXT: Record<ReferenceCase["outcome"], string> = {
  match: "Same",
  mismatch: "Differs",
  target_error: "Repository raised",
  reference_error: "Reference raised",
};

/** Stored checks for an analysis and the one in flight. Lives in the
 *  panel, so a test keeps running while another claim is open. */
export function useReferenceChecks(analysisId: string): ReferenceChecks {
  const [checks, setChecks] = useState<Record<string, ReferenceCheck>>({});
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<{ claimId: string; message: string } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setChecks({});
    setEnabled(null);
    getReferenceChecks(analysisId, controller.signal).then((result) => {
      if (controller.signal.aborted || !result.ok) return;
      setEnabled(result.enabled);
      setChecks(Object.fromEntries(result.checks.map((c) => [c.claim_id, c])));
    });
    return () => controller.abort();
  }, [analysisId]);

  async function run(claimId: string) {
    setRunning(claimId);
    setError(null);
    const result = await runReferenceCheck(analysisId, claimId);
    setRunning(null);
    if (result.ok) {
      setChecks((prev) => ({ ...prev, [claimId]: result.check }));
    } else {
      setError({ claimId, message: result.error });
    }
  }

  return { enabled, checks, running, error, run };
}

export default function ReferenceTest({ claimId, repositoryBound, reference }: ReferenceTestProps) {
  const { enabled, checks, running, error } = reference;
  const check = checks[claimId];
  const busy = running === claimId;
  const canRun = enabled === true && repositoryBound && running === null;

  return (
    <section className="mt-6" data-testid="reference-test">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-semibold">Reference test</h3>
        {enabled && (
          <button
            type="button"
            className="btn-secondary"
            disabled={!canRun}
            onClick={() => reference.run(claimId)}
            data-testid="reference-run"
          >
            {check ? "Run again" : "Run reference test"}
          </button>
        )}
      </div>
      <p className="mt-1 text-[13px] text-ink-muted">
        Runs the cited function next to an implementation written from the paper alone, on the
        same inputs, and compares the outputs.
      </p>

      {enabled === false && !check && (
        <p className="mt-2 text-ink-muted" data-testid="reference-disabled">
          Off on this server: it runs repository code. Set{" "}
          <code className="font-mono text-[12px]">REFERENCE_CHECK_ENABLED=true</code> to allow it.
        </p>
      )}
      {enabled && !repositoryBound && !check && (
        <p className="mt-2 text-ink-muted">Bind a repository first.</p>
      )}
      {busy && (
        <p className="mt-3 flex items-center gap-2 text-ink-muted" data-testid="reference-running">
          <Spinner />
          Writing inputs and a reference, then running both. This takes a few minutes on a local
          model.
        </p>
      )}
      {error && error.claimId === claimId && (
        <p className="error-block mt-3" role="alert">{error.message}</p>
      )}
      {check && !busy && <Result check={check} />}
    </section>
  );
}

function Result({ check }: { check: ReferenceCheck }) {
  return (
    <div className="mt-3" data-testid="reference-result">
      <p className="flex flex-wrap items-center gap-2">
        <span className={`pill ${STATUS_STYLE[check.status]}`} data-testid="reference-status">
          {STATUS_TEXT[check.status]}
        </span>
        {check.target && (
          <span className="font-mono text-[12px] text-ink-muted">
            {check.target.qualname} · {check.target.file_path}:{check.target.start_line}
          </span>
        )}
      </p>
      <p className="mt-2 leading-[22px]">{check.explanation}</p>

      {check.cases.length > 0 && (
        <div className="mt-3 overflow-x-auto rounded-card border border-line">
          <table className="w-full text-left text-[13px]">
            <thead className="bg-raised text-ink-muted">
              <tr>
                <th className="px-3 py-2 font-medium">Input</th>
                <th className="px-3 py-2 font-medium">Repository</th>
                <th className="px-3 py-2 font-medium">Reference</th>
                <th className="px-3 py-2 font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {check.cases.map((c) => (
                <tr key={c.index} className="border-t border-line-rule align-top">
                  <td className="max-w-[16rem] break-all px-3 py-2 font-mono text-[12px]">{c.inputs}</td>
                  <td className="max-w-[12rem] break-all px-3 py-2 font-mono text-[12px]">
                    {c.actual ?? (c.outcome === "target_error" ? c.detail : "")}
                  </td>
                  <td className="max-w-[12rem] break-all px-3 py-2 font-mono text-[12px]">
                    {c.expected ?? (c.outcome === "reference_error" ? c.detail : "")}
                  </td>
                  <td className="px-3 py-2">
                    <span className={c.outcome === "match" ? "" : "text-negative-ink"}>
                      {OUTCOME_TEXT[c.outcome]}
                    </span>
                    {c.outcome === "mismatch" && c.detail && (
                      <span className="block text-[12px] text-ink-muted">{c.detail}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {check.assumptions.length > 0 && (
        <div className="mt-3">
          <p className="text-[13px] font-medium">What the reference assumed</p>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-[13px]">
            {check.assumptions.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </div>
      )}

      <Code title="Reference implementation" source={check.reference_code} />
      <Code title="Inputs" source={check.inputs_code} />
      <Code title="Object setup" source={check.instance_code} />

      {check.environment.length > 0 && (
        <p className="mt-3 text-[13px]" data-testid="reference-environment">
          Test environment: {check.environment.join("; ")}. A difference here can come from the
          environment rather than the code.
        </p>
      )}
      {check.isolation.length > 0 && (
        <p className="mt-3 text-[12px] text-ink-muted">Ran with: {check.isolation.join(", ")}.</p>
      )}
    </div>
  );
}

function Code({ title, source }: { title: string; source: string }) {
  if (!source.trim()) return null;
  return (
    <details className="mt-3">
      <summary className="cursor-pointer text-[13px] font-medium">{title}</summary>
      <pre className="mt-2 overflow-x-auto rounded-control bg-raised px-3 py-2 font-mono text-[12px] leading-5">
        {source}
      </pre>
    </details>
  );
}
