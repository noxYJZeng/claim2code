import { useState } from "react";
import demo from "./data/portfolio-demo.json";
import type { VerificationStatus } from "./api";
import { STATUS_LABEL, STATUS_PILL } from "./verdicts";

export default function DemoAnalysis({ onOpenWorkbench }: { onOpenWorkbench: () => void }) {
  const [selectedId, setSelectedId] = useState(demo.cases[0].id);
  const selected = demo.cases.find((item) => item.id === selectedId) ?? demo.cases[0];

  const recorded = selected.recordedStatus as VerificationStatus;
  const annotated = selected.annotatedStatus as VerificationStatus;

  return (
    <div className="min-h-screen" data-testid="sample-analysis">
      <header className="flex h-12 items-center justify-between gap-3 px-4">
        <div className="flex items-center gap-2">
          <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden className="flex-none text-accent">
            <path d="M3 2.5h7l3 3v12H3z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            <path d="M6 9h4M6 12h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <path d="M15 8l2.5 3.5L15 15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="text-[15px] font-semibold">Claim2Code</span>
          <span className="hidden text-ink-muted sm:inline">&nbsp;/&nbsp;&nbsp;Sample analysis</span>
        </div>
        <button type="button" onClick={onOpenWorkbench} className="btn-secondary">
          Open the workbench
        </button>
      </header>

      <main className="mx-2 mb-2 rounded-sheet border border-line bg-sheet px-4 py-6 sm:px-8">
        <div className="mx-auto max-w-6xl">
          <h1 className="text-[22px] font-semibold leading-8">Sample analysis</h1>
          <p className="mt-1 max-w-3xl text-ink-muted">
            Three recorded results: a supported claim, a possible configuration discrepancy and
            a claim without enough evidence. They need no API key, model or upload.
          </p>
          <p className="mt-2 max-w-3xl text-[13px] text-ink-muted" data-testid="sample-scope">
            {demo.scope} A recorded verdict is a model output, not an independent proof.
          </p>

          <div className="mt-6 grid items-start gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
            <nav aria-label="Sample cases" className="card overflow-hidden">
              {demo.cases.map((item) => {
                const active = selectedId === item.id;
                const status = item.recordedStatus as VerificationStatus;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSelectedId(item.id)}
                    aria-pressed={active}
                    className={`block w-full border-b border-line-rule px-4 py-3 text-left last:border-b-0 ${
                      active ? "bg-active" : "hover:bg-raised"
                    }`}
                  >
                    <span className="flex items-center justify-between gap-3">
                      <span className="font-medium">{item.label}</span>
                      <span className={`pill ${STATUS_PILL[status]}`}>{STATUS_LABEL[status]}</span>
                    </span>
                    <span className="mt-0.5 block font-mono text-[12px] text-ink-muted">{item.id}</span>
                  </button>
                );
              })}
            </nav>

            <article className="card p-5 sm:p-6" aria-labelledby="sample-claim">
              <h2 id="sample-claim" className="text-[16px] font-medium leading-6">
                {selected.claim}
              </h2>
              <p className="mt-1 text-[13px] text-ink-muted">
                {selected.paperTitle} ·{" "}
                <a href={selected.paperUrl} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                  arXiv
                </a>{" "}
                ·{" "}
                <a href={selected.repoUrl} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                  repository
                </a>
              </p>

              <dl className="mt-5 grid grid-cols-[11rem_minmax(0,1fr)] gap-y-2">
                <dt className="text-ink-muted">Recorded model verdict</dt>
                <dd data-testid="sample-verdict">
                  <span className={`pill ${STATUS_PILL[recorded]}`}>{STATUS_LABEL[recorded]}</span>
                </dd>
                <dt className="text-ink-muted">Development annotation</dt>
                <dd>
                  <span className={`pill ${STATUS_PILL[annotated]}`}>{STATUS_LABEL[annotated]}</span>
                </dd>
              </dl>
              <p className="mt-4 leading-[22px]">{selected.note}</p>

              {selected.discrepancies.length > 0 && (
                <section className="mt-6" aria-label="Recorded discrepancies">
                  <h3 className="font-semibold">Possible discrepancies reported by the model</h3>
                  <ul className="mt-2 space-y-2">
                    {selected.discrepancies.map((item, index) => (
                      <li key={index} className="leading-[22px]">
                        {item.detail}
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              <section className="mt-6" aria-label="Recorded code citations">
                <h3 className="font-semibold">In the code</h3>
                <p className="mt-0.5 text-[13px] text-ink-muted">
                  Quotes saved from the evaluation run, not a fresh read of the repository.
                </p>
                {selected.evidence.length === 0 ? (
                  <p className="mt-3" data-testid="sample-no-evidence">
                    No code evidence was recorded. The result remains Unverified even though the
                    development annotation expects support.
                  </p>
                ) : (
                  <div className="mt-3 space-y-4">
                    {selected.evidence.map((item, index) => (
                      <div key={`${selected.id}-${index}`}>
                        <p className="font-mono text-[13px]">
                          {item.file_path}:{item.start_line}-{item.end_line}
                        </p>
                        <pre
                          className="mt-1.5 max-h-80 overflow-auto rounded-control border border-line border-l-2 border-l-accent bg-accent-wash px-3 py-2 font-mono text-[12px] leading-5"
                          data-testid="sample-code-quote"
                        >
                          {item.quote}
                        </pre>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <p className="mt-8 text-[12px] text-ink-muted">
                Source: {demo.source}. The examples keep their recorded verdicts.
              </p>
            </article>
          </div>
        </div>
      </main>
    </div>
  );
}
