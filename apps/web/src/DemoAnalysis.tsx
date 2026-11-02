import { useState } from "react";
import demo from "./data/portfolio-demo.json";
import type { VerificationStatus } from "./api";
import { STATUS_EDGE, STATUS_INK, STATUS_LABEL, STATUS_MARK } from "./verdicts";

export default function DemoAnalysis({ onOpenWorkbench }: { onOpenWorkbench: () => void }) {
  const [selectedId, setSelectedId] = useState(demo.cases[0].id);
  const selected = demo.cases.find((item) => item.id === selectedId) ?? demo.cases[0];

  const recorded = selected.recordedStatus as VerificationStatus;
  const annotated = selected.annotatedStatus as VerificationStatus;

  return (
    <main className="min-h-screen" data-testid="sample-analysis">
      <header className="border-b border-rule">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-baseline gap-3">
            <h1 className="font-serif text-xl font-semibold tracking-tight">Claim2Code</h1>
            <span className="font-mono text-xs text-ink-3">sample · read-only · no model running</span>
          </div>
          <button type="button" onClick={onOpenWorkbench} className="link text-sm">
            Open the workbench
          </button>
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-4 pb-24 pt-8 sm:px-6">
        <section className="max-w-3xl pb-8 pt-6">
          <h2 className="font-serif text-3xl leading-tight tracking-tight sm:text-4xl">
            Recorded claim-to-code results
          </h2>
          <p className="mt-4 text-[15px] leading-relaxed text-ink-2">
            A supported result, a possible configuration discrepancy, and a result with
            insufficient evidence. They are bundled with the page and need no API key,
            model server or upload.
          </p>
          <p className="mt-3 text-xs leading-relaxed text-ink-3" data-testid="sample-scope">
            {demo.scope} A recorded verdict is a model output, not an independent proof.
          </p>
        </section>

        <div className="grid gap-8 border-t border-rule pt-8 md:grid-cols-[14rem_minmax(0,1fr)]">
          <nav aria-label="Sample cases" className="flex flex-col gap-1">
            {demo.cases.map((item) => {
              const active = selectedId === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setSelectedId(item.id)}
                  aria-pressed={active}
                  className={`border-l-2 py-2 pl-3 text-left text-sm ${
                    active ? "border-ink text-ink" : "border-transparent text-ink-3 hover:text-ink"
                  }`}
                >
                  <span className={`block ${active ? "font-medium" : ""}`}>{item.label}</span>
                  <span className="mt-0.5 block font-mono text-xs opacity-70">{item.id}</span>
                </button>
              );
            })}
          </nav>

          <article className="min-w-0" aria-labelledby="sample-claim">
            <p className="eyebrow">Curated claim</p>
            <h3
              id="sample-claim"
              className={`mt-3 border-l-2 pl-4 font-serif text-xl leading-relaxed ${STATUS_EDGE[recorded]}`}
            >
              {selected.claim}
            </h3>
            <p className="mt-3 pl-4 text-sm text-ink-2">
              {selected.paperTitle} ·{" "}
              <a href={selected.paperUrl} target="_blank" rel="noreferrer" className="link">
                arXiv ↗
              </a>{" "}
              ·{" "}
              <a href={selected.repoUrl} target="_blank" rel="noreferrer" className="link">
                repository ↗
              </a>
            </p>

            <dl className="mt-8 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[12rem_minmax(0,1fr)]">
              <dt className="text-ink-3">Recorded model verdict</dt>
              <dd className={`font-medium ${STATUS_INK[recorded]}`} data-testid="sample-verdict">
                <span aria-hidden className="mr-1.5 font-mono">{STATUS_MARK[recorded]}</span>
                {STATUS_LABEL[recorded]}
              </dd>
              <dt className="text-ink-3">Development annotation</dt>
              <dd className={STATUS_INK[annotated]}>
                <span aria-hidden className="mr-1.5 font-mono">{STATUS_MARK[annotated]}</span>
                {STATUS_LABEL[annotated]}
              </dd>
            </dl>
            <p className="mt-4 text-sm leading-relaxed text-ink-2">{selected.note}</p>

            {selected.discrepancies.length > 0 && (
              <section className="mt-8" aria-label="Recorded discrepancies">
                <h4 className="eyebrow mb-3">Possible discrepancies reported by the model</h4>
                <ul className="space-y-3 text-sm leading-relaxed text-ink-2">
                  {selected.discrepancies.map((item, index) => (
                    <li key={index} className="border-l-2 border-verdict-partial pl-3">
                      {item.detail}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <section className="mt-8" aria-label="Recorded code citations">
              <h4 className="eyebrow">Recorded code citations</h4>
              <p className="mt-1 text-xs leading-relaxed text-ink-3">
                Saved quotes from the historical evaluation, not a fresh read of the
                current repository.
              </p>
              {selected.evidence.length === 0 ? (
                <p className="mt-4 text-sm text-ink-2" data-testid="sample-no-evidence">
                  No code evidence was recorded. The result remains Unverified even though
                  the development annotation expects support.
                </p>
              ) : (
                <div className="mt-4 space-y-3">
                  {selected.evidence.map((item, index) => (
                    <details key={`${selected.id}-${index}`} open={index === 0}>
                      <summary className="cursor-pointer font-mono text-xs text-ink-2 hover:text-ink">
                        {item.file_path}:{item.start_line}-{item.end_line}
                      </summary>
                      <pre
                        className="mt-1.5 max-h-80 overflow-auto rounded-sm border border-l-2 border-rule border-l-ink bg-hl/50 px-3 py-2 font-mono text-xs leading-relaxed"
                        data-testid="sample-code-quote"
                      >
                        {item.quote}
                      </pre>
                    </details>
                  ))}
                </div>
              )}
            </section>

            <p className="mt-12 border-t border-rule pt-4 text-xs text-ink-3">
              Source: {demo.source}. The examples keep their recorded verdicts.
            </p>
          </article>
        </div>
      </div>
    </main>
  );
}
