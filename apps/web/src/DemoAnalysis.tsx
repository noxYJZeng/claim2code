import { useState } from "react";
import demo from "./data/portfolio-demo.json";

const statusLabels: Record<string, string> = {
  VERIFIED: "Verified",
  PARTIALLY_VERIFIED: "Partially verified",
  CONTRADICTED: "Contradicted",
  UNVERIFIED: "Unverified",
};

export default function DemoAnalysis({ onOpenWorkbench }: { onOpenWorkbench: () => void }) {
  const [selectedId, setSelectedId] = useState(demo.cases[0].id);
  const selected = demo.cases.find((item) => item.id === selectedId) ?? demo.cases[0];

  return (
    <main className="min-h-screen bg-slate-50 font-sans text-slate-900" data-testid="sample-analysis">
      <div className="mx-auto max-w-5xl px-6 py-12">
        <header>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h1 className="text-3xl font-semibold tracking-tight">Claim2Code</h1>
            <span className="rounded-full bg-slate-200 px-3 py-1 text-xs font-medium text-slate-700">
              Read-only sample · no model running
            </span>
          </div>
          <h2 className="mt-8 text-2xl font-semibold">Inspect recorded claim-to-code results</h2>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-600">
            Explore a supported result, a potential configuration discrepancy, and a result
            with insufficient evidence. These examples are bundled with the site and work
            without an API key, model server, or uploaded files.
          </p>
          <p className="mt-3 max-w-3xl text-xs leading-relaxed text-slate-500" data-testid="sample-scope">
            {demo.scope} A recorded verdict is a model output, not an independent proof.
          </p>
        </header>

        <div className="mt-8 grid gap-5 md:grid-cols-[15rem_minmax(0,1fr)]">
          <nav aria-label="Sample cases" className="space-y-2">
            {demo.cases.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setSelectedId(item.id)}
                aria-pressed={selectedId === item.id}
                className={`w-full rounded-xl border px-4 py-3 text-left text-sm ${
                  selectedId === item.id
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-200 bg-white text-slate-700 hover:border-slate-400"
                }`}
              >
                <span className="block font-medium">{item.label}</span>
                <span className="mt-1 block text-xs opacity-70">{item.id}</span>
              </button>
            ))}
          </nav>

          <article className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-labelledby="sample-claim">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Manually curated claim
            </p>
            <h3 id="sample-claim" className="mt-2 text-lg font-medium leading-relaxed">
              {selected.claim}
            </h3>
            <p className="mt-3 text-sm text-slate-500">{selected.paperTitle}</p>
            <div className="mt-3 flex flex-wrap gap-4 text-xs">
              <a href={selected.paperUrl} target="_blank" rel="noreferrer" className="text-slate-600 underline">
                Paper on arXiv ↗
              </a>
              <a href={selected.repoUrl} target="_blank" rel="noreferrer" className="text-slate-600 underline">
                Reference repository ↗
              </a>
            </div>

            <dl className="mt-6 grid gap-3 rounded-xl bg-slate-50 p-4 text-sm sm:grid-cols-2">
              <div>
                <dt className="text-xs text-slate-500">Recorded model verdict</dt>
                <dd className="mt-1 font-semibold" data-testid="sample-verdict">
                  {statusLabels[selected.recordedStatus]}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Development annotation</dt>
                <dd className="mt-1 font-medium">{statusLabels[selected.annotatedStatus]}</dd>
              </div>
            </dl>
            <p className="mt-3 text-sm leading-relaxed text-slate-600">{selected.note}</p>

            {selected.discrepancies.length > 0 && (
              <section className="mt-6" aria-label="Recorded discrepancies">
                <h4 className="text-sm font-semibold">Potential discrepancies reported by the model</h4>
                <ul className="mt-2 space-y-2 text-sm leading-relaxed text-slate-600">
                  {selected.discrepancies.map((item, index) => (
                    <li key={index} className="rounded-lg border border-amber-200 bg-amber-50 p-3">
                      {item.detail}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <section className="mt-6" aria-label="Recorded code citations">
              <h4 className="text-sm font-semibold">Recorded code citations</h4>
              <p className="mt-1 text-xs leading-relaxed text-slate-500">
                Saved quotes from the historical evaluation, not a fresh read of the current
                repository. Source revisions and locations should be checked before reuse.
              </p>
              {selected.evidence.length === 0 ? (
                <p className="mt-3 rounded-lg border border-slate-200 p-4 text-sm text-slate-600" data-testid="sample-no-evidence">
                  No code evidence was recorded. The result remains Unverified even though
                  the development annotation expects support.
                </p>
              ) : (
                <div className="mt-3 space-y-3">
                  {selected.evidence.map((item, index) => (
                    <details key={`${selected.id}-${index}`} className="rounded-lg border border-slate-200">
                      <summary className="cursor-pointer px-3 py-2 font-mono text-xs text-slate-600">
                        {item.file_path}:{item.start_line}-{item.end_line}
                      </summary>
                      <pre className="max-h-80 overflow-auto rounded-b-lg bg-slate-900 p-3 text-xs leading-relaxed text-slate-100" data-testid="sample-code-quote">
                        {item.quote}
                      </pre>
                    </details>
                  ))}
                </div>
              )}
            </section>
          </article>
        </div>

        <footer className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-5 text-xs text-slate-500">
          <p>Source: {demo.source}. The examples retain their recorded verdicts.</p>
          <button type="button" onClick={onOpenWorkbench} className="font-medium text-slate-700 underline">
            Open connected workbench
          </button>
        </footer>
      </div>
    </main>
  );
}
