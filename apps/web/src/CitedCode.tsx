import { useEffect, useState } from "react";
import { getCodeExcerpt, type CodeEvidence, type CodeExcerpt } from "./api";

type CitedCodeProps = {
  analysisId: string;
  citation: CodeEvidence;
};

// CitedCode — the code one verdict citation points at.
//
// Loads the cited lines from the cloned repository (GET
// /analyses/{id}/code) and shows them with line numbers, the cited
// range highlighted and a few lines of context, plus a link to the
// same lines on GitHub at the cloned commit. If the file cannot be
// read, it falls back to the LLM's own quote, labelled as such.

const CONTEXT = 3;
const WIDE_CONTEXT = 15;

export default function CitedCode({ analysisId, citation }: CitedCodeProps) {
  const [context, setContext] = useState(CONTEXT);
  const [excerpt, setExcerpt] = useState<CodeExcerpt | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setExcerpt(null);
    setError(null);
    getCodeExcerpt(analysisId, citation, context, controller.signal).then((result) => {
      if (controller.signal.aborted) return;
      if (result.ok) {
        setExcerpt(result.excerpt);
        setError(null);
      } else {
        setError(result.error);
      }
    });
    return () => controller.abort();
  }, [analysisId, citation, context]);

  const range =
    citation.end_line !== citation.start_line
      ? `${citation.start_line}-${citation.end_line}`
      : `${citation.start_line}`;

  return (
    <div data-testid="code-citation">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="font-mono text-[11px] text-slate-600">
          {excerpt?.file_path ?? citation.file_path}:{range}
        </p>
        <div className="flex gap-3 text-[11px]">
          {excerpt && excerpt.lines.length < excerpt.total_lines && context === CONTEXT && (
            <button
              type="button"
              onClick={() => setContext(WIDE_CONTEXT)}
              className="text-slate-500 hover:underline"
              data-testid="code-more-context"
            >
              More context
            </button>
          )}
          {excerpt?.url && (
            <a
              href={excerpt.url}
              target="_blank"
              rel="noreferrer"
              className="text-slate-500 hover:underline"
            >
              View on GitHub ↗
            </a>
          )}
        </div>
      </div>

      {excerpt ? (
        <pre
          className="mt-1 max-h-96 overflow-auto rounded bg-slate-900 py-2 font-mono text-[11px] leading-relaxed text-slate-100"
          data-testid="code-excerpt"
        >
          {excerpt.lines.map((line) => {
            const cited = line.number >= excerpt.start_line && line.number <= excerpt.end_line;
            return (
              <div
                key={line.number}
                className={cited ? "bg-amber-400/20" : "opacity-60"}
                data-cited={cited || undefined}
              >
                <span className="inline-block w-12 select-none pr-3 text-right text-slate-500">
                  {line.number}
                </span>
                {line.text || " "}
              </div>
            );
          })}
        </pre>
      ) : error ? (
        <div className="mt-1" data-testid="code-excerpt-error">
          <p className="text-[11px] text-rose-700">
            Could not load the file ({error}); showing the model's quote instead.
          </p>
          <pre className="mt-1 overflow-x-auto rounded bg-slate-100 p-2 font-mono text-[11px]">
            {citation.quote}
          </pre>
        </div>
      ) : (
        <p className="mt-1 text-[11px] text-slate-400">Loading code…</p>
      )}
    </div>
  );
}
