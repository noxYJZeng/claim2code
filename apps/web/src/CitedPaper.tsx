import { useEffect, useState } from "react";
import {
  getPaperExcerpt,
  paperPdfUrl,
  type PaperEvidence,
  type PaperExcerpt,
} from "./api";

type CitedPaperProps = {
  analysisId: string;
  evidence: PaperEvidence;
};

// CitedPaper — the part of the paper a claim rests on.
//
// Shows the region of the PDF page around the quoted sentence with the
// quote highlighted (GET /analyses/{id}/paper/excerpt), a toggle for
// the whole page, and a link that opens the PDF at that page. The
// quote is the LLM's copy, so the server searches for it piecewise and
// may find it on a neighbouring page; both cases are labelled.

export default function CitedPaper({ analysisId, evidence }: CitedPaperProps) {
  const [full, setFull] = useState(false);
  const [excerpt, setExcerpt] = useState<PaperExcerpt | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getPaperExcerpt(analysisId, evidence, full, controller.signal).then((result) => {
      if (controller.signal.aborted) return;
      if (result.ok) {
        setExcerpt(result.excerpt);
        setError(null);
      } else {
        setError(result.error);
      }
    });
    return () => controller.abort();
  }, [analysisId, evidence, full]);

  const page = excerpt?.page ?? evidence.page;

  return (
    <div data-testid="paper-citation">
      <blockquote className="border-l-2 border-slate-300 pl-3 italic text-slate-600">
        "{evidence.quote}"
      </blockquote>
      <div className="mt-1 flex flex-wrap items-baseline justify-between gap-2 text-[11px]">
        <p className="text-slate-500" data-testid="paper-location">
          {evidence.section_title || "(preamble)"}, page {page}
          {excerpt && excerpt.page !== evidence.page && ` (cited as page ${evidence.page})`}
        </p>
        <div className="flex gap-3">
          {excerpt && (
            <button
              type="button"
              onClick={() => setFull((f) => !f)}
              className="text-slate-500 hover:underline"
              data-testid="paper-toggle-page"
            >
              {full ? "Show the cited part" : "Show whole page"}
            </button>
          )}
          <a
            href={paperPdfUrl(analysisId, page)}
            target="_blank"
            rel="noreferrer"
            className="text-slate-500 hover:underline"
            data-testid="paper-open-pdf"
          >
            Open PDF at page {page} ↗
          </a>
        </div>
      </div>

      {excerpt ? (
        <>
          {!excerpt.found && (
            <p className="mt-1 text-[11px] text-amber-700" data-testid="paper-not-found">
              The quote could not be located in the PDF text; showing the cited page.
            </p>
          )}
          <img
            src={excerpt.image}
            alt={`Page ${excerpt.page} of the paper${excerpt.found ? ", quote highlighted" : ""}`}
            className="mt-1 w-full rounded border border-slate-200 bg-white"
            data-testid="paper-excerpt"
          />
        </>
      ) : error ? (
        <p className="mt-1 text-[11px] text-rose-700" data-testid="paper-excerpt-error">
          Could not render the paper ({error}).
        </p>
      ) : (
        <p className="mt-1 text-[11px] text-slate-400">Loading paper…</p>
      )}
    </div>
  );
}
