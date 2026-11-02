import { useState } from "react";
import { getClaims } from "./api";

type OpenAnalysisFormProps = {
  onOpened: (result: { analysisId: string; nClaims: number }) => void;
};

// OpenAnalysisForm — reopen an earlier analysis by its id.
//
// Parsing a paper and indexing a repository each take minutes, so a
// returning user pastes the analysis_id from an earlier upload instead
// of starting over. The id is checked by loading its claims.
export default function OpenAnalysisForm({ onOpened }: OpenAnalysisFormProps) {
  const [analysisId, setAnalysisId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const id = analysisId.trim();
    if (!id) return;
    setSubmitting(true);
    setError(null);
    const result = await getClaims(id);
    setSubmitting(false);
    if (!result.ok) {
      setError(
        result.error === "HTTP 404" || result.error === "HTTP 422"
          ? "No analysis with that id."
          : result.error,
      );
      return;
    }
    onOpened({ analysisId: id, nClaims: result.claims.claims.length });
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="open-analysis"
    >
      <h2 className="text-lg font-medium">Open an earlier analysis</h2>
      <p className="mt-1 text-sm text-slate-600">
        Paste the analysis id returned by an earlier upload to see its claims and
        verdicts without re-parsing.
      </p>
      <div className="mt-4 flex flex-col gap-3 sm:flex-row">
        <input
          type="text"
          value={analysisId}
          onChange={(e) => setAnalysisId(e.target.value)}
          placeholder="analysis id"
          data-testid="open-analysis-input"
          className="w-full rounded-md border border-slate-300 px-3 py-2 font-mono text-sm"
        />
        <button
          type="submit"
          disabled={!analysisId.trim() || submitting}
          data-testid="open-analysis-button"
          className="inline-flex items-center justify-center rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {submitting ? "Opening…" : "Open"}
        </button>
      </div>
      {error && (
        <p className="mt-3 text-sm text-rose-700" data-testid="open-analysis-error">
          {error}
        </p>
      )}
    </form>
  );
}
