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
      className="h-fit rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
      data-testid="open-analysis"
    >
      <h2 className="text-base font-medium">Open an earlier analysis</h2>
      <p className="mt-1 text-sm text-slate-600">
        Paste an analysis id to return to its claims, repository and verdicts.
      </p>
      <div className="mt-4 flex flex-col gap-3">
        <input
          type="text"
          value={analysisId}
          onChange={(e) => setAnalysisId(e.target.value)}
          placeholder="analysis id"
          data-testid="open-analysis-input"
          className="w-full rounded-lg border border-slate-300 px-3 py-2 font-mono text-sm focus:border-slate-900 focus:outline-none focus:ring-1 focus:ring-slate-900"
        />
        <button
          type="submit"
          disabled={!analysisId.trim() || submitting}
          data-testid="open-analysis-button"
          className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-400"
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
