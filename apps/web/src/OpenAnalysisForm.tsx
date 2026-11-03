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
    <form onSubmit={handleSubmit} className="self-start" data-testid="open-analysis">
      <label htmlFor="open-analysis-id" className="mb-2 block font-medium">
        Analysis id
      </label>
      <div className="flex max-w-xl">
        <input
          id="open-analysis-id"
          type="text"
          value={analysisId}
          onChange={(e) => setAnalysisId(e.target.value)}
          placeholder="f562d476-b08c-…"
          data-testid="open-analysis-input"
          className="field rounded-r-none font-mono"
        />
        <button
          type="submit"
          disabled={!analysisId.trim() || submitting}
          data-testid="open-analysis-button"
          className="btn-secondary h-9 rounded-l-none border-l-0"
        >
          {submitting ? "Opening…" : "Open"}
        </button>
      </div>
      {error && (
        <p className="mt-2 text-[13px] text-negative" data-testid="open-analysis-error">
          {error}
        </p>
      )}
    </form>
  );
}
