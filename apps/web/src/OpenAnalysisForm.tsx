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
    <form onSubmit={handleSubmit} className="h-fit border-t border-rule pt-5 md:border-l md:border-t-0 md:pl-6 md:pt-6" data-testid="open-analysis">
      <h3 className="font-serif text-base font-medium">Return to an analysis</h3>
      <p className="mt-1 text-sm text-ink-2">
        Paste its id to see the claims, repository and verdicts again.
      </p>
      <div className="mt-4 flex flex-col gap-3">
        <input
          type="text"
          value={analysisId}
          onChange={(e) => setAnalysisId(e.target.value)}
          placeholder="analysis id"
          data-testid="open-analysis-input"
          className="field font-mono"
        />
        <button
          type="submit"
          disabled={!analysisId.trim() || submitting}
          data-testid="open-analysis-button"
          className="btn-quiet"
        >
          {submitting ? "Opening…" : "Open"}
        </button>
      </div>
      {error && (
        <p className="error-note mt-3" data-testid="open-analysis-error">
          {error}
        </p>
      )}
    </form>
  );
}
