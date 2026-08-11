import { useState } from "react";
import { cloneRepo, type RepoCreate } from "./api";

type RepositorySectionProps = {
  analysisId: string;
  onCloned: (repo: RepoCreate) => void;
};

// RepositorySection — Phase 3 GitHub repo binding.
//
// Form with a single input (GitHub URL) and a "Clone & index" button.
// POST /repositories {url, analysis_id} runs the full clone+walk+parse
// pipeline synchronously; the response carries
// {repository_id, n_files, n_symbols, …}. On success, the parent's
// onCloned callback fires so App can render <SymbolsList/> below.
//
// The status badge mirrors the ``index_status`` from the response:
//   * done    — green "indexed"
//   * error   — red "failed"  (detail in error message)
//   * other   — slate "pending"  (transient; the workflow is sync in v1)
//
// Phase 5+ replaces this with a router-aware /repo route; for now
// the section lives inline under the analysis.
export default function RepositorySection({
  analysisId,
  onCloned,
}: RepositorySectionProps) {
  const [url, setUrl] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) {
      setError("Paste a public repo URL first.");
      return;
    }
    setSubmitting(true);
    setError(null);
    const result = await cloneRepo(analysisId, trimmed);
    setSubmitting(false);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    onCloned(result.repo);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="repository-section"
    >
      <h2 className="text-lg font-medium">Bind a GitHub repo</h2>
      <p className="mt-1 text-sm text-slate-600">
        Drop in a public GitHub URL — we'll clone it, walk every file,
        and extract code symbols + dependencies. The whole pipeline
        runs synchronously (small repos only; max 100 MB).
      </p>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="url"
          inputMode="url"
          required
          placeholder="https://github.com/owner/repo"
          data-testid="repo-url-input"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="block w-full rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm placeholder:text-slate-400 focus:border-slate-900 focus:outline-none focus:ring-1 focus:ring-slate-900"
        />
        <button
          type="submit"
          disabled={submitting}
          data-testid="repo-clone-button"
          className="inline-flex items-center justify-center rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {submitting ? "Cloning…" : "Clone & index"}
        </button>
      </div>

      {error && (
        <p className="mt-3 text-sm text-rose-700" data-testid="repo-error">
          {error}
        </p>
      )}
    </form>
  );
}