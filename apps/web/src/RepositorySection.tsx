import { useState } from "react";
import { cloneRepo, type RepoCreate } from "./api";
import Spinner from "./Spinner";

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
// onCloned callback fires so App can show the repository and its tabs.
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
    <form onSubmit={handleSubmit} className="panel p-6" data-testid="repository-section">
      <h3 className="font-serif text-lg font-medium">Bind the paper's repository</h3>
      <p className="mt-1 text-sm text-ink-2">
        A public GitHub URL. The code is cloned and indexed for retrieval, never
        executed. Up to 200 MB.
      </p>

      <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="url"
          inputMode="url"
          required
          placeholder="https://github.com/owner/repo"
          data-testid="repo-url-input"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="field font-mono"
        />
        <button
          type="submit"
          disabled={submitting}
          data-testid="repo-clone-button"
          className="btn-primary shrink-0"
        >
          {submitting && <Spinner />}
          {submitting ? "Indexing…" : "Clone & index"}
        </button>
      </div>
      {submitting && (
        <p className="mt-3 text-xs text-ink-3">
          Cloning, parsing and embedding the code — a few minutes for a large repository.
        </p>
      )}

      {error && (
        <p className="error-note mt-4" data-testid="repo-error">
          {error}
        </p>
      )}
    </form>
  );
}
