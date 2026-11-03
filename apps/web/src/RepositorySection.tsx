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
    <form onSubmit={handleSubmit} className="card p-5" data-testid="repository-section">
      <h2 className="text-[16px] font-medium">Bind the paper's repository</h2>
      <p className="mt-1 text-ink-muted">
        Claims are verified against this code. It is cloned and indexed, never run; public
        GitHub repositories up to 200 MB.
      </p>
      <div className="mt-4 flex max-w-2xl flex-col gap-2 sm:flex-row">
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
        <button type="submit" disabled={submitting} data-testid="repo-clone-button" className="btn-primary h-9">
          {submitting && <Spinner />}
          {submitting ? "Indexing…" : "Clone and index"}
        </button>
      </div>
      {submitting && (
        <p className="mt-2 text-[13px] text-ink-muted">
          Cloning, parsing and embedding the code — a few minutes for a large repository.
        </p>
      )}
      {error && (
        <p className="mt-2 text-[13px] text-negative" data-testid="repo-error">
          {error}
        </p>
      )}
    </form>
  );
}
