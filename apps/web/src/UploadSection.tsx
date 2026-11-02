import { useState } from "react";
import { uploadPaper } from "./api";
import Spinner from "./Spinner";

type UploadSectionProps = {
  onUploaded: (result: {
    analysisId: string;
    paperId: string;
    nClaims: number;
  }) => void;
};

// UploadSection — Phase 2 PDF upload form.
//
// A drop zone wrapping a native file input (no drag-and-drop library).
// On successful upload the parent's onUploaded callback fires with
// the (analysis_id, paper_id) pair, which the App uses to fetch the
// extracted claims.
export default function UploadSection({ onUploaded }: UploadSectionProps) {
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a PDF first.");
      return;
    }
    setSubmitting(true);
    setError(null);
    const result = await uploadPaper(file);
    setSubmitting(false);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    onUploaded({
      analysisId: result.upload.analysis_id,
      paperId: result.upload.paper_id,
      nClaims: result.upload.n_claims,
    });
  }

  function pick(next: File | null | undefined) {
    if (!next) return;
    if (next.type && next.type !== "application/pdf") {
      setError("That is not a PDF.");
      return;
    }
    setFile(next);
    setError(null);
  }

  return (
    <form onSubmit={handleSubmit} className="panel p-6" data-testid="upload-section">
      <h3 className="font-serif text-lg font-medium">Upload a paper</h3>
      <p className="mt-1 text-sm text-ink-2">
        A digitally generated PDF (no scans), up to 50 MB.
      </p>

      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files?.[0]);
        }}
        className={`mt-5 flex cursor-pointer flex-col items-center justify-center rounded-sm border border-dashed px-4 py-10 text-center transition-colors ${
          dragging ? "border-ink bg-hl/40" : "border-rule-strong hover:border-ink-3"
        }`}
      >
        <input
          type="file"
          accept="application/pdf"
          required
          data-testid="file-input"
          onChange={(e) => pick(e.target.files?.[0])}
          className="sr-only"
        />
        {file ? (
          <>
            <span className="font-mono text-sm">{file.name}</span>
            <span className="mt-1 text-xs text-ink-3">
              {(file.size / 1024 / 1024).toFixed(1)} MB · click to choose another
            </span>
          </>
        ) : (
          <>
            <span className="font-serif text-base italic text-ink-2">Drop the PDF here</span>
            <span className="mt-1 text-xs text-ink-3">or click to choose a file</span>
          </>
        )}
      </label>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs text-ink-3">
          {submitting
            ? "Parsing and extracting claims — a minute or two."
            : "Claims are extracted with the passage each comes from."}
        </span>
        <button
          type="submit"
          disabled={!file || submitting}
          data-testid="upload-button"
          className="btn-primary"
        >
          {submitting && <Spinner />}
          {submitting ? "Extracting…" : "Extract claims"}
        </button>
      </div>

      {error && (
        <p className="error-note mt-4" data-testid="upload-error">
          {error}
        </p>
      )}
    </form>
  );
}
