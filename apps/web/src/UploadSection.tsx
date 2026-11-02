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
    <form onSubmit={handleSubmit} className="card p-5" data-testid="upload-section">
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
        className={`flex cursor-pointer flex-col items-center justify-center rounded-card border border-dashed px-4 py-8 text-center transition-colors ${
          dragging ? "border-accent bg-accent-wash" : "border-line-control bg-raised hover:border-ink-faint"
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
            <span className="font-medium">{file.name}</span>
            <span className="num mt-1 text-[12px] text-ink-muted">
              {(file.size / 1024 / 1024).toFixed(1)} MB · choose another
            </span>
          </>
        ) : (
          <>
            <span className="font-medium">Drop the paper's PDF here</span>
            <span className="mt-1 text-[12px] text-ink-muted">or click to choose a file</span>
          </>
        )}
      </label>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <span className="text-[13px] text-ink-muted">
          {submitting ? "Reading the paper and extracting claims — a minute or two." : ""}
        </span>
        <button type="submit" disabled={!file || submitting} data-testid="upload-button" className="btn-primary">
          {submitting && <Spinner />}
          {submitting ? "Extracting claims…" : "Extract claims"}
        </button>
      </div>

      {error && (
        <p className="mt-3 text-[13px] text-negative" data-testid="upload-error">
          {error}
        </p>
      )}
    </form>
  );
}
