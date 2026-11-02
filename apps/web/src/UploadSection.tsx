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
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="upload-section"
    >
      <h2 className="text-lg font-medium">Upload a paper</h2>
      <p className="mt-1 text-sm text-slate-600">
        A digitally generated PDF (no scans), up to 50 MB. Its testable claims
        are extracted with the paper passage each one comes from.
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
        className={`mt-4 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors ${
          dragging
            ? "border-slate-900 bg-slate-100"
            : "border-slate-300 bg-slate-50 hover:border-slate-400"
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
            <span className="font-medium text-slate-900">{file.name}</span>
            <span className="mt-1 text-xs text-slate-500">
              {(file.size / 1024 / 1024).toFixed(1)} MB · click to choose another
            </span>
          </>
        ) : (
          <>
            <span className="font-medium text-slate-700">Drop a PDF here</span>
            <span className="mt-1 text-xs text-slate-500">or click to choose a file</span>
          </>
        )}
      </label>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs text-slate-500">
          {submitting && "Parsing and extracting claims — this can take a minute or two."}
        </span>
        <button
          type="submit"
          disabled={!file || submitting}
          data-testid="upload-button"
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {submitting && <Spinner />}
          {submitting ? "Extracting…" : "Extract claims"}
        </button>
      </div>

      {error && (
        <p className="mt-3 text-sm text-rose-700" data-testid="upload-error">
          {error}
        </p>
      )}
    </form>
  );
}
