import { useState } from "react";
import { uploadPaper } from "./api";

type UploadSectionProps = {
  onUploaded: (result: {
    analysisId: string;
    paperId: string;
    nClaims: number;
  }) => void;
};

// UploadSection — Phase 2 PDF upload form.
//
// Native HTML5 file input + Tailwind; no drag-and-drop library in v1.
// On successful upload the parent's onUploaded callback fires with
// the (analysis_id, paper_id) pair, which the App uses to fetch the
// extracted claims.
export default function UploadSection({ onUploaded }: UploadSectionProps) {
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="upload-section"
    >
      <h2 className="text-lg font-medium">Upload a paper</h2>
      <p className="mt-1 text-sm text-slate-600">
        Drop in a PDF — we'll parse it, extract structured claims, and
        list them below. The whole pipeline runs synchronously (max 50 MB).
      </p>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="file"
          accept="application/pdf"
          required
          data-testid="file-input"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="block w-full text-sm text-slate-700 file:mr-3 file:rounded-md file:border-0 file:bg-slate-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-800"
        />
        <button
          type="submit"
          disabled={!file || submitting}
          data-testid="upload-button"
          className="inline-flex items-center justify-center rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {submitting ? "Parsing…" : "Parse & extract"}
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