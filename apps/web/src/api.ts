// Claim2Code — thin client for the FastAPI backend.
//
// The Vite dev server reads VITE_API_BASE_URL at build time. When the
// frontend is served from the same host as the API (production), this
// can be a relative path; in dev (Vite on :5173, API on :8000) it must
// be an absolute URL. The default in .env.example is the absolute form.

const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

export type Health = {
  status: "ok" | "degraded";
  version?: string;
  db?: boolean;
  error?: string;
};

export type HealthResult =
  | { ok: true; health: Health }
  | { ok: false; error: string };

export type PaperUpload = {
  analysis_id: string;
  paper_id: string;
  parse_status: string;
  n_claims: number;
};

export type PaperUploadResult =
  | { ok: true; upload: PaperUpload }
  | { ok: false; error: string };

export type ClaimType =
  | "METHOD"
  | "IMPLEMENTATION"
  | "PERFORMANCE"
  | "ABLATION"
  | "HYPERPARAMETER"
  | "DATASET"
  | "COMPLEXITY"
  | "TRAINING"
  | "EVALUATION";

export type PaperEvidence = {
  section_title: string;
  page: number;
  char_start: number;
  char_end: number;
  quote: string;
};

export type Claim = {
  claim_id: string;
  text: string;
  claim_type: ClaimType;
  confidence: number;
  paper_evidence: PaperEvidence;
};

export type ClaimsResponse = {
  analysis_id: string;
  paper_status: string;
  claims: Claim[];
};

export type ClaimsResult =
  | { ok: true; claims: ClaimsResponse }
  | { ok: false; error: string };

export async function getHealth(signal?: AbortSignal): Promise<HealthResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { signal });
    if (!response.ok) {
      return { ok: false, error: `HTTP ${response.status}` };
    }
    const health = (await response.json()) as Health;
    return { ok: true, health };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** Upload a PDF to POST /papers. The whole parse+extract runs sync. */
export async function uploadPaper(
  file: File,
  signal?: AbortSignal,
): Promise<PaperUploadResult> {
  try {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${API_BASE_URL}/papers`, {
      method: "POST",
      body: form,
      signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      return { ok: false, error: `HTTP ${response.status}: ${detail.slice(0, 200)}` };
    }
    const upload = (await response.json()) as PaperUpload;
    return { ok: true, upload };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** Fetch the extracted claims for one analysis. */
export async function getClaims(
  analysisId: string,
  signal?: AbortSignal,
): Promise<ClaimsResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/claims`,
      { signal },
    );
    if (!response.ok) {
      return { ok: false, error: `HTTP ${response.status}` };
    }
    const claims = (await response.json()) as ClaimsResponse;
    return { ok: true, claims };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

export { API_BASE_URL };