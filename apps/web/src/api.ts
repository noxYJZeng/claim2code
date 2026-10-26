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
  { ok: true; health: Health } | { ok: false; error: string };

export type PaperUpload = {
  analysis_id: string;
  paper_id: string;
  parse_status: string;
  n_claims: number;
};

export type PaperUploadResult =
  { ok: true; upload: PaperUpload } | { ok: false; error: string };

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
  { ok: true; claims: ClaimsResponse } | { ok: false; error: string };

export type CodeSymbolKind =
  "function" | "method" | "class" | "import" | "constant" | "config_schema";

export type CodeLanguage =
  "python" | "javascript" | "typescript" | "java" | "go" | "rust" | "shell" | "config";

export type CodeSymbol = {
  kind: CodeSymbolKind;
  language: CodeLanguage;
  file_path: string;
  start_line: number;
  start_col: number;
  end_line: number;
  end_col: number;
  name: string;
  signature: string;
  docstring: string | null;
};

export type RepoCreate = {
  repository_id: string;
  analysis_id: string;
  url: string;
  owner: string;
  name: string;
  n_files: number;
  n_symbols: number;
  n_configs: number;
  n_dependencies: number;
  index_status: string;
};

export type RepoCreateResult =
  { ok: true; repo: RepoCreate } | { ok: false; error: string };

export type RepoMetadata = {
  repository_id: string;
  analysis_id: string;
  url: string;
  owner: string;
  name: string;
  default_branch: string;
  clone_path: string;
  n_files: number | null;
  n_symbols: number | null;
  index_status: string;
  cloned_at: string;
  indexed_at: string | null;
  error: string | null;
};

export type RepoMetadataResult =
  { ok: true; metadata: RepoMetadata } | { ok: false; error: string };

export type SymbolsResponse = {
  repository_id: string;
  symbols: CodeSymbol[];
};

export type SymbolsResult =
  { ok: true; response: SymbolsResponse } | { ok: false; error: string };

export type RetrieveMatch = {
  symbol_id: string;
  kind: CodeSymbolKind;
  language: CodeLanguage;
  file_path: string;
  start_line: number;
  start_col: number;
  end_line: number;
  end_col: number;
  name: string;
  signature: string;
  docstring: string | null;
  similarity: number;
};

export type RetrieveResponse = {
  repository_id: string;
  claim: string;
  symbols: RetrieveMatch[];
  scores: number[];
};

export type RetrieveResult =
  { ok: true; response: RetrieveResponse } | { ok: false; error: string };

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
      return {
        ok: false,
        error: `HTTP ${response.status}: ${detail.slice(0, 200)}`,
      };
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

/** Clone + index a public GitHub repo, bound to an analysis.
 *
 *  POST /repositories with body {url, analysis_id}. The whole pipeline
 *  runs synchronously; the response carries n_files / n_symbols /
 *  n_configs / n_dependencies once the index is done.
 */
export async function cloneRepo(
  analysisId: string,
  url: string,
  signal?: AbortSignal,
): Promise<RepoCreateResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/repositories`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ url, analysis_id: analysisId }),
      signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      return {
        ok: false,
        error: `HTTP ${response.status}: ${detail.slice(0, 200)}`,
      };
    }
    const repo = (await response.json()) as RepoCreate;
    return { ok: true, repo };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** Fetch metadata for one cloned repository.
 *  GET /repositories/{repository_id} — 404 if the row is missing. */
export async function getRepository(
  repositoryId: string,
  signal?: AbortSignal,
): Promise<RepoMetadataResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/repositories/${encodeURIComponent(repositoryId)}`,
      { signal },
    );
    if (!response.ok) {
      return { ok: false, error: `HTTP ${response.status}` };
    }
    const metadata = (await response.json()) as RepoMetadata;
    return { ok: true, metadata };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** Fetch the extracted code symbols for one repository.
 *  GET /repositories/{repository_id}/symbols */
export async function getSymbols(
  repositoryId: string,
  signal?: AbortSignal,
): Promise<SymbolsResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/repositories/${encodeURIComponent(repositoryId)}/symbols`,
      { signal },
    );
    if (!response.ok) {
      return { ok: false, error: `HTTP ${response.status}` };
    }
    const json = (await response.json()) as SymbolsResponse;
    return { ok: true, response: json };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** Embed a claim string and return the top-k matching symbols by
 *  cosine similarity against the repository's code_embeddings index.
 *
 *  POST /repositories/{repository_id}/retrieve
 *  body: {claim, top_k}  (top_k defaults to 5 server-side; pass
 *  explicitly if you want a different cap). */
export async function retrieveSymbols(
  repositoryId: string,
  claim: string,
  topK: number = 5,
  signal?: AbortSignal,
): Promise<RetrieveResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/repositories/${encodeURIComponent(repositoryId)}/retrieve`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ claim, top_k: topK }),
        signal,
      },
    );
    if (!response.ok) {
      const detail = await response.text();
      return {
        ok: false,
        error: `HTTP ${response.status}: ${detail.slice(0, 200)}`,
      };
    }
    const json = (await response.json()) as RetrieveResponse;
    return { ok: true, response: json };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

export { API_BASE_URL };
