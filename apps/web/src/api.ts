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

export type VerificationStatus =
  "VERIFIED" | "PARTIALLY_VERIFIED" | "CONTRADICTED" | "UNVERIFIED";

export type CodeEvidence = {
  symbol_id: string;
  file_path: string;
  start_line: number;
  end_line: number;
  quote: string;
};

export type Discrepancy = {
  type: string;
  severity: "LOW" | "MEDIUM" | "HIGH";
  detail: string;
  code_quote: string | null;
};

export type Verification = {
  claim_id: string;
  status: VerificationStatus;
  confidence: number;
  explanation: string;
  paper_evidence: PaperEvidence;
  code_evidence: CodeEvidence[];
  discrepancies: Discrepancy[];
};

export type VerificationsResult =
  { ok: true; verifications: Verification[] } | { ok: false; error: string };

export type ProviderPreset = {
  id: string;
  label: string;
  kind: "openai_compat" | "anthropic";
  base_url: string | null;
  default_model: string;
  models: string[];
  needs_key: boolean;
  key_env: string | null;
  key_configured: boolean;
  json_mode: "schema" | "object";
  local: boolean;
  note: string;
};

export type ProvidersView = {
  active: { provider_id: string; label: string; model: string; base_url: string };
  presets: ProviderPreset[];
  embedding: { model: string; base_url: string; dim: number };
  management?: { enabled: boolean; auth_required: boolean };
};

export type ProviderSelection = {
  provider_id: string;
  model: string;
  base_url: string;
  api_key: string;
};

export type ProviderTest = {
  ok: boolean;
  error: string | null;
  latency_ms: number | null;
  model: string;
  base_url: string;
};

export type ProvidersResult =
  { ok: true; view: ProvidersView } | { ok: false; error: string };

export type ProviderTestResult =
  { ok: true; test: ProviderTest } | { ok: false; error: string };

export type CodeExcerpt = {
  file_path: string;
  start_line: number;
  end_line: number;
  total_lines: number;
  lines: { number: number; text: string }[];
  url: string | null;
};

export type CodeExcerptResult =
  { ok: true; excerpt: CodeExcerpt } | { ok: false; error: string };

export type PaperExcerpt = {
  page: number;
  n_pages: number;
  found: boolean;
  image: string;
  width: number;
  height: number;
};

export type PaperExcerptResult =
  { ok: true; excerpt: PaperExcerpt } | { ok: false; error: string };

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

/** Fetch the repository bound to an analysis.
 *  GET /analyses/{analysis_id}/repository — 404 if none is bound. */
export async function getAnalysisRepository(
  analysisId: string,
  signal?: AbortSignal,
): Promise<RepoMetadataResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/repository`,
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

/** Fetch the stored verdicts for one analysis.
 *  GET /analyses/{analysis_id}/verifications — empty list until a run. */
export async function getVerifications(
  analysisId: string,
  signal?: AbortSignal,
): Promise<VerificationsResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/verifications`,
      { signal },
    );
    if (!response.ok) {
      return { ok: false, error: `HTTP ${response.status}` };
    }
    const json = (await response.json()) as { verifications: Verification[] };
    return { ok: true, verifications: json.verifications };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** Verify the given claims against the analysis's bound repository.
 *
 *  POST /analyses/{analysis_id}/verifications with body {claim_ids}.
 *  Synchronous: one LLM call per claim, so the UI sends one claim per
 *  request and renders each verdict as it lands. The server keeps the
 *  stored verdicts of claims outside the request. */
export async function runVerification(
  analysisId: string,
  claimIds: string[],
  signal?: AbortSignal,
): Promise<VerificationsResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/verifications`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ claim_ids: claimIds }),
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
    const json = (await response.json()) as { verifications: Verification[] };
    return { ok: true, verifications: json.verifications };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

export type ReferenceCheckStatus = "MATCH" | "MISMATCH" | "INCONCLUSIVE" | "ERROR" | "UNTESTABLE";

export type ReferenceCase = {
  index: number;
  inputs: string;
  outcome: "match" | "mismatch" | "target_error" | "reference_error";
  actual: string | null;
  expected: string | null;
  detail: string;
  // Property tests: false when the equality survived perturbing the function.
  sensitive?: boolean | null;
  sensitivity?: string;
};

export type ReferenceCheck = {
  claim_id: string;
  status: ReferenceCheckStatus;
  explanation: string;
  // "property": the claim says two computations agree; check() ran both
  // through the repository's code and there is no reference.
  test_kind: "reference" | "property";
  property_statement: string;
  property_code: string;
  target: {
    symbol_id: string;
    file_path: string;
    start_line: number;
    end_line: number;
    qualname: string;
    signature: string;
  } | null;
  reference_code: string;
  second_reference_code: string;
  confirmed: boolean | null;
  inputs_code: string;
  instance_code: string;
  assumptions: string[];
  rtol: number | null;
  atol: number | null;
  cases: ReferenceCase[];
  isolation: string[];
  repaired?: boolean;
  environment: string[];
  generated_at: string;
};

export type ReferenceChecksResult =
  { ok: true; enabled: boolean; checks: ReferenceCheck[] } | { ok: false; error: string };

export type ReferenceCheckResult =
  { ok: true; check: ReferenceCheck } | { ok: false; error: string };

/** Stored reference checks, and whether this server may run new ones.
 *  GET /analyses/{analysis_id}/reference-checks */
export async function getReferenceChecks(
  analysisId: string,
  signal?: AbortSignal,
): Promise<ReferenceChecksResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/reference-checks`,
      { signal },
    );
    if (!response.ok) {
      return { ok: false, error: `HTTP ${response.status}` };
    }
    const json: unknown = await response.json();
    if (!isObject(json) || typeof json.enabled !== "boolean" || !Array.isArray(json.checks)) {
      return { ok: false, error: "Invalid reference checks response" };
    }
    return { ok: true, enabled: json.enabled, checks: json.checks as ReferenceCheck[] };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "unknown error" };
  }
}

/** Run the cited function next to a reference written from the paper.
 *  POST /analyses/{analysis_id}/reference-checks — two LLM calls and a
 *  sandboxed run, so it takes as long as a verification or longer.
 *  testKind forces a reference or a property test; omitted, the model
 *  picks from the claim. */
export async function runReferenceCheck(
  analysisId: string,
  claimId: string,
  testKind?: "reference" | "property",
  signal?: AbortSignal,
): Promise<ReferenceCheckResult> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/reference-checks`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(testKind ? { claim_id: claimId, test_kind: testKind } : { claim_id: claimId }),
        signal,
      },
    );
    if (!response.ok) {
      return { ok: false, error: await errorDetail(response) };
    }
    return { ok: true, check: (await response.json()) as ReferenceCheck };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "unknown error" };
  }
}

/** Chat-model providers, which have a key, and the active choice.
 *  GET /llm/providers — keys are never returned. */
export async function getProviders(signal?: AbortSignal): Promise<ProvidersResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/llm/providers`, { signal });
    if (!response.ok) {
      return { ok: false, error: `HTTP ${response.status}` };
    }
    const view: unknown = await response.json();
    if (!isObject(view) || !isObject(view.active) || !Array.isArray(view.presets)
      || !isObject(view.embedding) || typeof view.active.model !== "string"
      || typeof view.active.label !== "string") {
      return { ok: false, error: "Invalid model settings response" };
    }
    return { ok: true, view: view as ProvidersView };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

async function postJson(path: string, body: unknown, managementToken?: string): Promise<Response> {
  return fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(managementToken ? { Authorization: `Bearer ${managementToken}` } : {}),
    },
    body: JSON.stringify(body),
  });
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const json = (await response.json()) as { detail?: unknown };
    if (typeof json.detail === "string") return json.detail;
  } catch {
    // fall through to the status line
  }
  return `HTTP ${response.status}`;
}

/** Switch the chat model. POST /llm/provider — the next LLM call uses it. */
export async function setProvider(
  selection: ProviderSelection, managementToken?: string,
): Promise<ProvidersResult> {
  try {
    const response = await postJson("/llm/provider", selection, managementToken);
    if (!response.ok) {
      return { ok: false, error: await errorDetail(response) };
    }
    return { ok: true, view: (await response.json()) as ProvidersView };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** Check a selection reaches its server, without switching to it.
 *  POST /llm/provider/test */
export async function testProvider(
  selection: ProviderSelection, managementToken?: string,
): Promise<ProviderTestResult> {
  try {
    const response = await postJson("/llm/provider/test", selection, managementToken);
    if (!response.ok) {
      return { ok: false, error: await errorDetail(response) };
    }
    return { ok: true, test: (await response.json()) as ProviderTest };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** The code a citation points at, read from the cloned repository.
 *  GET /analyses/{analysis_id}/code?file_path&start_line&end_line&context */
export async function getCodeExcerpt(
  analysisId: string,
  citation: { file_path: string; start_line: number; end_line: number },
  context: number,
  signal?: AbortSignal,
): Promise<CodeExcerptResult> {
  const query = new URLSearchParams({
    file_path: citation.file_path,
    start_line: String(citation.start_line),
    end_line: String(citation.end_line),
    context: String(context),
  });
  try {
    const response = await fetch(
      `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/code?${query}`,
      { signal },
    );
    if (!response.ok) {
      return { ok: false, error: await errorDetail(response) };
    }
    const excerpt: unknown = await response.json();
    if (!isCodeExcerpt(excerpt)) {
      return { ok: false, error: "Invalid code excerpt response" };
    }
    return { ok: true, excerpt };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

/** The uploaded PDF; append `#page=N` to open it at a page. */
export function paperPdfUrl(analysisId: string, page?: number): string {
  const url = `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/paper/pdf`;
  return page ? `${url}#page=${page}` : url;
}

/** The cited region of the paper, rendered with the quote highlighted.
 *  GET /analyses/{analysis_id}/paper/excerpt?page&quote&full */
export async function getPaperExcerpt(
  analysisId: string,
  evidence: { page: number; quote: string },
  full: boolean,
  signal?: AbortSignal,
): Promise<PaperExcerptResult> {
  const query = new URLSearchParams({
    page: String(evidence.page),
    quote: evidence.quote,
    full: String(full),
  });
  try {
    const response = await fetch(
      `${API_BASE_URL}/analyses/${encodeURIComponent(analysisId)}/paper/excerpt?${query}`,
      { signal },
    );
    if (!response.ok) {
      return { ok: false, error: await errorDetail(response) };
    }
    const excerpt: unknown = await response.json();
    if (!isPaperExcerpt(excerpt)) {
      return { ok: false, error: "Invalid paper excerpt response" };
    }
    return { ok: true, excerpt };
  } catch (err) {
    if (err instanceof Error) {
      return { ok: false, error: err.message };
    }
    return { ok: false, error: "unknown error" };
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isCodeExcerpt(value: unknown): value is CodeExcerpt {
  return isObject(value)
    && typeof value.file_path === "string"
    && Number.isInteger(value.start_line) && Number.isInteger(value.end_line)
    && Number.isInteger(value.total_lines)
    && (value.url === null || typeof value.url === "string")
    && Array.isArray(value.lines)
    && value.lines.every((line: unknown) => isObject(line)
      && Number.isInteger(line.number) && typeof line.text === "string");
}

function isPaperExcerpt(value: unknown): value is PaperExcerpt {
  return isObject(value)
    && Number.isInteger(value.page) && Number.isInteger(value.n_pages)
    && typeof value.found === "boolean"
    && typeof value.image === "string" && /^data:image\/(png|jpeg|webp);base64,/.test(value.image)
    && typeof value.width === "number" && value.width > 0
    && typeof value.height === "number" && value.height > 0;
}

export { API_BASE_URL };
