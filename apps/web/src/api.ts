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

export { API_BASE_URL };
