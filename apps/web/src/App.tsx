import { useEffect, useState } from "react";
import { API_BASE_URL, getHealth, type Health } from "./api";

// Claim2Code — Phase 1 welcome page.
//
// Calls GET /health on mount and renders an online/offline badge so we
// can confirm the frontend can reach the API at the URL advertised by
// VITE_API_BASE_URL. Phase 2 replaces this with the upload + claims UI.
export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal).then((result) => {
      if (result.ok) {
        setHealth(result.health);
        setError(null);
      } else {
        setHealth(null);
        setError(result.error);
      }
    });
    return () => controller.abort();
  }, []);

  const online = health?.status === "ok";

  return (
    <main className="min-h-screen bg-slate-50 font-sans text-slate-900">
      <div className="mx-auto max-w-3xl px-6 py-16">
        <header className="mb-10">
          <h1 className="text-4xl font-semibold tracking-tight">Claim2Code</h1>
          <p className="mt-2 text-slate-600">
            AI Research Agent for Paper–Code Verification and Reproducibility.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-medium">Foundation status</h2>
          <p className="mt-1 text-sm text-slate-600">
            API endpoint:{" "}
            <code className="rounded bg-slate-100 px-1 py-0.5 text-xs">
              {API_BASE_URL}
            </code>
          </p>

          <div className="mt-4 flex items-center gap-3">
            <span
              data-testid="status-badge"
              className={
                online
                  ? "inline-flex items-center gap-2 rounded-full bg-emerald-100 px-3 py-1 text-sm font-medium text-emerald-800"
                  : "inline-flex items-center gap-2 rounded-full bg-rose-100 px-3 py-1 text-sm font-medium text-rose-800"
              }
            >
              <span
                className={
                  online
                    ? "h-2 w-2 rounded-full bg-emerald-500"
                    : "h-2 w-2 rounded-full bg-rose-500"
                }
              />
              {online ? "online" : "offline"}
            </span>
            {health?.version && (
              <span className="text-sm text-slate-500">
                API v{health.version}
              </span>
            )}
          </div>

          {error && (
            <p className="mt-3 text-sm text-rose-700" data-testid="status-error">
              {error}
            </p>
          )}
        </section>

        <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-medium">What ships in Phase 1</h2>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-700">
            <li>FastAPI backend with <code>/health</code> and <code>/health/ready</code> probes</li>
            <li>PostgreSQL + pgvector schema migration <code>0001_init</code></li>
            <li>pytest unit suite + integration scaffold</li>
            <li>This page (Vite + React + Tailwind + Vitest)</li>
            <li>CI: scaffold smoke + backend lint/test + frontend build</li>
          </ul>
          <p className="mt-4 text-sm text-slate-500">
            See <code>docs/architecture/phase-roadmap.md</code> for the full plan.
          </p>
        </section>
      </div>
    </main>
  );
}
