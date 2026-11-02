import { useEffect, useState } from "react";
import {
  getProviders,
  setProvider,
  testProvider,
  type ProviderPreset,
  type ProviderSelection,
  type ProvidersView,
} from "./api";

// ModelProviderPanel — choose the chat model.
//
// Claim extraction, verification and related-work comparison all use
// one chat model; this panel shows which, and lets the user switch to
// another provider (local LM Studio / Ollama, Anthropic, OpenAI,
// Gemini, DeepSeek, OpenRouter, or any OpenAI-compatible server).
// "Test connection" checks a choice without switching; "Use this
// model" switches, and the next LLM call uses it.
//
// An API key typed here is sent to the local API, kept in its memory
// only (lost on restart), and never sent back. Keys meant to persist
// belong in .env. Embeddings are shown but not switchable: every
// stored vector came from one embedding model, so changing it means
// re-indexing.

type TestState =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "ok"; latencyMs: number | null }
  | { kind: "failed"; error: string };

export default function ModelProviderPanel() {
  const [view, setView] = useState<ProvidersView | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<ProviderSelection | null>(null);
  const [test, setTest] = useState<TestState>({ kind: "idle" });
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [managementToken, setManagementToken] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    getProviders(controller.signal).then((result) => {
      if (result.ok) {
        setView(result.view);
        setLoadError(null);
      } else {
        setLoadError(result.error);
      }
    });
    return () => controller.abort();
  }, []);

  if (loadError) {
    return (
      <p className="mt-4 text-sm text-verdict-bad" data-testid="provider-error">
        Could not load chat-model settings: {loadError}
      </p>
    );
  }
  if (!view) return null;

  const presets = Object.fromEntries(view.presets.map((p) => [p.id, p]));
  const preset: ProviderPreset | undefined = form ? presets[form.provider_id] : undefined;
  const managementEnabled = view.management?.enabled === true;
  const authRequired = view.management?.auth_required === true;
  const fixedEndpoint = preset && !preset.local && preset.id !== "custom";

  function startEditing() {
    if (!view) return;
    setForm({
      provider_id: view.active.provider_id,
      model: view.active.model,
      base_url: view.active.base_url,
      api_key: "",
    });
    setTest({ kind: "idle" });
    setSaveError(null);
    setOpen(true);
  }

  function pickPreset(id: string) {
    const p = presets[id];
    setForm({ provider_id: id, model: p.default_model, base_url: p.base_url ?? "", api_key: "" });
    setTest({ kind: "idle" });
    setSaveError(null);
  }

  function update(patch: Partial<ProviderSelection>) {
    setForm((f) => (f ? { ...f, ...patch } : f));
    setTest({ kind: "idle" });
  }

  async function runTest() {
    if (!form) return;
    setTest({ kind: "running" });
    const result = await testProvider(form, managementToken);
    if (!result.ok) setTest({ kind: "failed", error: result.error });
    else if (!result.test.ok) setTest({ kind: "failed", error: result.test.error ?? "failed" });
    else setTest({ kind: "ok", latencyMs: result.test.latency_ms });
  }

  async function save() {
    if (!form) return;
    setSaving(true);
    setSaveError(null);
    const result = await setProvider(form, managementToken);
    setSaving(false);
    if (!result.ok) {
      setSaveError(result.error);
      return;
    }
    setView(result.view);
    setOpen(false);
    setForm(null);
    setManagementToken("");
  }

  return (
    <section
      className="panel p-5 text-sm"
      data-testid="provider-panel"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <span className="eyebrow">Chat model</span>
          <p className="mt-1 font-medium" data-testid="provider-active">
            {view.active.label} · <span className="font-mono">{view.active.model}</span>
          </p>
          <p className="mt-1 text-xs text-ink-3">
            Embeddings: <span className="font-mono">{view.embedding.model}</span> (fixed in
            .env — changing it means re-indexing repositories)
          </p>
        </div>
        {!open && managementEnabled && (
          <button
            type="button"
            onClick={startEditing}
            className="btn-quiet"
            data-testid="provider-change"
          >
            Change
          </button>
        )}
      </div>

      {!managementEnabled && (
        <p className="mt-3 text-xs text-ink-3" data-testid="provider-read-only">
          Model settings are read-only on this server.
        </p>
      )}

      {managementEnabled && open && form && preset && (
        <div className="mt-4 space-y-4 border-t border-rule pt-4" data-testid="provider-form">
          {authRequired && (
            <label className="block">
              <span className="eyebrow">Management token</span>
              <input
                type="password"
                autoComplete="off"
                value={managementToken}
                onChange={(e) => setManagementToken(e.target.value)}
                className="field mt-1 font-mono"
                data-testid="provider-management-token"
              />
              <span className="mt-1 block text-xs text-ink-3">
                Required by this server for model changes. Kept in memory until this form closes.
              </span>
            </label>
          )}
          <label className="block">
            <span className="eyebrow">Provider</span>
            <select
              value={form.provider_id}
              onChange={(e) => pickPreset(e.target.value)}
              className="field mt-1"
              data-testid="provider-select"
            >
              {view.presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                  {p.needs_key ? (p.key_configured ? " · key set" : " · needs key") : ""}
                </option>
              ))}
            </select>
          </label>
          {preset.note && <p className="text-xs text-ink-3">{preset.note}</p>}

          <label className="block">
            <span className="eyebrow">Model</span>
            <input
              type="text"
              list="provider-models"
              value={form.model}
              onChange={(e) => update({ model: e.target.value })}
              placeholder="model name"
              className="field mt-1 font-mono"
              data-testid="provider-model"
            />
            <datalist id="provider-models">
              {preset.models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>

          <label className="block">
            <span className="eyebrow">Base URL</span>
            <input
              type="text"
              value={form.base_url}
              readOnly={fixedEndpoint}
              onChange={(e) => update({ base_url: e.target.value })}
              placeholder="https://host/v1"
              className="field mt-1 font-mono"
              data-testid="provider-base-url"
            />
            {fixedEndpoint && (
              <span className="mt-1 block text-xs text-ink-3">
                This provider uses its configured endpoint.
              </span>
            )}
          </label>

          {(preset.needs_key || preset.id === "custom") && (
            <label className="block">
              <span className="eyebrow">API key</span>
              <input
                type="password"
                autoComplete="off"
                value={form.api_key}
                onChange={(e) => update({ api_key: e.target.value })}
                placeholder={
                  preset.key_configured
                    ? "configured — leave blank to keep it"
                    : preset.key_env
                      ? `or set ${preset.key_env} in .env`
                      : "if the server needs one"
                }
                className="field mt-1 font-mono"
                data-testid="provider-key"
              />
              <span className="mt-1 block text-xs text-ink-3">
                Sent to the local API and kept in its memory only; it is lost when the API
                restarts.
              </span>
            </label>
          )}

          {preset.json_mode === "object" && (
            <p className="text-xs text-verdict-partial">
              This provider has no JSON-schema mode; answers are checked against the schema
              afterwards and may fail more often.
            </p>
          )}

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={runTest}
              disabled={test.kind === "running" || (authRequired && !managementToken.trim())}
              className="btn-quiet"
              data-testid="provider-test"
            >
              {test.kind === "running" ? "Testing…" : "Test connection"}
            </button>
            <button
              type="button"
              onClick={save}
              disabled={saving || (authRequired && !managementToken.trim())}
              className="btn-primary"
              data-testid="provider-save"
            >
              {saving ? "Saving…" : "Use this model"}
            </button>
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                setForm(null);
                setManagementToken("");
              }}
              className="px-2 py-1.5 text-ink-3 hover:text-ink"
            >
              Cancel
            </button>
          </div>

          {test.kind === "ok" && (
            <p className="text-xs text-verdict-ok" data-testid="provider-test-result">
              Connected{test.latencyMs !== null ? ` in ${test.latencyMs} ms` : ""}.
            </p>
          )}
          {test.kind === "failed" && (
            <p className="text-xs text-verdict-bad" data-testid="provider-test-result">
              Connection failed: {test.error}
            </p>
          )}
          {saveError && (
            <p className="text-xs text-verdict-bad" data-testid="provider-save-error">
              {saveError}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
