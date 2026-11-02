import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import ModelProviderPanel from "./ModelProviderPanel";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function preset(id: string, label: string, extra: object = {}) {
  return {
    id,
    label,
    kind: "openai_compat",
    base_url: `http://${id}/v1`,
    default_model: `${id}-model`,
    models: [],
    needs_key: false,
    key_env: null,
    key_configured: false,
    json_mode: "schema",
    local: false,
    note: "",
    ...extra,
  };
}

const view = {
  management: { enabled: true, auth_required: false },
  active: {
    provider_id: "lmstudio",
    label: "LM Studio (local)",
    model: "qwen3.8-27b-mlx",
    base_url: "http://localhost:1234/v1",
  },
  presets: [
    preset("lmstudio", "LM Studio (local)", { local: true }),
    preset("anthropic", "Anthropic (Claude)", {
      kind: "anthropic",
      default_model: "claude-opus-5-5",
      models: ["claude-opus-5-5", "claude-sonnet-5-5"],
      needs_key: true,
      key_env: "ANTHROPIC_API_KEY",
    }),
    preset("deepseek", "DeepSeek", { needs_key: true, key_configured: true, json_mode: "object" }),
  ],
  embedding: { model: "nomic-embed", base_url: "http://localhost:1234/v1", dim: 768 },
};

function json(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function stubApi(handlers: { test?: () => Response; save?: () => Response; view?: object } = {}) {
  const posts: { path: string; body: Record<string, string>; headers: HeadersInit | undefined }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (init?.method === "POST") {
        const path = new URL(url).pathname;
        posts.push({ path, body: JSON.parse(init.body as string), headers: init.headers });
        if (path === "/llm/provider/test") {
          return Promise.resolve(handlers.test?.() ?? json({ ok: true, latency_ms: 42 }));
        }
        return Promise.resolve(handlers.save?.() ?? json(handlers.view ?? view));
      }
      return Promise.resolve(json(handlers.view ?? view));
    }),
  );
  return posts;
}

describe("ModelProviderPanel", () => {
  it("shows the active chat model and the fixed embedding model", async () => {
    stubApi();
    render(<ModelProviderPanel />);
    expect(await screen.findByTestId("provider-active")).toHaveTextContent(
      "LM Studio (local) · qwen3.8-27b-mlx",
    );
    expect(screen.getByTestId("provider-panel")).toHaveTextContent("nomic-embed");
  });

  it("keeps disabled or older server configurations read-only", async () => {
    stubApi({ view: { ...view, management: undefined } });
    render(<ModelProviderPanel />);

    expect(await screen.findByTestId("provider-read-only")).toBeInTheDocument();
    expect(screen.queryByTestId("provider-change")).not.toBeInTheDocument();
  });

  it("sends the management token only in authorization headers and clears it on close", async () => {
    const posts = stubApi({ view: { ...view, management: { enabled: true, auth_required: true } } });
    render(<ModelProviderPanel />);
    fireEvent.click(await screen.findByTestId("provider-change"));
    expect(screen.getByTestId("provider-test")).toBeDisabled();
    fireEvent.change(screen.getByTestId("provider-management-token"), { target: { value: "local-admin-test" } });
    fireEvent.click(screen.getByTestId("provider-test"));
    await screen.findByTestId("provider-test-result");
    fireEvent.click(screen.getByTestId("provider-save"));
    await waitFor(() => expect(screen.queryByTestId("provider-form")).not.toBeInTheDocument());

    for (const post of posts) {
      expect(new Headers(post.headers).get("Authorization")).toBe("Bearer local-admin-test");
      expect(JSON.stringify(post.body)).not.toContain("local-admin-test");
    }
    expect(window.localStorage.getItem("managementToken")).toBeNull();
    fireEvent.click(screen.getByTestId("provider-change"));
    expect(screen.getByTestId("provider-management-token")).toHaveValue("");
  });

  it("switching provider fills its defaults and asks for a key", async () => {
    stubApi();
    render(<ModelProviderPanel />);
    fireEvent.click(await screen.findByTestId("provider-change"));
    expect(screen.queryByTestId("provider-key")).not.toBeInTheDocument();

    fireEvent.change(screen.getByTestId("provider-select"), { target: { value: "anthropic" } });
    expect(screen.getByTestId("provider-model")).toHaveValue("claude-opus-5-5");
    expect(screen.getByTestId("provider-base-url")).toHaveValue("http://anthropic/v1");
    expect(screen.getByTestId("provider-key")).toHaveAttribute(
      "placeholder",
      "or set ANTHROPIC_API_KEY in .env",
    );
    expect(screen.getByRole("option", { name: "Anthropic (Claude) · needs key" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "DeepSeek · key set" })).toBeInTheDocument();
  });

  it("tests a choice, then saves it with the typed key", async () => {
    const posts = stubApi();
    render(<ModelProviderPanel />);
    fireEvent.click(await screen.findByTestId("provider-change"));
    fireEvent.change(screen.getByTestId("provider-select"), { target: { value: "anthropic" } });
    fireEvent.change(screen.getByTestId("provider-key"), { target: { value: "sk-typed" } });

    fireEvent.click(screen.getByTestId("provider-test"));
    expect(await screen.findByTestId("provider-test-result")).toHaveTextContent(
      "Connected in 42 ms",
    );

    fireEvent.click(screen.getByTestId("provider-save"));
    await waitFor(() => expect(screen.queryByTestId("provider-form")).not.toBeInTheDocument());
    expect(posts.map((p) => p.path)).toEqual(["/llm/provider/test", "/llm/provider"]);
    expect(posts[1].body).toEqual({
      provider_id: "anthropic",
      model: "claude-opus-5-5",
      base_url: "http://anthropic/v1",
      api_key: "sk-typed",
    });
  });

  it("reports a failed test and a rejected save", async () => {
    stubApi({
      test: () => json({ ok: false, error: "APIConnectionError: refused" }),
      save: () => json({ detail: "Anthropic (Claude): no API key" }, 422),
    });
    render(<ModelProviderPanel />);
    fireEvent.click(await screen.findByTestId("provider-change"));
    fireEvent.change(screen.getByTestId("provider-select"), { target: { value: "anthropic" } });

    fireEvent.click(screen.getByTestId("provider-test"));
    expect(await screen.findByTestId("provider-test-result")).toHaveTextContent(
      "Connection failed: APIConnectionError: refused",
    );
    fireEvent.click(screen.getByTestId("provider-save"));
    expect(await screen.findByTestId("provider-save-error")).toHaveTextContent(
      "Anthropic (Claude): no API key",
    );
    expect(screen.getByTestId("provider-form")).toBeInTheDocument();
  });

  it("warns when a provider only has JSON mode", async () => {
    stubApi();
    render(<ModelProviderPanel />);
    fireEvent.click(await screen.findByTestId("provider-change"));
    fireEvent.change(screen.getByTestId("provider-select"), { target: { value: "deepseek" } });
    expect(screen.getByTestId("provider-form")).toHaveTextContent("no JSON-schema mode");
    expect(screen.getByTestId("provider-key")).toHaveAttribute(
      "placeholder",
      "configured — leave blank to keep it",
    );
  });
});
