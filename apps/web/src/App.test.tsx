import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/"); // App mirrors the analysis in the URL
});

describe("App", () => {
  it("names the product and opens on a new analysis", () => {
    render(<App />);
    expect(screen.getAllByText("Claim2Code").length).toBeGreaterThan(0);
    expect(
      screen.getByRole("heading", { level: 1, name: "New analysis" }),
    ).toBeInTheDocument();
  });

  it("shows the online badge when /health returns ok", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: "ok", version: "0.1.0" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    render(<App />);

    const badge = await screen.findByTestId("status-badge");
    await waitFor(() => expect(badge).toHaveTextContent("online"));
  });

  it("shows the offline badge when fetch fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("connection refused")),
    );

    render(<App />);

    const badge = await screen.findByTestId("status-badge");
    await waitFor(() => expect(badge).toHaveTextContent("offline"));
    expect(screen.getByTestId("status-error")).toHaveTextContent(
      /connection refused/,
    );
  });

  it("renders the upload section on first visit", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    render(<App />);
    expect(await screen.findByTestId("upload-section")).toBeInTheDocument();
    expect(screen.getByTestId("file-input")).toBeInTheDocument();
  });

  it("transitions to the claims list after a successful upload", async () => {
    const analysisId = "11111111-1111-1111-1111-111111111111";
    const paperId = "22222222-2222-2222-2222-222222222222";

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/health")) {
        return Promise.resolve(
          new Response(JSON.stringify({ status: "ok" }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      if (url.endsWith("/papers")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: analysisId,
              paper_id: paperId,
              parse_status: "done",
              n_claims: 2,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.includes(`/analyses/${analysisId}/claims`)) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: analysisId,
              paper_status: "done",
              claims: [
                {
                  claim_id: "c1",
                  text: "The model uses β=0.1.",
                  claim_type: "HYPERPARAMETER",
                  confidence: 0.9,
                  paper_evidence: {
                    section_title: "Method",
                    page: 3,
                    char_start: 0,
                    char_end: 10,
                    quote: "we use beta=0.1",
                  },
                },
              ],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      return Promise.reject(new Error(`unexpected URL ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await screen.findByTestId("upload-section");

    // Pick a fake file and submit. Submit the form (rather than click
    // the button) so we bypass the ``disabled`` state synchronously.
    const file = new File(["%PDF-1.4 fake"], "tiny.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByTestId("file-input"), {
      target: { files: [file] },
    });
    fireEvent.submit(screen.getByTestId("upload-section"));

    // Summary banner appears.
    const summary = await screen.findByTestId("upload-summary");
    await waitFor(() =>
      expect(summary).toHaveTextContent(/extracted 2 claims/i),
    );

    // Claims list is fetched and rendered under its tab.
    fireEvent.click(screen.getByRole("button", { name: "Claims" }));
    const list = await screen.findByTestId("claims-list");
    expect(list).toBeInTheDocument();
    const cards = await screen.findAllByTestId("claim-card");
    expect(cards).toHaveLength(1);
    expect(screen.getByTestId("claim-confidence")).toHaveTextContent("90%");
  });

  it("shows an upload error when POST /papers fails", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/health")) {
        return Promise.resolve(
          new Response(JSON.stringify({ status: "ok" }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      if (url.endsWith("/papers")) {
        return Promise.resolve(
          new Response("LLM unreachable", { status: 500 }),
        );
      }
      return Promise.reject(new Error(`unexpected URL ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await screen.findByTestId("upload-section");

    const file = new File(["%PDF-1.4"], "tiny.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByTestId("file-input"), {
      target: { files: [file] },
    });
    fireEvent.submit(screen.getByTestId("upload-section"));

    const error = await screen.findByTestId("upload-error");
    await waitFor(() => expect(error).toHaveTextContent(/HTTP 500/i));
  });

  // ============================================================
  // RepositorySection / SymbolsList (Phase 3)
  // ============================================================

  it("renders the repository section after a successful upload", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/health")) {
        return Promise.resolve(
          new Response(JSON.stringify({ status: "ok" }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      if (url.endsWith("/papers")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: "11111111-1111-1111-1111-111111111111",
              paper_id: "22222222-2222-2222-2222-222222222222",
              parse_status: "done",
              n_claims: 0,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.includes("/analyses/") && url.endsWith("/claims")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: "11111111-1111-1111-1111-111111111111",
              paper_status: "done",
              claims: [],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      return Promise.reject(new Error(`unexpected URL ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await screen.findByTestId("upload-section");
    const file = new File(["%PDF-1.4 fake"], "tiny.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByTestId("file-input"), {
      target: { files: [file] },
    });
    fireEvent.submit(screen.getByTestId("upload-section"));

    const section = await screen.findByTestId("repository-section");
    expect(section).toBeInTheDocument();
    expect(screen.getByTestId("repo-url-input")).toBeInTheDocument();
    expect(screen.getByTestId("repo-clone-button")).toBeInTheDocument();
  });

  it("clones a repo and renders SymbolsList on success", async () => {
    const analysisId = "11111111-1111-1111-1111-111111111111";
    const repositoryId = "33333333-3333-3333-3333-333333333333";

    const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith("/health")) {
        return Promise.resolve(
          new Response(JSON.stringify({ status: "ok" }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      if (url.endsWith("/papers")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: analysisId,
              paper_id: "22222222-2222-2222-2222-222222222222",
              parse_status: "done",
              n_claims: 0,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.includes(`/analyses/${analysisId}/claims`)) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: analysisId,
              paper_status: "done",
              claims: [],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.endsWith("/repositories") && init?.method === "POST") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              repository_id: repositoryId,
              analysis_id: analysisId,
              url: "https://github.com/acme/demo",
              owner: "acme",
              name: "demo",
              n_files: 2,
              n_symbols: 2,
              n_configs: 1,
              n_dependencies: 2,
              index_status: "done",
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.includes(`/repositories/${repositoryId}/symbols`)) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              repository_id: repositoryId,
              symbols: [
                {
                  kind: "function",
                  language: "python",
                  file_path: "pkg/core.py",
                  start_line: 1,
                  start_col: 0,
                  end_line: 3,
                  end_col: 0,
                  name: "add",
                  signature: "(a, b)",
                  docstring: "Add two ints.",
                },
                {
                  kind: "class",
                  language: "python",
                  file_path: "pkg/widget.py",
                  start_line: 5,
                  start_col: 0,
                  end_line: 20,
                  end_col: 0,
                  name: "Widget",
                  signature: "",
                  docstring: null,
                },
              ],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      return Promise.reject(new Error(`unexpected URL ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await screen.findByTestId("upload-section");
    const file = new File(["%PDF-1.4 fake"], "tiny.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByTestId("file-input"), {
      target: { files: [file] },
    });
    fireEvent.submit(screen.getByTestId("upload-section"));

    await screen.findByTestId("repository-section");
    fireEvent.change(screen.getByTestId("repo-url-input"), {
      target: { value: "https://github.com/acme/demo" },
    });
    fireEvent.submit(screen.getByTestId("repository-section"));

    // Repo summary banner appears.
    const summary = await screen.findByTestId("repo-summary");
    await waitFor(() =>
      expect(summary).toHaveTextContent(/acme\/demo/i),
    );

    // SymbolsList fetches and renders the symbols grouped by file.
    fireEvent.click(screen.getByRole("button", { name: "Code symbols" }));
    const list = await screen.findByTestId("symbols-list");
    expect(list).toBeInTheDocument();
    const files = await screen.findAllByTestId("symbols-file");
    expect(files).toHaveLength(2);
    const filePaths = await screen.findAllByTestId("symbols-file-path");
    expect(filePaths.map((n) => n.textContent)).toContain("pkg/core.py");
    expect(filePaths.map((n) => n.textContent)).toContain("pkg/widget.py");
    const rows = await screen.findAllByTestId("symbol-row");
    expect(rows).toHaveLength(2);
    const symbolNames = await screen.findAllByTestId("symbol-name");
    expect(symbolNames.map((n) => n.textContent ?? "").some((t) => /add/.test(t))).toBe(true);
  });

  it("shows an error when POST /repositories fails", async () => {
    const analysisId = "11111111-1111-1111-1111-111111111111";

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/health")) {
        return Promise.resolve(
          new Response(JSON.stringify({ status: "ok" }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      if (url.endsWith("/papers")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: analysisId,
              paper_id: "22222222-2222-2222-2222-222222222222",
              parse_status: "done",
              n_claims: 0,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.includes(`/analyses/${analysisId}/claims`)) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: analysisId,
              paper_status: "done",
              claims: [],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.endsWith("/repositories")) {
        return Promise.resolve(
          new Response("repository too large", { status: 413 }),
        );
      }
      return Promise.reject(new Error(`unexpected URL ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await screen.findByTestId("upload-section");
    const file = new File(["%PDF-1.4 fake"], "tiny.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByTestId("file-input"), {
      target: { files: [file] },
    });
    fireEvent.submit(screen.getByTestId("upload-section"));

    await screen.findByTestId("repository-section");
    fireEvent.change(screen.getByTestId("repo-url-input"), {
      target: { value: "https://github.com/huge/monorepo" },
    });
    fireEvent.submit(screen.getByTestId("repository-section"));

    const error = await screen.findByTestId("repo-error");
    await waitFor(() => expect(error).toHaveTextContent(/HTTP 413/i));
    // SymbolsList should never appear.
    expect(screen.queryByTestId("symbols-list")).not.toBeInTheDocument();
  });

  it("shows symbols-empty when the repo has no symbols", async () => {
    const analysisId = "11111111-1111-1111-1111-111111111111";
    const repositoryId = "33333333-3333-3333-3333-333333333333";

    const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith("/health")) {
        return Promise.resolve(
          new Response(JSON.stringify({ status: "ok" }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      if (url.endsWith("/papers")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: analysisId,
              paper_id: "22222222-2222-2222-2222-222222222222",
              parse_status: "done",
              n_claims: 0,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.includes(`/analyses/${analysisId}/claims`)) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              analysis_id: analysisId,
              paper_status: "done",
              claims: [],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.endsWith("/repositories") && init?.method === "POST") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              repository_id: repositoryId,
              analysis_id: analysisId,
              url: "https://github.com/empty/nope",
              owner: "empty",
              name: "nope",
              n_files: 0,
              n_symbols: 0,
              n_configs: 0,
              n_dependencies: 0,
              index_status: "done",
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.includes(`/repositories/${repositoryId}/symbols`)) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ repository_id: repositoryId, symbols: [] }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      return Promise.reject(new Error(`unexpected URL ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await screen.findByTestId("upload-section");
    const file = new File(["%PDF-1.4 fake"], "tiny.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByTestId("file-input"), {
      target: { files: [file] },
    });
    fireEvent.submit(screen.getByTestId("upload-section"));

    await screen.findByTestId("repository-section");
    fireEvent.change(screen.getByTestId("repo-url-input"), {
      target: { value: "https://github.com/empty/nope" },
    });
    fireEvent.submit(screen.getByTestId("repository-section"));

    fireEvent.click(await screen.findByRole("button", { name: "Code symbols" }));
    expect(await screen.findByTestId("symbols-empty")).toBeInTheDocument();
  });

  it("reopens an earlier analysis by id and shows its verification panel", async () => {
    const analysisId = "33333333-3333-3333-3333-333333333333";
    const respond = (body: unknown, status = 200) =>
      Promise.resolve(
        new Response(JSON.stringify(body), {
          status,
          headers: { "content-type": "application/json" },
        }),
      );
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        if (url.endsWith("/health")) return respond({ status: "ok" });
        if (url.includes("/llm/")) return respond({}, 404);
        if (url.includes("/analyses/missing/")) return respond({ detail: "nope" }, 404);
        if (url.endsWith("/claims")) {
          return respond({ analysis_id: analysisId, paper_status: "done", claims: [] });
        }
        return respond({ analysis_id: analysisId, verifications: [] });
      }),
    );
    render(<App />);

    fireEvent.change(screen.getByTestId("open-analysis-input"), {
      target: { value: "missing" },
    });
    fireEvent.click(screen.getByTestId("open-analysis-button"));
    expect(await screen.findByTestId("open-analysis-error")).toHaveTextContent(
      "No analysis with that id.",
    );

    fireEvent.change(screen.getByTestId("open-analysis-input"), {
      target: { value: analysisId },
    });
    fireEvent.click(screen.getByTestId("open-analysis-button"));
    expect(await screen.findByTestId("upload-summary")).toHaveTextContent("0 claims");
    expect(await screen.findByTestId("verification-panel")).toBeInTheDocument();
  });

  it("opens the analysis named in the URL", async () => {
    const analysisId = "44444444-4444-4444-4444-444444444444";
    window.history.replaceState(null, "", `/?analysis=${analysisId}`);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        if (url.includes("/llm/")) {
          return Promise.resolve(new Response("{}", { status: 404 }));
        }
        const body = url.endsWith("/health")
          ? { status: "ok" }
          : url.endsWith("/claims")
            ? { analysis_id: analysisId, paper_status: "done", claims: [] }
            : { analysis_id: analysisId, verifications: [] };
        return Promise.resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }),
    );
    render(<App />);

    expect(await screen.findByTestId("verification-panel")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /new analysis/i }));
    expect(window.location.search).toBe("");
  });
  it("restores the repository bound to a reopened analysis", async () => {
    const analysisId = "55555555-5555-5555-5555-555555555555";
    window.history.replaceState(null, "", `/?analysis=${analysisId}`);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        if (url.includes("/llm/")) {
          return Promise.resolve(new Response("{}", { status: 404 }));
        }
        const body = url.endsWith("/health")
          ? { status: "ok" }
          : url.endsWith("/claims")
            ? { analysis_id: analysisId, paper_status: "done", claims: [] }
            : url.endsWith("/repository")
              ? {
                  repository_id: "66666666-6666-6666-6666-666666666666",
                  analysis_id: analysisId,
                  url: "https://github.com/acme/demo",
                  owner: "acme",
                  name: "demo",
                  n_files: 3,
                  n_symbols: 7,
                  index_status: "done",
                }
              : { analysis_id: analysisId, verifications: [] };
        return Promise.resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }),
    );
    render(<App />);

    const summary = await screen.findByTestId("repo-summary");
    expect(summary).toHaveTextContent("acme/demo");
    expect(summary).toHaveTextContent("3 files, 7 symbols");
    expect(screen.queryByTestId("repository-section")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Code symbols" })).toBeInTheDocument();
  });
});
