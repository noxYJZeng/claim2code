import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the project title", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { level: 1, name: /claim2code/i }),
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

    // Claims list is fetched and rendered.
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
});