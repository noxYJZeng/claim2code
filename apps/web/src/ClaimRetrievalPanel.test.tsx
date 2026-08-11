import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import ClaimRetrievalPanel from "./ClaimRetrievalPanel";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const analysisId = "11111111-1111-1111-1111-111111111111";
const repoId = "22222222-2222-2222-2222-222222222222";

const claimA = {
  claim_id: "claim-a",
  text: "The model uses softmax(QK^T / sqrt(d_k)).",
  claim_type: "METHOD",
  confidence: 0.9,
  paper_evidence: {
    section_title: "Method",
    page: 3,
    char_start: 0,
    char_end: 1,
    quote: "...",
  },
};

const claimB = {
  claim_id: "claim-b",
  text: "We train for 100 epochs.",
  claim_type: "HYPERPARAMETER",
  confidence: 0.85,
  paper_evidence: {
    section_title: "Training",
    page: 5,
    char_start: 0,
    char_end: 1,
    quote: "...",
  },
};

const matches = [
  {
    symbol_id: "sym-1",
    kind: "function" as const,
    language: "python" as const,
    file_path: "pkg/attention.py",
    start_line: 10,
    start_col: 0,
    end_line: 20,
    end_col: 0,
    name: "softmax",
    signature: "(x, axis=-1)",
    docstring: "Compute softmax.",
    similarity: 0.91,
  },
];

function mockJsonResponse(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("ClaimRetrievalPanel", () => {
  it("renders the loading state while claims are in flight", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => new Promise(() => {})),
    );
    render(
      <ClaimRetrievalPanel repositoryId={repoId} analysisId={analysisId} />,
    );
    expect(screen.getByTestId("retrieval-loading-claims")).toBeInTheDocument();
  });

  it("renders an empty state when the analysis has no claims", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockJsonResponse({
          analysis_id: analysisId,
          paper_status: "done",
          claims: [],
        }),
      ),
    );
    render(
      <ClaimRetrievalPanel repositoryId={repoId} analysisId={analysisId} />,
    );
    await waitFor(() =>
      expect(screen.getByTestId("retrieval-empty")).toBeInTheDocument(),
    );
  });

  it("renders an error state when the claims fetch fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(new Response("HTTP 500: internal", { status: 500 })),
    );
    render(
      <ClaimRetrievalPanel repositoryId={repoId} analysisId={analysisId} />,
    );
    await waitFor(() =>
      expect(screen.getByTestId("retrieval-claims-error")).toBeInTheDocument(),
    );
  });

  it("picks the first claim by default and triggers retrieval", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes(`/analyses/${analysisId}/claims`)) {
        return Promise.resolve(
          mockJsonResponse({
            analysis_id: analysisId,
            paper_status: "done",
            claims: [claimA, claimB],
          }),
        );
      }
      if (url.includes(`/repositories/${repoId}/retrieve`)) {
        return Promise.resolve(
          mockJsonResponse({
            repository_id: repoId,
            claim: claimA.text,
            symbols: matches,
            scores: [0.91],
          }),
        );
      }
      return Promise.reject(new Error(`unexpected URL: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ClaimRetrievalPanel repositoryId={repoId} analysisId={analysisId} />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("retrieval-panel")).toBeInTheDocument(),
    );

    const picker = screen.getByTestId("claim-picker") as HTMLSelectElement;
    expect(picker.value).toBe(claimA.claim_id);

    await waitFor(() =>
      expect(screen.getByTestId("retrieval-matches")).toBeInTheDocument(),
    );

    const score = screen.getByTestId("retrieval-match-score");
    expect(score).toHaveTextContent("0.910");

    const name = screen.getByTestId("retrieval-match-name");
    expect(name).toHaveTextContent("softmax");

    // The retrieval fetch should have been called once with claim A's text.
    const retrieveCalls = fetchMock.mock.calls.filter((call) =>
      String(call[0]).includes("/retrieve"),
    );
    expect(retrieveCalls).toHaveLength(1);
    const body = JSON.parse(retrieveCalls[0][1].body);
    expect(body.claim).toBe(claimA.text);
    expect(body.top_k).toBe(5);
  });

  it("re-runs retrieval when the user picks a different claim", async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation((url: string, init?: RequestInit) => {
        if (url.includes(`/analyses/${analysisId}/claims`)) {
          return Promise.resolve(
            mockJsonResponse({
              analysis_id: analysisId,
              paper_status: "done",
              claims: [claimA, claimB],
            }),
          );
        }
        if (url.includes(`/repositories/${repoId}/retrieve`)) {
          const body = JSON.parse(init?.body as string);
          return Promise.resolve(
            mockJsonResponse({
              repository_id: repoId,
              claim: body.claim,
              symbols: body.claim === claimB.text ? [] : matches,
              scores: body.claim === claimB.text ? [] : [0.91],
            }),
          );
        }
        return Promise.reject(new Error(`unexpected URL: ${url}`));
      });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ClaimRetrievalPanel repositoryId={repoId} analysisId={analysisId} />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("retrieval-panel")).toBeInTheDocument(),
    );
    await waitFor(() =>
      expect(screen.getByTestId("retrieval-matches")).toBeInTheDocument(),
    );

    // Switch to claim B → expect empty results.
    fireEvent.change(screen.getByTestId("claim-picker"), {
      target: { value: claimB.claim_id },
    });

    await waitFor(() =>
      expect(screen.getByTestId("retrieval-empty-results")).toBeInTheDocument(),
    );

    // And back to claim A → matches return.
    fireEvent.change(screen.getByTestId("claim-picker"), {
      target: { value: claimA.claim_id },
    });
    await waitFor(() =>
      expect(screen.getByTestId("retrieval-matches")).toBeInTheDocument(),
    );

    // At least one retrieve call per claim was made.
    const retrieveCalls = fetchMock.mock.calls.filter((call) =>
      String(call[0]).includes("/retrieve"),
    );
    expect(retrieveCalls.length).toBeGreaterThanOrEqual(2);
  });

  it("renders an error state when retrieval fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        if (url.includes(`/analyses/${analysisId}/claims`)) {
          return Promise.resolve(
            mockJsonResponse({
              analysis_id: analysisId,
              paper_status: "done",
              claims: [claimA],
            }),
          );
        }
        if (url.includes(`/repositories/${repoId}/retrieve`)) {
          return Promise.resolve(
            new Response("HTTP 500: simulated outage", { status: 500 }),
          );
        }
        return Promise.reject(new Error(`unexpected URL: ${url}`));
      }),
    );

    render(
      <ClaimRetrievalPanel repositoryId={repoId} analysisId={analysisId} />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("retrieval-error")).toBeInTheDocument(),
    );
  });
});
