import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import VerificationPanel from "./VerificationPanel";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const analysisId = "11111111-1111-1111-1111-111111111111";

const paperEvidence = {
  section_title: "Method",
  page: 3,
  char_start: 0,
  char_end: 1,
  quote: "We set beta_1 = 1e-4.",
};

const claims = ["C1", "C2", "C3"].map((claim_id) => ({
  claim_id,
  text: `Claim ${claim_id}.`,
  claim_type: "METHOD",
  confidence: 0.9,
  paper_evidence: paperEvidence,
}));

function verdict(claim_id: string, status: string, extra: object = {}) {
  return {
    claim_id,
    status,
    confidence: 0.8,
    explanation: `Explanation for ${claim_id}.`,
    paper_evidence: paperEvidence,
    code_evidence: [],
    discrepancies: [],
    hyperparameters: [],
    ...extra,
  };
}

function json(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// Routes the panel's three calls; POSTs answer with `postVerdict`.
function stubApi(
  stored: object[],
  postVerdict: (claimId: string) => Response = (id) =>
    json({ verifications: [verdict(id, "VERIFIED")] }),
) {
  const posts: string[][] = [];
  const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
    if (url.endsWith("/claims")) {
      return Promise.resolve(json({ analysis_id: analysisId, claims }));
    }
    if (init?.method === "POST") {
      const ids = JSON.parse(init.body as string).claim_ids as string[];
      posts.push(ids);
      return Promise.resolve(postVerdict(ids[0]));
    }
    return Promise.resolve(json({ analysis_id: analysisId, verifications: stored }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return posts;
}

describe("VerificationPanel", () => {
  it("shows stored verdicts and counts the claims not yet checked", async () => {
    stubApi([verdict("C1", "CONTRADICTED")]);
    render(<VerificationPanel analysisId={analysisId} />);

    await waitFor(() => expect(screen.getByTestId("verification-panel")).toBeInTheDocument());
    expect(screen.getAllByTestId("verdict-card")).toHaveLength(3);
    expect(screen.getByTestId("verdict-status")).toHaveTextContent("Contradicted");
    expect(screen.getByTestId("verification-counts")).toHaveTextContent("Contradicted: 1");
    expect(screen.getByTestId("verification-counts")).toHaveTextContent("Not checked: 2");
    expect(screen.getByTestId("verify-all")).toHaveTextContent("Verify 2 unverified claims");
  });

  it("verifies the unchecked claims one request at a time", async () => {
    const posts = stubApi([verdict("C1", "VERIFIED")]);
    render(<VerificationPanel analysisId={analysisId} />);

    fireEvent.click(await screen.findByTestId("verify-all"));

    await waitFor(() => expect(screen.getAllByTestId("verdict-status")).toHaveLength(3));
    expect(posts).toEqual([["C2"], ["C3"]]);
    expect(screen.getByTestId("verification-counts")).toHaveTextContent("Verified: 3");
    expect(screen.getByTestId("verify-all")).toBeDisabled();
  });

  it("stops the queue and reports the claim whose request failed", async () => {
    const posts = stubApi([], () => json({ detail: "LLM down" }, 500));
    render(<VerificationPanel analysisId={analysisId} />);

    fireEvent.click(await screen.findByTestId("verify-all"));

    await waitFor(() =>
      expect(screen.getByTestId("verification-run-error")).toHaveTextContent("C1: HTTP 500"),
    );
    expect(posts).toEqual([["C1"]]);
  });

  it("shows the explanation, discrepancies and code citations on demand", async () => {
    stubApi([
      verdict("C1", "PARTIALLY_VERIFIED", {
        code_evidence: [
          {
            symbol_id: "s1",
            file_path: "diffusion_tf/utils.py",
            start_line: 12,
            end_line: 14,
            quote: "betas = np.linspace(beta_start, beta_end, n)",
          },
        ],
        discrepancies: [
          {
            type: "HYPERPARAMETER_MISMATCH",
            severity: "MEDIUM",
            detail: "Paper says 1e-4, code defaults to 1e-3.",
            code_quote: "beta_start=1e-3",
          },
        ],
      }),
    ]);
    render(<VerificationPanel analysisId={analysisId} />);

    const toggle = await screen.findByTestId("verdict-toggle");
    expect(toggle).toHaveTextContent("1 code citation · 1 discrepancy");
    fireEvent.click(toggle);

    const details = screen.getByTestId("verdict-details");
    expect(details).toHaveTextContent("Explanation for C1.");
    expect(details).toHaveTextContent("Paper says 1e-4, code defaults to 1e-3.");
    expect(screen.getByTestId("code-citation")).toHaveTextContent("diffusion_tf/utils.py:12-14");
  });

  it("renders the load error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({}, 404)));
    render(<VerificationPanel analysisId={analysisId} />);
    expect(await screen.findByTestId("verification-error")).toHaveTextContent("HTTP 404");
  });
});
