import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ReferenceCheck } from "./api";
import ReferenceTest, { useReferenceChecks } from "./ReferenceTest";

function Harness({ claimId = "c1", bound = true }: { claimId?: string; bound?: boolean }) {
  const reference = useReferenceChecks("a1");
  return <ReferenceTest claimId={claimId} repositoryBound={bound} reference={reference} />;
}

const MISMATCH: ReferenceCheck = {
  claim_id: "c1",
  status: "MISMATCH",
  explanation: "1 of 2 inputs gave a different output from smooth_l1 than from the reference.",
  target: {
    symbol_id: "s1", file_path: "losses.py", start_line: 2, end_line: 4,
    qualname: "smooth_l1", signature: "(x, y, beta=1.0)",
  },
  reference_code: "def reference(x, y, beta=1.0):\n    ...",
  inputs_code: "def make_inputs():\n    ...",
  instance_code: "",
  assumptions: ["Mean over elements."],
  rtol: 1e-5,
  atol: 1e-8,
  cases: [
    { index: 0, inputs: "{'args': [[0.0], [0.5]]}", outcome: "match", actual: "0.125", expected: "0.125", detail: "" },
    { index: 1, inputs: "{'args': [[3.0], [0.0]]}", outcome: "mismatch", actual: "3.0", expected: "2.5", detail: "3.0 != 2.5" },
  ],
  isolation: ["separate process with an empty environment", "no network"],
  environment: ["Not installed, replaced by stubs: wandb"],
  generated_at: "2026-11-02T00:00:00Z",
};

function stubFetch(enabled: boolean, checks: ReferenceCheck[], post?: ReferenceCheck) {
  const fetchMock = vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
    if (init?.method === "POST") {
      return Promise.resolve(new Response(JSON.stringify(post), { status: 200 }));
    }
    return Promise.resolve(new Response(JSON.stringify({ analysis_id: "a1", enabled, checks })));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("ReferenceTest", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("explains that the server has it turned off", async () => {
    stubFetch(false, []);
    render(<Harness />);
    expect(await screen.findByTestId("reference-disabled")).toHaveTextContent("REFERENCE_CHECK_ENABLED=true");
    expect(screen.queryByTestId("reference-run")).not.toBeInTheDocument();
  });

  it("shows a stored result with each input side by side", async () => {
    stubFetch(true, [MISMATCH]);
    render(<Harness />);
    expect(await screen.findByTestId("reference-status")).toHaveTextContent("Different output");
    expect(screen.getByText("smooth_l1 · losses.py:2")).toBeInTheDocument();
    expect(screen.getByText("Differs")).toBeInTheDocument();
    expect(screen.getByText("3.0 != 2.5")).toBeInTheDocument();
    expect(screen.getByText("Mean over elements.")).toBeInTheDocument();
    expect(screen.getByText(/Ran with: separate process/)).toBeInTheDocument();
    expect(screen.getByTestId("reference-environment")).toHaveTextContent("replaced by stubs: wandb");
    expect(screen.getByTestId("reference-run")).toHaveTextContent("Run again");
  });

  it("runs a test for the open claim and shows the outcome", async () => {
    const fetchMock = stubFetch(true, [], { ...MISMATCH, status: "MATCH", explanation: "All 2 inputs matched." });
    render(<Harness />);
    fireEvent.click(await screen.findByTestId("reference-run"));
    expect(await screen.findByText("All 2 inputs matched.")).toBeInTheDocument();
    expect(screen.getByTestId("reference-status")).toHaveTextContent("Same output");
    const post = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
    expect(JSON.parse(post?.[1]?.body as string)).toEqual({ claim_id: "c1" });
  });

  it("cannot run without a repository", async () => {
    stubFetch(true, []);
    render(<Harness bound={false} />);
    await waitFor(() => expect(screen.getByTestId("reference-run")).toBeDisabled());
    expect(screen.getByText("Bind a repository first.")).toBeInTheDocument();
  });
});
