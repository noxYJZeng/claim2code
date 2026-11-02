import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import CitedPaper from "./CitedPaper";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const analysisId = "11111111-1111-1111-1111-111111111111";
const evidence = {
  section_title: "4 Experiments",
  page: 5,
  char_start: 0,
  char_end: 10,
  quote: "We set T = 1000 for all experiments",
};

function json(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function excerpt(extra: object = {}) {
  return {
    page: 5,
    n_pages: 25,
    found: true,
    image: "data:image/png;base64,AAAA",
    width: 1224,
    height: 400,
    ...extra,
  };
}

describe("CitedPaper", () => {
  it("shows the highlighted region and links the PDF at that page", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json(excerpt()));
    vi.stubGlobal("fetch", fetchMock);
    render(<CitedPaper analysisId={analysisId} evidence={evidence} />);

    const img = await screen.findByTestId("paper-excerpt");
    expect(img).toHaveAttribute("src", "data:image/png;base64,AAAA");
    expect(img).toHaveAttribute("alt", "Page 5 of the paper, quote highlighted");
    expect(screen.getByTestId("paper-open-pdf")).toHaveAttribute(
      "href",
      `http://localhost:8000/analyses/${analysisId}/paper/pdf#page=5`,
    );
    const url = new URL(fetchMock.mock.calls[0][0] as string);
    expect(url.pathname).toBe(`/analyses/${analysisId}/paper/excerpt`);
    expect(url.searchParams.get("quote")).toBe(evidence.quote);
    expect(url.searchParams.get("full")).toBe("false");

    fireEvent.click(screen.getByTestId("paper-toggle-page"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(new URL(fetchMock.mock.calls[1][0] as string).searchParams.get("full")).toBe("true");
  });

  it("labels a quote found on another page, or not found at all", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(excerpt({ page: 6 }))));
    const { unmount } = render(<CitedPaper analysisId={analysisId} evidence={evidence} />);
    expect(await screen.findByTestId("paper-location")).toHaveTextContent(
      "4 Experiments, page 6 (cited as page 5)",
    );
    await screen.findByText("Open PDF at page 6 ↗");
    unmount();

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(excerpt({ found: false }))));
    render(<CitedPaper analysisId={analysisId} evidence={evidence} />);
    expect(await screen.findByTestId("paper-not-found")).toBeInTheDocument();
  });

  it("reports a render error and keeps the quote", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(json({ detail: "the uploaded PDF is no longer stored" }, 404)),
    );
    render(<CitedPaper analysisId={analysisId} evidence={evidence} />);
    expect(await screen.findByTestId("paper-excerpt-error")).toHaveTextContent(
      "the uploaded PDF is no longer stored",
    );
    expect(screen.getByTestId("paper-citation")).toHaveTextContent(evidence.quote);
  });

  it("rejects a malformed image response and keeps the recorded quote", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(excerpt({ image: "https://other.example/image.png" }))));
    render(<CitedPaper analysisId={analysisId} evidence={evidence} />);

    expect(await screen.findByTestId("paper-excerpt-error")).toHaveTextContent(
      "Invalid paper excerpt response",
    );
    expect(screen.queryByTestId("paper-excerpt")).not.toBeInTheDocument();
    expect(screen.getByTestId("paper-citation")).toHaveTextContent(evidence.quote);
  });
});
