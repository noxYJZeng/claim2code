import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import CitedCode from "./CitedCode";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const analysisId = "11111111-1111-1111-1111-111111111111";
const citation = {
  symbol_id: "s1",
  file_path: "pkg/m.py",
  start_line: 5,
  end_line: 6,
  quote: "x = 5",
};

function excerpt(context: number) {
  const lo = 5 - context;
  const hi = 6 + context;
  return {
    file_path: "pkg/m.py",
    start_line: 5,
    end_line: 6,
    total_lines: 40,
    lines: Array.from({ length: hi - lo + 1 }, (_, i) => ({
      number: lo + i,
      text: `line ${lo + i}`,
    })),
    url: "https://github.com/o/r/blob/abc123/pkg/m.py#L5-L6",
  };
}

function json(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("CitedCode", () => {
  it("shows the cited lines from the repository, highlighted, with a GitHub link", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      const context = Number(new URL(url).searchParams.get("context"));
      return Promise.resolve(json(excerpt(context)));
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<CitedCode analysisId={analysisId} citation={citation} />);

    const code = await screen.findByTestId("code-excerpt");
    expect(code).toHaveTextContent("line 2");
    const cited = code.querySelectorAll("[data-cited]");
    expect([...cited].map((el) => el.textContent)).toEqual(["5line 5", "6line 6"]);
    expect(screen.getByRole("link", { name: /view on github/i })).toHaveAttribute(
      "href",
      "https://github.com/o/r/blob/abc123/pkg/m.py#L5-L6",
    );
    const url = new URL(fetchMock.mock.calls[0][0] as string);
    expect(url.pathname).toBe(`/analyses/${analysisId}/code`);
    expect(url.searchParams.get("file_path")).toBe("pkg/m.py");

    fireEvent.click(screen.getByTestId("code-more-context"));
    expect(await screen.findByText("line 15")).toBeInTheDocument();
    expect(screen.queryByTestId("code-more-context")).not.toBeInTheDocument();
  });

  it("falls back to the model's quote when the file cannot be read", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(json({ detail: "pkg/m.py not found in the repository" }, 404)),
    );
    render(<CitedCode analysisId={analysisId} citation={citation} />);

    const fallback = await screen.findByTestId("code-excerpt-error");
    expect(fallback).toHaveTextContent("pkg/m.py not found in the repository");
    expect(fallback).toHaveTextContent("x = 5");
  });
});
