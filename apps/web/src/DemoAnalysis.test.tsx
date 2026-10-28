import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/");
});

describe("read-only sample analysis", () => {
  it("opens a shareable sample without fetching a backend or mounting model controls", () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("No backend should be called"));
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(null, "", "/?demo=1");
    render(<App />);

    expect(screen.getByTestId("sample-analysis")).toBeInTheDocument();
    expect(screen.getByTestId("sample-verdict")).toHaveTextContent("Verified");
    expect(screen.getByTestId("sample-scope")).toHaveTextContent("not a verbatim PDF quotation");
    expect(screen.getByText("main_pretrain.py:53-54")).toBeInTheDocument();
    expect(screen.getAllByTestId("sample-code-quote")[0]).toHaveTextContent("default=0.75");
    expect(screen.queryByTestId("provider-panel")).not.toBeInTheDocument();
    expect(screen.queryByTestId("upload-section")).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /configuration discrepancy/i }));
    expect(screen.getByTestId("sample-verdict")).toHaveTextContent("Partially verified");
    expect(screen.getByRole("region", { name: "Recorded discrepancies" })).toHaveTextContent(
      "Base learning rate default is 1e-3",
    );

    fireEvent.click(screen.getByRole("button", { name: /insufficient evidence/i }));
    expect(screen.getByTestId("sample-verdict")).toHaveTextContent("Unverified");
    expect(screen.getByTestId("sample-no-evidence")).toHaveTextContent("No code evidence was recorded");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
