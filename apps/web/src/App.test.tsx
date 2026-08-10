import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
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
});
