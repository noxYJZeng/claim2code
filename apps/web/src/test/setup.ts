// Vitest setup — extends expect with @testing-library/jest-dom matchers
// and tears down the rendered tree between tests.
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
