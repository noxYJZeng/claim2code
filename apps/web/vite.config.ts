import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Claim2Code — Vite config (Phase 0 stub, Phase 1 wires real app).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
});