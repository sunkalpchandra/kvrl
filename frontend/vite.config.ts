import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// GHPAGES_BASE lets the static build live under a repo subpath (GitHub Pages).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: process.env.GHPAGES_BASE || "/",
  server: { proxy: { "/api": "http://localhost:8000" } },
  build: { outDir: "dist", sourcemap: false },
});
