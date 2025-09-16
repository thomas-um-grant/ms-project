import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import vueDevTools from "vite-plugin-vue-devtools";

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), vueDevTools()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      // Proxy backend endpoints during dev
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/rag": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/index": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/retrieve": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/answer": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/open-path": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
