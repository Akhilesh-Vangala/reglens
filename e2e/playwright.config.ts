import { defineConfig, devices } from "@playwright/test";

// Runs against the real FastAPI app: the same server, corpus and calculator that are deployed. Live answering stays
// off (no key), so model-backed views are exercised through the recorded real runs.
const PORT = 8766;
const python = process.env.PYTHON ?? "python";

export default defineConfig({
  testDir: "tests",
  timeout: 45_000,
  expect: { timeout: 10_000 },
  globalSetup: "./warmup.ts",
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "retain-on-failure" },
  webServer: {
    command: `${python} -m uvicorn reglens.server.app:app --host 127.0.0.1 --port ${PORT}`,
    cwd: "..",
    url: `http://127.0.0.1:${PORT}/healthz`,
    env: { REGLENS_NO_DOTENV: "1", ANTHROPIC_API_KEY: "" },
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1360, height: 900 } }, grepInvert: /@mobile/ },
    { name: "mobile", use: { ...devices["Pixel 7"] }, grep: /@mobile/ },
  ],
});
