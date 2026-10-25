import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// Recorded runs replay instantly under reduced motion.
test.use({ reducedMotion: "reduce" });

const TABS = ["copilot", "deadlines", "checker", "explore", "ask", "results", "how"];

test("tabs follow the WAI-ARIA pattern and the URL", async ({ page }) => {
  await page.goto("/#copilot");
  const first = page.getByRole("tab", { name: "Copilot" });
  await expect(first).toHaveAttribute("aria-selected", "true");
  await first.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Deadline calculator" })).toBeFocused();
  await expect(page).toHaveURL(/#deadlines/);
  await expect(page.getByRole("tabpanel")).toHaveAttribute("aria-labelledby", "tab-deadlines");
  await page.keyboard.press("End");
  await expect(page).toHaveURL(/#how$/);
  await page.keyboard.press("Home");
  await expect(page).toHaveURL(/#copilot$/);
  await expect(page).toHaveTitle(/^Copilot · RegLens/);
});

test("a recorded copilot run replays from its URL with verified, cross-linked citations", async ({ page }) => {
  await page.goto("/#copilot?run=C01");
  const out = page.locator("#copilotOut");
  await expect(out.getByRole("heading", { name: "Result" })).toBeVisible();
  await expect(out.locator(".pill").filter({ hasText: /verified citation/ })).toBeVisible();
  await expect(page.locator('.stage[data-stage="final"]')).toHaveClass(/done/);
  await expect(page.locator("#timelineCard")).toBeVisible();
  await expect(page.getByRole("log", { name: "Agent trace" }).locator("li").first()).toBeVisible();

  // a citation re-opens its paragraph in the graph
  await out.getByRole("link", { name: "Open in citation graph" }).first().click();
  await expect(page).toHaveURL(/#explore\?node=/);
  await expect(page.locator("#graphOut .graphsvg, #graphOut .empty")).toBeVisible();
});

test("citation re-check deep-links into the checker and accepts the real quote", async ({ page }) => {
  await page.goto("/#copilot?run=C01");
  await page.locator("#copilotOut").getByRole("link", { name: "Re-check this quote" }).first().click();
  await expect(page).toHaveURL(/#checker\?chunk=/);
  await expect(page.locator("#ck_out .stamp")).toHaveText("ACCEPTED");
});

test("checker rejects a changed number and shows where it diverges", async ({ page }) => {
  await page.goto("/#checker");
  await page.getByRole("button", { name: "change a number" }).click();
  await expect(page.locator("#ck_out .stamp")).toHaveText("REJECTED");
  await expect(page.locator("#ck_out .diff .x")).toBeVisible();
  await page.getByRole("button", { name: "exact quote" }).click();
  await expect(page.locator("#ck_out .stamp")).toHaveText("ACCEPTED");
  await page.getByRole("button", { name: "real quote, passage not retrieved" }).click();
  await expect(page.locator("#ck_out")).toContainText("not_retrieved");
  await expect(page).toHaveURL(/retrieved=0/);
});

test("deadline presets compute and are addressable", async ({ page }) => {
  await page.goto("/#deadlines");
  await page.getByRole("button", { name: "reported three weeks late" }).click();
  await expect(page.locator("#dl_out table")).toBeVisible();
  await expect(page.locator("#dl_out .timeline")).toBeVisible();
  await expect(page).toHaveURL(/preset=reported/);
  await page.reload();
  await expect(page.getByRole("button", { name: "reported three weeks late" })).toHaveAttribute("aria-pressed", "true");
});

test("search and graph walk by keyboard; state is in the URL", async ({ page }) => {
  await page.goto("/#explore?q=lost%20or%20stolen%20card%20liability&ret=bm25&node=12_CFR_1005_6_b_1");
  await expect(page.locator("#hits .hit").first()).toBeVisible();
  await expect(page.locator("#graphOut .graphsvg")).toBeVisible();
  const node = page.locator("#graphOut .node").first();
  const id = await node.getAttribute("data-id");
  await node.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`node=${id}`));
  await expect(page).toHaveURL(/ret=bm25/);
});

test("measured results render with intervals", async ({ page }) => {
  await page.goto("/#results");
  await expect(page.locator("#bars .bar").first()).toBeVisible();
  await expect(page.locator("#answers table")).toBeVisible();
  await expect(page.locator("#stores .stat").first()).toBeVisible();
});

test("theme toggle persists across reloads", async ({ page }) => {
  await page.goto("/#how");
  await page.emulateMedia({ colorScheme: "light" });
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByRole("button", { name: "Switch to light theme" })).toBeVisible();
});

for (const theme of ["light", "dark"]) {
  for (const tab of TABS) {
    test(`axe: ${tab} has no WCAG 2 A/AA violations (${theme})`, async ({ page }) => {
      await page.addInitScript((t) => localStorage.setItem("reglens.theme", t), theme);
      await page.goto(tab === "copilot" ? "/#copilot?run=C01" : `/#${tab}`);
      if (tab === "copilot") await expect(page.locator("#copilotOut h2")).toBeVisible();
      if (tab === "results") await expect(page.locator("#answers table")).toBeVisible();
      if (tab === "explore") await expect(page.locator("#graphOut .graphsvg")).toBeVisible();
      if (tab === "checker" || tab === "deadlines") await expect(page.locator(`#panel-${tab} [aria-live] :first-child`).first()).toBeVisible();
      await page.waitForTimeout(300);
      const { violations } = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
      expect(violations.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`)).toEqual([]);
    });
  }
}

test("mobile: no horizontal overflow, tabs scroll @mobile", async ({ page }) => {
  for (const tab of TABS) {
    await page.goto(`/#${tab}`);
    await page.waitForTimeout(250);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, tab).toBeLessThanOrEqual(1);
  }
});
