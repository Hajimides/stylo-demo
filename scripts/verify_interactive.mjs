#!/usr/bin/env node
/** Real-browser checks of the final standalone PCA explorer. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = path.resolve(process.argv[2] || path.join(root, "outputs"));
const htmlPath = path.join(outputDirectory, "pca_interactive.html");
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true });
const checks = [];
const errors = [];
const externalRequests = [];
let page;

try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, offline: true });
  page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (/^https?:/.test(request.url())) externalRequests.push(request.url());
  });
  const source = await readFile(htmlPath, "utf8");
  for (const placeholder of ["COLLECTION_CARDS", "FEATURE_BUTTONS", "PLOT_HTML", "ANALYSIS_DATA", "HOVER_TEMPLATE", "INITIAL_FEATURES", "FEATURE_SETTINGS_LABEL"]) {
    assert.ok(!source.includes(`__${placeholder}__`), `Unfilled placeholder: ${placeholder}`);
  }
  await page.goto(pathToFileURL(htmlPath).href);
  await page.waitForFunction(() => window.__STYLO_STATE__?.point_count === 43);
  assert.equal(await page.evaluate(() => window.__STYLO_STATE__.n_features), 100);
  assert.equal(await page.locator('[data-features="100"]').getAttribute("aria-pressed"), "true");
  checks.push("default_100_features");

  const expectedCounts = { "呐喊": 12, "彷徨": 11, "故事新编": 9, "沈从文": 11 };
  const settings = [];
  for (const features of [30, 100, 300]) {
    await page.locator(`[data-features="${features}"]`).click();
    await page.waitForFunction(n => window.__STYLO_STATE__.n_features === n, features);
    const observed = await page.evaluate(() => {
      const plot = document.getElementById("pca-plot");
      return {
        state: window.__STYLO_STATE__,
        counts: Object.fromEntries(plot.data.map(trace => [trace.name, trace.x.length])),
        finite: plot.data.every(trace => [...trace.x, ...trace.y].every(Number.isFinite)),
        traceCount: plot.data.length,
      };
    });
    assert.deepEqual(observed.counts, expectedCounts);
    assert.equal(observed.traceCount, 4);
    assert.equal(observed.state.point_count, 43);
    assert.ok(observed.finite);
    settings.push({ n_features: features, counts: observed.counts });
  }
  checks.push("all_feature_settings_have_43_points_and_four_collections");

  await page.locator('[data-features="100"]').click();
  await page.waitForFunction(() => window.__STYLO_STATE__.n_features === 100);
  await page.locator(".legend .traces .legendtoggle").first().click();
  await page.waitForFunction(() => document.getElementById("pca-plot").data[0].visible === "legendonly");
  checks.push("legend_toggle");
  await page.locator("#reset-view").click();
  await page.waitForFunction(() => document.getElementById("pca-plot").data.every(trace => trace.visible === true));
  checks.push("reset_restores_hidden_collection");

  await page.locator("#show-titles").check();
  await page.waitForFunction(() => document.getElementById("pca-plot").data.every(trace => trace.mode === "markers+text"));
  await page.locator("#show-titles").uncheck();
  await page.waitForFunction(() => document.getElementById("pca-plot").data.every(trace => trace.mode === "markers"));
  checks.push("title_toggle");

  const firstPoint = page.locator(".scatterlayer .trace .points .point").first();
  await firstPoint.scrollIntoViewIfNeeded();
  const markerBounds = await firstPoint.boundingBox();
  assert.ok(markerBounds, "Expected a visible PCA marker.");
  // Plotly deliberately places a transparent drag surface over its markers.
  // Move the real pointer to the marker coordinates through that surface.
  await page.mouse.move(markerBounds.x + markerBounds.width / 2, markerBounds.y + markerBounds.height / 2);
  await page.waitForFunction(() => document.querySelector(".hoverlayer")?.textContent.includes("作者"));
  const hoverText = await page.locator(".hoverlayer").textContent();
  for (const label of ["作者", "文集", "文档 ID", "原文字符数", "分词数", "PC1", "PC2", "来源"]) {
    assert.ok(hoverText.includes(label), `Missing hover field: ${label}`);
  }
  checks.push("real_pointer_hover_shows_document_metadata");

  const readRange = () => page.evaluate(() => [...document.getElementById("pca-plot").layout.xaxis.range]);
  const beforeZoom = await readRange();
  const plotArea = await page.evaluate(() => {
    const plot = document.getElementById("pca-plot");
    const bounds = plot.getBoundingClientRect();
    const size = plot._fullLayout._size;
    return { x: bounds.x + size.l + size.w / 2, y: bounds.y + size.t + size.h / 2 };
  });
  await page.mouse.move(plotArea.x, plotArea.y);
  await page.mouse.wheel(0, -500);
  await page.waitForFunction(previous => JSON.stringify(document.getElementById("pca-plot").layout.xaxis.range) !== JSON.stringify(previous), beforeZoom);
  const afterZoom = await readRange();
  assert.ok(afterZoom[1] - afterZoom[0] < beforeZoom[1] - beforeZoom[0]);
  checks.push("mouse_wheel_zoom");

  await page.mouse.move(plotArea.x, plotArea.y);
  await page.mouse.down();
  await page.mouse.move(plotArea.x + 80, plotArea.y + 30, { steps: 8 });
  await page.mouse.up();
  await page.waitForFunction(previous => JSON.stringify(document.getElementById("pca-plot").layout.xaxis.range) !== JSON.stringify(previous), afterZoom);
  checks.push("pointer_pan");

  await page.locator("#reset-view").click();
  await page.waitForFunction(() => document.getElementById("pca-plot").layout.xaxis.autorange === true && document.getElementById("pca-plot").layout.yaxis.autorange === true);
  const afterReset = await readRange();
  assert.ok(Math.abs(afterReset[0] - beforeZoom[0]) < 1e-9);
  assert.ok(Math.abs(afterReset[1] - beforeZoom[1]) < 1e-9);
  checks.push("reset_restores_axis_ranges");
  for (const closeButton of await page.locator(".notifier-close").all()) {
    await closeButton.click();
  }
  await page.locator(".notifier-note").waitFor({ state: "hidden" });
  await page.mouse.move(10, 10);
  await page.screenshot({ path: path.join(outputDirectory, "pca_interactive.png"), fullPage: true });

  assert.deepEqual(errors, []);
  assert.deepEqual(externalRequests, []);
  checks.push("no_javascript_errors_or_external_network_requests");
  const result = {
    status: "passed",
    browser: "Chromium",
    browser_version: browser.version(),
    offline: true,
    document_count: 43,
    settings,
    checks,
    javascript_errors: errors,
    external_requests: externalRequests,
    screenshot: "pca_interactive.png",
  };
  await writeFile(path.join(outputDirectory, "browser_validation.json"), JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  await writeFile(path.join(outputDirectory, "browser_validation.json"), JSON.stringify({ status: "failed", checks, javascript_errors: errors, external_requests: externalRequests, error: String(error) }, null, 2) + "\n");
  throw error;
} finally {
  await browser.close();
}
