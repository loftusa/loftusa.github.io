import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseWindowData, extractHouses, extractJobs, extractNetworks, buildPreviews,
} from "./build_previews.mjs";

// ---------- fixtures ----------
const listing = (over = {}) => ({
  id: "L1", price: 1500, pdisp: "$1,500", beds: 1, bucket: "apt",
  hood: "mission", lat: 37.76, lon: -122.42, fit: 7.0,
  scores: { nature: 5, quiet: 5, nice: 5, social: 5, value: 5, commute: 5, aesthetic: 5 },
  ...over,
});
const HOUSES = {
  meta: {
    n_scouted: 100, price_min: 800, price_max: 2900, price_med: 1700, generated: "2026-07-05",
    fit_weights: { alex: { nice: 0.17, nature: 0.15, soft: 0.13, value: 0.13, commute: 0.26, aesthetic: 0.16 } },
  },
  listings: [
    listing({ id: "L1", fit: 7.0 }),
    listing({ id: "L2", fit: 9.1, hood: "marina", scores: { nature: 9, quiet: 5, nice: 5, social: 5, value: 5, commute: 5, aesthetic: 5 } }),
    listing({ id: "L3", fit: 8.0, gone: true }),
    listing({ id: "L4", fit: 5.5, bucket: "room", scores: { nature: 5, quiet: 5, nice: 5, social: 9, value: 5, commute: 5, aesthetic: 5 } }),
  ],
};
const job = (over = {}) => ({
  company: "Anthropic", group: "anthropic", title: "Research Engineer",
  comp: [300000, 405000], published: "2026-07-01", first_seen: "2026-07-01", closed: false,
  ...over,
});
const JOBS = {
  meta: { open: 3, date: "2026-07-05", companies: ["Anthropic", "OpenAI"] },
  jobs: [
    job({ title: "Old", published: "2026-06-01" }),
    job({ title: "Newest", company: "OpenAI", group: "frontier", published: "2026-07-04", comp: null }),
    job({ title: "Closed", published: "2026-07-05", closed: true }),
    job({ title: "Mid", published: "2026-07-02" }),
  ],
};
const NET = {
  meta: { n_nodes: 3, n_links: 2 },
  communities: [{ id: 0, label: "EleutherAI" }, { id: 1, label: "David Bau" }],
  nodes: [
    { id: "a", label: "Ada", community: 0, x: -1.0, y: 0.5 },
    { id: "b", label: "Bo", community: 1, x: 0.2, y: -0.3 },
    { id: "c", label: "Cy", community: 1, x: 1.1, y: 1.2 },
  ],
  links: [{ source: "a", target: "b" }, { source: "b", target: "c" }],
};

// ---------- parseWindowData ----------
test("parseWindowData round-trips the JSON payload", () => {
  const src = `// header comment\nwindow.FOO_DATA = {"a": 1, "b": [2]};\n`;
  assert.deepEqual(parseWindowData(src, "FOO_DATA"), { a: 1, b: [2] });
});
test("parseWindowData throws when the marker is missing", () => {
  assert.throws(() => parseWindowData("var x = 1;", "FOO_DATA"), /FOO_DATA/);
});

// ---------- extractHouses ----------
test("extractHouses sorts by fit desc and drops gone listings", () => {
  const h = extractHouses(HOUSES);
  assert.deepEqual(h.listings.map((l) => l.fit), [9.1, 7.0, 5.5]);
  assert.equal(h.listings.length, 3);
  // L3 (fit 8.0, gone: true) must not appear
  assert.equal(h.listings.some((l) => l.fit === 8.0), false);
});
test("extractHouses computes drivers: lifted component wins; soft maps by bucket", () => {
  const h = extractHouses(HOUSES);
  assert.equal(h.listings[0].driver, "nature");   // L2: nature 9 vs field
  assert.equal(h.listings[2].driver, "social");   // L4: room bucket, social 9
});
test("extractHouses keeps meta numbers and both anchors", () => {
  const h = extractHouses(HOUSES);
  assert.equal(h.meta.n_scouted, 100);
  assert.equal(h.anchors.length, 2);
  assert.equal(h.anchors[0].label, "Downtown SF");
});
test("extractHouses throws on empty listings and missing weights", () => {
  assert.throws(() => extractHouses({ meta: HOUSES.meta, listings: [] }), /listings/);
  assert.throws(() => extractHouses({ meta: { ...HOUSES.meta, fit_weights: {} }, listings: HOUSES.listings }), /fit_weights/);
});
test("extractHouses throws on non-finite coordinates", () => {
  const bad = { meta: HOUSES.meta, listings: [listing({ lat: null })] };
  assert.throws(() => extractHouses(bad), /lat/);
});
test("extractHouses throws on bad coordinates in non-top-ranked listings", () => {
  // Create a fixture: one good high-fit listing, one bad low-fit listing.
  // Without pre-validation, bad listing would be dropped by slice(0, 25).
  const bad = {
    meta: HOUSES.meta,
    listings: [
      listing({ id: "L_good", fit: 9.0 }),
      listing({ id: "L_bad", fit: 1.0, lon: NaN }),
    ],
  };
  assert.throws(() => extractHouses(bad), /lon of L_bad/);
});

// ---------- extractJobs ----------
test("extractJobs filters closed and orders newest-first like the jobs page", () => {
  const j = extractJobs(JOBS);
  assert.deepEqual(j.latest.map((x) => x.title), ["Newest", "Mid", "Old"]);
});
test("extractJobs keeps comp null when absent and counts open roles per lab", () => {
  const j = extractJobs(JOBS);
  assert.equal(j.latest[0].comp, null);                  // "Newest" has comp: null
  assert.equal(j.latest[1].comp, "$300–405K");           // array [300000,405000] formatted like jobs page
  assert.deepEqual(j.byLab, [{ company: "Anthropic", n: 2 }, { company: "OpenAI", n: 1 }]);
  assert.equal(j.meta.n_labs, 2);
});
test("extractJobs throws when meta.companies is missing", () => {
  assert.throws(() => extractJobs({ meta: { open: 1 }, jobs: JOBS.jobs }), /companies/);
});
test("extractJobs throws on empty jobs array", () => {
  assert.throws(() => extractJobs({ meta: JOBS.meta, jobs: [] }), /jobs empty/);
});

// ---------- extractNetworks ----------
test("extractNetworks maps links to node-index pairs and keeps communities", () => {
  const n = extractNetworks(NET);
  assert.deepEqual(n.links, [[0, 1], [1, 2]]);
  assert.equal(n.nodes[0].label, "Ada");
  assert.deepEqual(n.communities[1], { id: 1, label: "David Bau" });
});
test("extractNetworks throws on unresolvable link endpoints and non-finite positions", () => {
  // Override meta counts to match the overridden arrays so the meta check passes first
  assert.throws(
    () => extractNetworks({ ...NET, meta: { n_nodes: 3, n_links: 1 }, links: [{ source: "a", target: "zzz" }] }),
    /zzz/,
  );
  const badNodes = [{ ...NET.nodes[0], x: undefined }, ...NET.nodes.slice(1)];
  assert.throws(() => extractNetworks({ ...NET, nodes: badNodes }), /x of/);
});
test("extractNetworks throws when meta counts disagree with array lengths", () => {
  assert.throws(
    () => extractNetworks({ ...NET, meta: { n_nodes: 99, n_links: 2 } }),
    /n_nodes/,
  );
  assert.throws(
    () => extractNetworks({ ...NET, meta: { n_nodes: 3, n_links: 99 } }),
    /n_links/,
  );
});
test("extractNetworks throws on a star graph that exhausts the force-sim stability margin", () => {
  // K_{1,40}: hub (node 0) linked to 40 leaves → λ_max ≈ 41, gain ≈ 2.56 > 1.9
  const starNodes = Array.from({ length: 41 }, (_, i) => ({
    id: String(i), label: `N${i}`, community: 0,
    x: i === 0 ? 0 : Math.cos((2 * Math.PI * i) / 40),
    y: i === 0 ? 0 : Math.sin((2 * Math.PI * i) / 40),
  }));
  const starLinks = Array.from({ length: 40 }, (_, i) => ({ source: "0", target: String(i + 1) }));
  const STAR_NET = { meta: { n_nodes: 41, n_links: 40 }, communities: [], nodes: starNodes, links: starLinks };
  assert.throws(() => extractNetworks(STAR_NET), /stability/);
});

// ---------- buildPreviews ----------
test("buildPreviews assembles all three sections from raw sources", () => {
  const out = buildPreviews({
    housesSrc: `window.HOUSES_DATA = ${JSON.stringify(HOUSES)};`,
    jobsSrc: `window.JOBS_DATA = ${JSON.stringify(JOBS)};`,
    networksJson: JSON.stringify(NET),
  });
  assert.equal(out.houses.listings.length, 3);
  assert.equal(out.jobs.latest.length, 3);
  assert.equal(out.networks.nodes.length, 3);
});
