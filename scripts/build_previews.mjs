#!/usr/bin/env node
// Build lib/previews.json — small data slices for the front-page "Projects,
// live" strip — from the committed artifacts behind /houses, /jobs, /networks.
// Runs at the start of `pnpm build` / `pnpm dev`. Fails loudly on shape drift:
// these inputs are committed artifacts that should always be valid.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SPRING_DEFAULT, HOME_DEFAULT } from "../lib/force-sim.mjs";

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");

export function parseWindowData(src, varName) {
  const marker = `window.${varName} = `;
  const i = src.indexOf(marker);
  if (i < 0) throw new Error(`window.${varName} assignment not found`);
  let body = src.slice(i + marker.length).trim();
  if (body.endsWith(";")) body = body.slice(0, -1);
  return JSON.parse(body);
}

function num(v, what) {
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`${what} is not a finite number: ${v}`);
  }
  return v;
}

// Mirrors the "driver" logic in public/houses/index.html:541-575 — the score
// component that most lifts a listing above the field, for the alex audience.
const DRIVER_KEYS = ["commute", "nice", "aesthetic", "nature", "value", "soft"];

function compScores(x) {
  const s = x.scores || {};
  return {
    commute: s.commute ?? 5, nice: s.nice ?? 5, aesthetic: s.aesthetic ?? 5,
    nature: s.nature ?? 5, value: s.value ?? 5,
    soft: x.bucket === "apt" ? (s.quiet ?? 5) : (s.social ?? 5),
  };
}

export function extractHouses(data) {
  const { meta, listings } = data;
  if (!Array.isArray(listings) || listings.length === 0) throw new Error("houses: listings empty");
  const weights = meta?.fit_weights?.alex;
  if (!weights || Object.keys(weights).length === 0) throw new Error("houses: meta.fit_weights.alex missing");
  const live = listings.filter((x) => !x.gone);
  if (live.length === 0) throw new Error("houses: all listings gone");
  // Validate all live listings before sorting to catch bad coordinates/fit on any listing.
  for (const x of live) {
    num(x.lat, `houses lat of ${x.id}`);
    num(x.lon, `houses lon of ${x.id}`);
    num(x.fit, `houses fit of ${x.id}`);
  }
  const scored = live.map(compScores);
  const means = {};
  for (const k of DRIVER_KEYS) {
    means[k] = scored.reduce((t, r) => t + r[k], 0) / scored.length;
  }
  const driver = (x) => {
    const sc = compScores(x);
    let bestKey = "commute";
    let bestVal = -Infinity;
    for (const k of DRIVER_KEYS) {
      const v = (weights[k] ?? 0) * (sc[k] - means[k]);
      if (v > bestVal) { bestVal = v; bestKey = k; }
    }
    if (bestKey === "soft") return x.bucket === "apt" ? "quiet" : "social";
    return bestKey;
  };
  const top = [...live].sort((a, b) => b.fit - a.fit).slice(0, 25).map((x) => ({
    lat: x.lat,
    lon: x.lon,
    fit: x.fit,
    pdisp: String(x.pdisp ?? `$${x.price}`),
    hood: String(x.hood ?? "").trim() || "unknown",
    driver: driver(x),
  }));
  return {
    meta: {
      n_scouted: num(meta.n_scouted, "houses n_scouted"),
      price_min: num(meta.price_min, "houses price_min"),
      price_max: num(meta.price_max, "houses price_max"),
      price_med: num(meta.price_med, "houses price_med"),
      generated: String(meta.generated ?? ""),
    },
    // Coordinates from public/houses/index.html:629-630
    anchors: [
      { lat: 37.7935, lon: -122.397, label: "Downtown SF" },
      { lat: 37.8703, lon: -122.268, label: "Berkeley · FAR Labs" },
    ],
    listings: top,
  };
}

export function extractJobs(data) {
  const { meta, jobs } = data;
  if (!Array.isArray(jobs) || jobs.length === 0) throw new Error("jobs: jobs empty");
  const companies = meta?.companies;
  if (!Array.isArray(companies) || companies.length === 0) throw new Error("jobs: meta.companies missing");
  const open = jobs.filter((j) => !j.closed);
  if (open.length === 0) throw new Error("jobs: no open jobs");
  // Same "newest" ordering as public/jobs/index.html:326
  const latest = [...open]
    .sort((a, b) => (b.published || b.first_seen || "").localeCompare(a.published || a.first_seen || ""))
    .slice(0, 6)
    .map((j) => ({
      company: String(j.company),
      group: String(j.group || "frontier"),
      title: String(j.title),
      comp: Array.isArray(j.comp) && j.comp.length === 2
        ? `$${Math.round(j.comp[0] / 1000)}–${Math.round(j.comp[1] / 1000)}K`
        : null,
      date: String(j.published || j.first_seen || ""),
    }));
  return {
    meta: {
      open: num(meta.open, "jobs meta.open"),
      n_labs: companies.length,
      generated: String(meta.date || meta.generated || ""),
    },
    latest,
    byLab: companies.map((c) => ({ company: c, n: open.filter((j) => j.company === c).length })),
  };
}

// Power-iteration estimate of the largest eigenvalue of the graph Laplacian.
// The networks mini relaxes positions with x ← x − (spring·L + home·I)(x − x0);
// that iteration diverges once spring·λ_max(L) + home ≥ 2. Assert a margin so
// graph densification fails the build here instead of exploding drags at runtime.
function laplacianLambdaMax(nNodes, links) {
  const deg = new Float64Array(nNodes);
  for (const [s, t] of links) { deg[s]++; deg[t]++; }
  let v = Float64Array.from({ length: nNodes }, (_, i) => 1 + (i % 7)); // fixed, non-uniform seed
  // Normalise initial vector
  let initSumSq = 0;
  for (let i = 0; i < nNodes; i++) initSumSq += v[i] * v[i];
  const initNorm = Math.sqrt(initSumSq);
  for (let i = 0; i < nNodes; i++) v[i] /= initNorm;
  let lambda = 0;
  for (let it = 0; it < 100; it++) {
    const w = new Float64Array(nNodes);
    for (let i = 0; i < nNodes; i++) w[i] = deg[i] * v[i];
    for (const [s, t] of links) { w[s] -= v[t]; w[t] -= v[s]; }
    // Manual sqrt-of-sum-of-squares (safe for large arrays; avoids Math.hypot argument limits)
    let sumSq = 0;
    for (let i = 0; i < nNodes; i++) sumSq += w[i] * w[i];
    const norm = Math.sqrt(sumSq);
    if (norm === 0) return 0;
    lambda = norm; // ||L v|| with ||v|| = 1 converges to λ_max
    for (let i = 0; i < nNodes; i++) v[i] = w[i] / norm;
  }
  return lambda;
}

export function extractNetworks(data) {
  const { nodes, links, communities, meta } = data;
  if (!Array.isArray(nodes) || nodes.length === 0) throw new Error("networks: nodes empty");
  if (!Array.isArray(links) || links.length === 0) throw new Error("networks: links empty");

  // Meta cross-check: footer numbers must not lie
  if (meta.n_nodes !== nodes.length) {
    throw new Error(`networks: meta.n_nodes=${meta.n_nodes} but nodes.length=${nodes.length}`);
  }
  if (meta.n_links !== links.length) {
    throw new Error(`networks: meta.n_links=${meta.n_links} but links.length=${links.length}`);
  }

  const index = new Map(nodes.map((n, i) => [n.id, i]));
  const outLinks = links.map((l) => {
    const s = index.get(l.source);
    const t = index.get(l.target);
    if (s === undefined || t === undefined) {
      throw new Error(`networks: link endpoint not found: ${l.source} -> ${l.target}`);
    }
    return [s, t];
  });

  // Stability guard: build fails here rather than producing an exploding drag at runtime
  const lambdaMax = laplacianLambdaMax(nodes.length, outLinks);
  const gain = SPRING_DEFAULT * lambdaMax + HOME_DEFAULT;
  if (gain >= 1.9) {
    throw new Error(
      `networks: force-sim stability margin exhausted — spring·λ_max + home = ${gain.toFixed(3)} ` +
      `(divergence at 2.0). Lower SPRING_DEFAULT in lib/force-sim.mjs or thin the graph.`,
    );
  }

  return {
    meta: { n_nodes: num(meta.n_nodes, "networks n_nodes"), n_links: num(meta.n_links, "networks n_links") },
    communities: (communities ?? []).map((c) => ({ id: c.id, label: String(c.label) })),
    nodes: nodes.map((n) => ({
      label: String(n.label),
      community: num(n.community ?? 0, `community of ${n.id}`),
      x: num(n.x, `x of ${n.id}`),
      y: num(n.y, `y of ${n.id}`),
    })),
    links: outLinks,
  };
}

export function buildPreviews({ housesSrc, jobsSrc, networksJson }) {
  return {
    houses: extractHouses(parseWindowData(housesSrc, "HOUSES_DATA")),
    jobs: extractJobs(parseWindowData(jobsSrc, "JOBS_DATA")),
    networks: extractNetworks(JSON.parse(networksJson)),
  };
}

// ---------------- CLI ----------------
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const read = (p) => fs.readFileSync(path.join(ROOT, p), "utf8");
  const out = buildPreviews({
    housesSrc: read("public/houses/data.js"),
    jobsSrc: read("public/jobs/data.js"),
    networksJson: read("public/assets/data/coauthorship.json"),
  });
  const json = JSON.stringify(out);
  if (json.length > 60_000) throw new Error(`previews.json unexpectedly large: ${json.length} bytes`);
  fs.writeFileSync(path.join(ROOT, "lib", "previews.json"), json);
  console.log(
    `lib/previews.json written: ${json.length} bytes — ` +
    `${out.houses.listings.length} listings, ${out.jobs.latest.length} jobs, ` +
    `${out.networks.nodes.length} nodes / ${out.networks.links.length} links`,
  );
}
