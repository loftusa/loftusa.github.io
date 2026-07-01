// Assemble the 4 self-contained /networks/ pages from the Jekyll sources into
// static HTML under public/_networks/, served via next.config rewrites.
// These pages are standalone vanilla-D3 apps; we reproduce them byte-faithfully
// (fullscreen shell + page body + expanded tabs nav). Re-run after editing the
// _pages/coauthorship*.html sources:  node scripts/build_networks_html.mjs
import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const PAGES = path.join(ROOT, "_pages");
const OUT = path.join(ROOT, "public", "_networks");

const NETS = [
  { id: "papers", label: "papers", base: "/networks/" },
  { id: "careers", label: "careers", base: "/networks/affiliations/" },
];

// Faithful port of _includes/coauthorship-tabs.html (the only Liquid in these pages).
function renderTabs(net, view) {
  const base = NETS.find((n) => n.id === net).base;
  const netRow = NETS.map((n) => {
    if (n.id === net) return `<span class="tab on">${n.label}</span>`;
    if (view === "analyses") return `<a class="tab" href="${n.base}analyses/">${n.label}</a>`;
    return `<a class="tab" href="${n.base}">${n.label}</a>`;
  }).join("\n      ");
  const mapTab = view === "map" ? `<span class="tab on">map</span>` : `<a class="tab" href="${base}">map</a>`;
  const anTab =
    view === "analyses" ? `<span class="tab on">analyses</span>` : `<a class="tab" href="${base}analyses/">analyses</a>`;
  return `<nav class="page-tabs">
  <div class="tab-row" aria-label="network">
      ${netRow}
  </div>
  <div class="tab-row" aria-label="view">
    ${mapTab}${anTab}
  </div>
</nav>`;
}

function parseFrontmatter(src) {
  const m = src.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!m) return { data: {}, body: src };
  const data = {};
  for (const line of m[1].split("\n")) {
    const mm = line.match(/^(\w+):\s*(.*)$/);
    if (mm) data[mm[1]] = mm[2].replace(/^["']|["']$/g, "");
  }
  return { data, body: m[2] };
}

// Faithful port of _layouts/fullscreen.html.
function shell(title, description, body) {
  const desc = description ? `\n  <meta name="description" content="${description}">` : "";
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
  <title>${title} · Alex Loftus</title>${desc}
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0; width: 100%; height: 100%;
      background: #faf8f3; color: #2b2b2b;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
      -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
      overflow: hidden;
    }
  </style>
</head>
<body>
${body}
</body>
</html>
`;
}

const MAP = {
  "coauthorship.html": "papers.html",
  "coauthorship-affiliations.html": "affiliations.html",
  "coauthorship-analyses.html": "papers-analyses.html",
  "coauthorship-affiliations-analyses.html": "affiliations-analyses.html",
};

const INCLUDE_RE = /\{%\s*include coauthorship-tabs\.html\s+network="([^"]+)"\s+view="([^"]+)"\s*%\}/;

fs.mkdirSync(OUT, { recursive: true });
for (const [srcName, outName] of Object.entries(MAP)) {
  const src = fs.readFileSync(path.join(PAGES, srcName), "utf8");
  const { data, body: rawBody } = parseFrontmatter(src);
  const incl = rawBody.match(INCLUDE_RE);
  if (!incl) throw new Error(`no coauthorship-tabs include found in ${srcName}`);
  let body = rawBody.replace(INCLUDE_RE, renderTabs(incl[1], incl[2]));
  body = body.replace(/\{\{\s*site\.baseurl\s*\}\}/g, "");
  if (/\{%|\{\{/.test(body)) throw new Error(`unexpanded Liquid remains in ${srcName}`);
  fs.writeFileSync(path.join(OUT, outName), shell(data.title || "Networks", data.description || "", body.trim()));
  console.log(`wrote public/_networks/${outName}  (from ${srcName}; net=${incl[1]} view=${incl[2]})`);
}
