// Assemble the 2 /perfumes/ pages from their Jekyll sources into static HTML
// under public/_perfumes/, served via next.config rewrites. Mirrors
// scripts/build_networks_html.mjs. The atlas page uses the fullscreen shell;
// the analyses page uses the bare (scrolling) shell — both faithful ports of
// _layouts/{fullscreen,bare}.html (favicon corrected to /favicon.svg, which is
// the file that actually exists, matching the networks port). All page assets
// (CSS/JS/JSON) already live under public/assets/ and are referenced by
// absolute /assets/... paths, so nothing else needs moving.
// Re-run after editing the _pages/perfumes*.html sources:
//   node scripts/build_perfumes_html.mjs
import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const PAGES = path.join(ROOT, "_pages");
const OUT = path.join(ROOT, "public", "_perfumes");

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

// Shared <head> (faithful to _layouts/*.html; favicon → /favicon.svg).
function head(title, description) {
  const desc = description ? `\n  <meta name="description" content="${description}">` : "";
  return `  <meta charset="utf-8">
  <!-- no maximum-scale: never block browser pinch-zoom of the text UI (WCAG 1.4.4 Resize Text) -->
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title} · Alex Loftus</title>${desc}
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">`;
}

// Faithful port of _layouts/fullscreen.html (overflow:hidden canvas app).
function fullscreen(title, description, body) {
  return `<!doctype html>
<html lang="en">
<head>
${head(title, description)}
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

// Faithful port of _layouts/bare.html (scrolling text page).
function bare(title, description, body) {
  return `<!doctype html>
<html lang="en">
<head>
${head(title, description)}
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0; width: 100%;
      background: #faf8f3; color: #38332e;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
      -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
    }
  </style>
</head>
<body>
${body}
</body>
</html>
`;
}

const PAGESPEC = [
  { src: "perfumes.html", out: "atlas.html", layout: fullscreen },
  { src: "perfumes-analyses.html", out: "analyses.html", layout: bare },
];

fs.mkdirSync(OUT, { recursive: true });
for (const p of PAGESPEC) {
  const src = fs.readFileSync(path.join(PAGES, p.src), "utf8");
  const { data, body } = parseFrontmatter(src);
  const html = p.layout(data.title || "", data.description || "", body.trim());
  fs.writeFileSync(path.join(OUT, p.out), html);
  console.log(`wrote public/_perfumes/${p.out}  (${(html.length / 1024).toFixed(1)} KB, from _pages/${p.src})`);
}
