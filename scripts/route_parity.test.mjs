// URL-parity guards for links that live OUTSIDE this repo and can never be
// updated — e.g. URLs printed inside submitted PDFs. If one of these breaks,
// people following old links hit 404s (next.config.ts owns the redirects).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repoRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const config = readFileSync(path.join(repoRoot, "next.config.ts"), "utf8");

test("/thesis redirects to the thesis PDF (URL is printed in the PDF itself)", () => {
  const redirect = /source:\s*"\/thesis"[\s\S]{0,120}?destination:\s*"(\/files\/[^"]+)"/.exec(config);
  assert.ok(redirect, 'next.config.ts must contain a redirect for source "/thesis"');
  const pdfPath = path.join(repoRoot, "public", redirect[1]);
  assert.ok(existsSync(pdfPath), `redirect destination ${redirect[1]} must exist under public/`);
});
