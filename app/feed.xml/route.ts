import { getAllPostsMeta } from "@/lib/content";

export const dynamic = "force-static";

const BASE = "https://alex-loftus.com";

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function GET() {
  const posts = getAllPostsMeta();
  const updated = posts[0]?.date || "1970-01-01T00:00:00.000Z";
  const entries = posts
    .map(
      (p) => `  <entry>
    <title>${esc(p.title)}</title>
    <link href="${BASE}${p.permalink}"/>
    <id>${BASE}${p.permalink}</id>
    <updated>${p.date || updated}</updated>
    <summary>${esc(p.excerpt)}</summary>
  </entry>`
    )
    .join("\n");

  const xml = `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Alex Loftus</title>
  <subtitle>Writing by Alex Loftus</subtitle>
  <link href="${BASE}/feed.xml" rel="self"/>
  <link href="${BASE}/"/>
  <updated>${updated}</updated>
  <id>${BASE}/</id>
  <author><name>Alex Loftus</name></author>
${entries}
</feed>
`;

  return new Response(xml, {
    headers: { "Content-Type": "application/atom+xml; charset=utf-8" },
  });
}
