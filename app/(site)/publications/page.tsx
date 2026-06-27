import fs from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import { renderMarkdown } from "@/lib/content";

export const metadata: Metadata = {
  title: "Publications",
  description: "Talks & publications by Alex Loftus.",
};

export default async function PublicationsPage() {
  const md = fs.readFileSync(path.join(process.cwd(), "content", "publications.md"), "utf8");
  const html = await renderMarkdown(md);
  return (
    <main className="page-main">
      <div className="prose" dangerouslySetInnerHTML={{ __html: html }} />
    </main>
  );
}
