import fs from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import { renderMarkdown } from "@/lib/content";
import PuzzleForm from "@/components/PuzzleForm";

export const metadata: Metadata = {
  title: "Happy Birthday, Aina!",
  robots: { index: false },
};

export default async function HappyBirthdayPage() {
  const md = fs.readFileSync(path.join(process.cwd(), "content", "happybirthday.md"), "utf8");
  const html = await renderMarkdown(md);
  return (
    <main className="page-main reading">
      <h1>Happy Birthday, Aina!</h1>
      <div className="prose" dangerouslySetInnerHTML={{ __html: html }} />
      <PuzzleForm />
    </main>
  );
}
