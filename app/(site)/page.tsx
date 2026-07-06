import fs from "node:fs";
import path from "node:path";
import AuthorHeader from "@/components/AuthorHeader";
import HomeTabs from "@/components/HomeTabs";
import { renderMarkdown, getAllPostsMeta } from "@/lib/content";
import styles from "./home.module.css";

function readContent(file: string): string {
  return fs.readFileSync(path.join(process.cwd(), "content", file), "utf8");
}

export default async function HomePage() {
  const [aboutIntroHtml, aboutRestHtml, pubsHtml] = await Promise.all([
    renderMarkdown(readContent("home-about-intro.md")),
    renderMarkdown(readContent("home-about-rest.md")),
    renderMarkdown(readContent("publications.md")),
  ]);
  const posts = getAllPostsMeta()
    .slice(0, 8)
    .map((p) => ({ permalink: p.permalink, title: p.title, date: p.date, excerpt: p.excerpt }));

  return (
    <main className={`page-main ${styles.home}`}>
      <AuthorHeader />
      <HomeTabs
        aboutIntroHtml={aboutIntroHtml}
        aboutRestHtml={aboutRestHtml}
        pubsHtml={pubsHtml}
        posts={posts}
      />
    </main>
  );
}
