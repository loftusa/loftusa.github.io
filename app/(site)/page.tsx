import fs from "node:fs";
import path from "node:path";
import AuthorHeader from "@/components/AuthorHeader";
import HomeTabs from "@/components/HomeTabs";
import ProjectPreviews from "@/components/previews/ProjectPreviews";
import type { PreviewsData } from "@/components/previews/types";
import { renderMarkdown } from "@/lib/content";
import styles from "./home.module.css";

function readContent(file: string): string {
  return fs.readFileSync(path.join(process.cwd(), "content", file), "utf8");
}

function readPreviews(): PreviewsData {
  const raw = fs.readFileSync(path.join(process.cwd(), "lib", "previews.json"), "utf8");
  return JSON.parse(raw) as PreviewsData;
}

export default async function HomePage() {
  const [aboutIntroHtml, aboutRestHtml, pubsHtml] = await Promise.all([
    renderMarkdown(readContent("home-about-intro.md")),
    renderMarkdown(readContent("home-about-rest.md")),
    renderMarkdown(readContent("publications.md")),
  ]);
  const previews = readPreviews();

  return (
    <main className={`page-main ${styles.home}`}>
      <AuthorHeader />
      <HomeTabs
        aboutIntroHtml={aboutIntroHtml}
        aboutRestHtml={aboutRestHtml}
        previews={<ProjectPreviews data={previews} />}
        pubsHtml={pubsHtml}
      />
    </main>
  );
}
