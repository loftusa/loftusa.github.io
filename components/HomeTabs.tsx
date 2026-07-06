"use client";

import { useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import ChatWidget from "./ChatWidget";
import { formatDate } from "@/lib/date";
import styles from "./HomeTabs.module.css";

type PostMeta = { permalink: string; title: string; date: string; excerpt: string };
type Tab = "about" | "publications" | "blog";

const EXTERNAL = [
  { href: "/files/cv.pdf", label: "CV" },
  { href: "/files/submitted_thesis.pdf", label: "Thesis" },
  {
    href: "https://youtube.com/playlist?list=PLlP-93ntHnnu-ETNlIfelO9C6T8VrADAh",
    label: "Linear Algebra",
  },
];

export default function HomeTabs({
  aboutIntroHtml,
  aboutRestHtml,
  previews,
  pubsHtml,
  posts,
}: {
  aboutIntroHtml: string;
  aboutRestHtml: string;
  previews?: ReactNode;
  pubsHtml: string;
  posts: PostMeta[];
}) {
  const [tab, setTab] = useState<Tab>("about");

  useEffect(() => {
    const h = window.location.hash.replace("#", "") as Tab;
    if (h === "about" || h === "publications" || h === "blog") setTab(h);
  }, []);

  function select(t: Tab) {
    setTab(t);
    history.replaceState(null, "", "#" + t);
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.tabBar} role="tablist">
        {(["about", "publications", "blog"] as Tab[]).map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={tab === t}
            className={tab === t ? styles.active : undefined}
            onClick={() => select(t)}
          >
            {t === "about" ? "About" : t === "publications" ? "Publications" : "Blog"}
          </button>
        ))}
        {EXTERNAL.map((e) => (
          <a key={e.label} href={e.href} target="_blank" rel="noopener noreferrer">
            {e.label} ↗
          </a>
        ))}
      </div>

      {tab === "about" && (
        <div className={styles.panel}>
          <ChatWidget />
          <div className="prose" dangerouslySetInnerHTML={{ __html: aboutIntroHtml }} />
          {previews}
          <div className="prose" dangerouslySetInnerHTML={{ __html: aboutRestHtml }} />
        </div>
      )}

      {tab === "publications" && (
        <div className={`${styles.panel} prose`} dangerouslySetInnerHTML={{ __html: pubsHtml }} />
      )}

      {tab === "blog" && (
        <div className={styles.panel}>
          <h2>Recent Posts</h2>
          <ul className={styles.postList}>
            {posts.map((p) => (
              <li key={p.permalink} className={styles.postItem}>
                <span className={styles.postDate}>{formatDate(p.date)}</span>
                <Link href={p.permalink} className={styles.postTitle}>
                  {p.title}
                </Link>
                {p.excerpt && <p className={styles.postExcerpt}>{p.excerpt}</p>}
              </li>
            ))}
          </ul>
          <Link href="/year-archive/" className={styles.viewAll}>
            View all posts →
          </Link>
        </div>
      )}
    </div>
  );
}
