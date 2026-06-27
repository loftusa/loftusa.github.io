import type { Metadata } from "next";
import Link from "next/link";
import { getAllPostsMeta, type PostMeta } from "@/lib/content";
import { formatDate, yearOf } from "@/lib/date";
import styles from "./year-archive.module.css";

export const metadata: Metadata = {
  title: "Blog",
  description: "Writing by Alex Loftus, by year.",
};

export default function YearArchivePage() {
  const posts = getAllPostsMeta();
  const groups: { year: string; posts: PostMeta[] }[] = [];
  for (const p of posts) {
    const y = yearOf(p.date);
    const last = groups[groups.length - 1];
    if (last && last.year === y) last.posts.push(p);
    else groups.push({ year: y, posts: [p] });
  }

  return (
    <main className="page-main reading">
      <h1>Blog</h1>
      {groups.map((g) => (
        <section key={g.year} className={styles.group}>
          <h2 id={g.year} className={styles.year}>
            {g.year}
          </h2>
          <ul className={styles.list}>
            {g.posts.map((p) => (
              <li key={p.permalink} className={styles.item}>
                <span className={styles.date}>{formatDate(p.date)}</span>
                <Link href={p.permalink} className={styles.title}>
                  {p.title}
                </Link>
                {p.excerpt && <p className={styles.excerpt}>{p.excerpt}</p>}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </main>
  );
}
