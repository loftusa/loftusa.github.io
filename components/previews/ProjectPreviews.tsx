import Link from "next/link";
import type { ReactNode } from "react";
import JobsMini from "./JobsMini";
import type { PreviewsData } from "./types";
import styles from "./ProjectPreviews.module.css";

const fmt = (n: number) => n.toLocaleString("en-US");

function Card({ href, title, foot, children }: {
  href: string; title: string; foot: string; children: ReactNode;
}) {
  return (
    <article className={styles.card}>
      <h3 className={styles.cardTitle}>{title}</h3>
      <div className={styles.mini}>{children}</div>
      <p className={styles.foot}>{foot}</p>
      <Link href={href} className={styles.open}>open →</Link>
    </article>
  );
}

export default function ProjectPreviews({ data }: { data: PreviewsData }) {
  const { houses, jobs, networks } = data;
  return (
    <section className={styles.strip} aria-label="Live previews of Alex's project pages">
      <h2 className={styles.label}>Projects, live</h2>
      <div className={styles.grid}>
        <Card
          href="/houses/"
          title="Houses"
          foot={`${fmt(houses.meta.n_scouted)} scouted · $${fmt(houses.meta.price_min)}–$${fmt(houses.meta.price_max)} · med $${fmt(houses.meta.price_med)}`}
        >
          {/* HousesMini lands in Task 6 */}
          <div />
        </Card>
        <Card
          href="/jobs/"
          title="Jobs"
          foot={`${fmt(jobs.meta.open)} open · ${jobs.meta.n_labs} labs · refreshed daily`}
        >
          <JobsMini data={jobs} />
        </Card>
        <Card
          href="/networks/"
          title="Networks"
          foot={`${networks.meta.n_nodes} researchers · ${networks.meta.n_links} co-authorships`}
        >
          {/* NetworksMini lands in Task 7 */}
          <div />
        </Card>
      </div>
    </section>
  );
}
