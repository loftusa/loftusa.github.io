import Link from "next/link";
import type { ReactNode } from "react";
import HousesMini from "./HousesMini";
import JobsMini from "./JobsMini";
import NetworksMini from "./NetworksMini";
import ChatCard from "./ChatCard";
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
      <h2 className={styles.label}>Recent Projects</h2>
      <Link href="/camp-sherman/" className={styles.houseProject}>
        <img src="/camp-sherman/assets/preview.jpg" alt="Wood house and stone chimney among the trees in Camp Sherman" width={800} height={525} loading="lazy" />
        <div>
          <h3>Camp Sherman</h3>
          <p>A house and its surroundings, rebuilt in Blender from architectural drawings and Oregon terrain data.</p>
          <span>Explore the house in 3D →</span>
        </div>
      </Link>
      <div className={styles.grid}>
        <Card
          href="/houses/"
          title="Houses"
          foot={`${fmt(houses.meta.n_scouted)} scouted · $${fmt(houses.meta.price_min)}–$${fmt(houses.meta.price_max)} · med $${fmt(houses.meta.price_med)}`}
        >
          <HousesMini data={houses} />
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
          <NetworksMini data={networks} />
        </Card>
      </div>
      <ChatCard />
    </section>
  );
}
