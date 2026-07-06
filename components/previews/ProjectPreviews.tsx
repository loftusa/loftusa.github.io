import Link from "next/link";
import type { ReactNode } from "react";
import HousesMini from "./HousesMini";
import JobsMini from "./JobsMini";
import NetworksMini from "./NetworksMini";
import type { PreviewsData } from "./types";
import styles from "./ProjectPreviews.module.css";

const fmt = (n: number) => n.toLocaleString("en-US");

function TitleIcon({ children }: { children: ReactNode }) {
  return (
    <svg
      className={styles.titleIcon}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

const ICONS: Record<string, ReactNode> = {
  houses: (
    <TitleIcon>
      <path d="M3.5 11.5 12 4.5l8.5 7" />
      <path d="M6 10.5V19h12v-8.5" />
    </TitleIcon>
  ),
  jobs: (
    <TitleIcon>
      <rect x="4" y="8" width="16" height="11" rx="1.5" />
      <path d="M9.5 8V6.8A1.8 1.8 0 0 1 11.3 5h1.4a1.8 1.8 0 0 1 1.8 1.8V8" />
    </TitleIcon>
  ),
  networks: (
    <TitleIcon>
      <circle cx="5.5" cy="6" r="2.2" />
      <circle cx="18.5" cy="6" r="2.2" />
      <circle cx="12" cy="18" r="2.2" />
      <path d="M7.5 7.6 10.8 16M16.5 7.6 13.2 16M7.7 6h8.6" />
    </TitleIcon>
  ),
};

function Card({ href, title, icon, foot, children }: {
  href: string; title: string; icon: ReactNode; foot: string; children: ReactNode;
}) {
  return (
    <article className={styles.card}>
      <h3 className={styles.cardTitle}>{icon}{title}</h3>
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
          icon={ICONS.houses}
          foot={`${fmt(houses.meta.n_scouted)} scouted · $${fmt(houses.meta.price_min)}–$${fmt(houses.meta.price_max)} · med $${fmt(houses.meta.price_med)}`}
        >
          <HousesMini data={houses} />
        </Card>
        <Card
          href="/jobs/"
          title="Jobs"
          icon={ICONS.jobs}
          foot={`${fmt(jobs.meta.open)} open · ${jobs.meta.n_labs} labs · refreshed daily`}
        >
          <JobsMini data={jobs} />
        </Card>
        <Card
          href="/networks/"
          title="Networks"
          icon={ICONS.networks}
          foot={`${networks.meta.n_nodes} researchers · ${networks.meta.n_links} co-authorships`}
        >
          <NetworksMini data={networks} />
        </Card>
      </div>
    </section>
  );
}
