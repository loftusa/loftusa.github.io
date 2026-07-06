import Link from "next/link";
import styles from "./SiteHeader.module.css";
import { NetworkIcon, HouseIcon, BriefcaseIcon } from "./icons";

const NAV = [
  { href: "/networks/", label: "Networks", Icon: NetworkIcon },
  { href: "/houses/", label: "Houses", Icon: HouseIcon },
  { href: "/jobs/", label: "Jobs", Icon: BriefcaseIcon },
  { href: "/year-archive/", label: "Writing", Icon: null },
  { href: "/cv/", label: "CV", Icon: null },
];

export default function SiteHeader() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <Link href="/" className={styles.brand}>
          Alex&nbsp;Loftus
        </Link>
        <nav className={styles.nav} aria-label="Primary">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href} className={styles.link}>
              {n.Icon && <n.Icon className={styles.navIcon} />}
              {n.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
