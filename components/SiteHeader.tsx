import Link from "next/link";
import styles from "./SiteHeader.module.css";

const NAV = [
  { href: "/networks/", label: "Networks" },
  { href: "/houses/", label: "Houses" },
  { href: "/jobs/", label: "Jobs" },
  { href: "/year-archive/", label: "Writing" },
  { href: "/cv/", label: "CV" },
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
              {n.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
