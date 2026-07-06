import Link from "next/link";
import styles from "./SiteHeader.module.css";
import { NetworkIcon, HouseIcon, BriefcaseIcon } from "./icons";

// One navigation for the whole site: home-page sections first, then a thin
// divider, then the project apps (with icons) and the CV page. The old
// on-page tab bar is gone — About/Publications switch via /#about and
// /#publications (HomeTabs listens for hashchange).
const SECTIONS = [
  { href: "/#about", label: "About" },
  { href: "/#publications", label: "Publications" },
  { href: "/year-archive/", label: "Writing" },
];

const PROJECTS = [
  { href: "/networks/", label: "Networks", Icon: NetworkIcon },
  { href: "/houses/", label: "Houses", Icon: HouseIcon },
  { href: "/jobs/", label: "Jobs", Icon: BriefcaseIcon },
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
          {SECTIONS.map((n) =>
            n.href.includes("#") ? (
              // plain <a>: next/link changes hashes via pushState, which never
              // fires the hashchange event HomeTabs relies on to switch sections
              <a key={n.href} href={n.href} className={styles.link}>
                {n.label}
              </a>
            ) : (
              <Link key={n.href} href={n.href} className={styles.link}>
                {n.label}
              </Link>
            )
          )}
          <span className={styles.navSep} aria-hidden="true" />
          {PROJECTS.map((n) => (
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
