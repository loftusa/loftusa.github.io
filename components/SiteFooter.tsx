import Link from "next/link";
import styles from "./SiteFooter.module.css";

export default function SiteFooter() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <span className={styles.copy}>© Alex Loftus</span>
        <nav className={styles.links} aria-label="Footer">
          <a href="https://github.com/loftusa">GitHub</a>
          <a href="https://x.com/AlexLoftus19">Twitter</a>
          <Link href="/networks/">Networks</Link>
          <a href="/feed.xml">RSS</a>
        </nav>
      </div>
    </footer>
  );
}
