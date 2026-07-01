"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import styles from "./not-found.module.css";

export default function NotFound() {
  const [redirecting, setRedirecting] = useState(false);

  useEffect(() => {
    // /networks/<person-slug> router — forwards the 48 vanity URLs AND brand-new
    // self-joined members (whose static seat isn't built yet) to their "your seat"
    // view. Ports the old _pages/404.md inline router.
    const m = location.pathname.match(/^\/networks\/([a-z0-9][a-z0-9-]*)\/?$/);
    if (m && !["affiliations", "analyses"].includes(m[1])) {
      setRedirecting(true);
      location.replace(
        "/networks/affiliations/analyses/?p=" +
          encodeURIComponent(m[1].replace(/-/g, " ")) +
          "#your-seat"
      );
    }
  }, []);

  if (redirecting) return <main className="page-main" aria-busy="true" />;

  return (
    <main className={`page-main ${styles.wrap}`}>
      <p className={styles.code}>404</p>
      <h1 className={styles.title}>Page not found</h1>
      <p className={styles.sub}>Your pixels are in another canvas.</p>
      <p>
        <Link href="/" className={styles.home}>
          ← back to alex-loftus.com
        </Link>
      </p>
    </main>
  );
}
