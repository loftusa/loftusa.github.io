"use client";

import { useState } from "react";
import styles from "./Disqus.module.css";

// Disqus shortname from the old _config.yml. NOTE: verify this is the real
// shortname in the Disqus admin — "alex-loftus.com" (with a dot) is unusual.
const SHORTNAME = "alex-loftus-com";

declare global {
  interface Window {
    disqus_config?: () => void;
  }
}

export default function Disqus({
  identifier,
  title,
}: {
  identifier: string;
  title: string;
}) {
  const [loaded, setLoaded] = useState(false);

  function load() {
    if (loaded) return;
    setLoaded(true);
    window.disqus_config = function (this: { page: { url: string; identifier: string; title: string } }) {
      this.page.url = window.location.href;
      this.page.identifier = identifier;
      this.page.title = title;
    };
    const s = document.createElement("script");
    s.src = `https://${SHORTNAME}.disqus.com/embed.js`;
    s.setAttribute("data-timestamp", String(+new Date()));
    document.body.appendChild(s);
  }

  return (
    <section className={styles.comments}>
      <h2 className={styles.title}>Comments</h2>
      {!loaded ? (
        <button className={styles.loadBtn} onClick={load}>
          Load comments →
        </button>
      ) : (
        <div id="disqus_thread" />
      )}
    </section>
  );
}
