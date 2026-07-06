"use client";

import { useEffect, useState, type ReactNode } from "react";
import ChatWidget from "./ChatWidget";
import styles from "./HomeTabs.module.css";

type Section = "about" | "publications";

// Section switching lives in the site header now (About / Publications link
// to /#about and /#publications) — this component just renders whichever
// section the hash names. Legacy hashes (e.g. #blog) fall back to About.
export default function HomeTabs({
  aboutIntroHtml,
  aboutRestHtml,
  previews,
  pubsHtml,
}: {
  aboutIntroHtml: string;
  aboutRestHtml: string;
  previews?: ReactNode;
  pubsHtml: string;
}) {
  const [section, setSection] = useState<Section>("about");

  useEffect(() => {
    function syncFromHash() {
      const h = window.location.hash.replace("#", "");
      setSection(h === "publications" ? "publications" : "about");
    }
    syncFromHash();
    window.addEventListener("hashchange", syncFromHash);
    return () => window.removeEventListener("hashchange", syncFromHash);
  }, []);

  return (
    <div className={styles.wrap}>
      {section === "about" && (
        <div className={styles.panel} id="about">
          <ChatWidget />
          <div className="prose" dangerouslySetInnerHTML={{ __html: aboutIntroHtml }} />
          {previews}
          <div className="prose" dangerouslySetInnerHTML={{ __html: aboutRestHtml }} />
        </div>
      )}

      {section === "publications" && (
        <div className={`${styles.panel} prose`} id="publications" dangerouslySetInnerHTML={{ __html: pubsHtml }} />
      )}
    </div>
  );
}
