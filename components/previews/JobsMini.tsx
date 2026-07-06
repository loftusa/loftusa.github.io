import type { JobsPreview } from "./types";
import styles from "./ProjectPreviews.module.css";

// Same group hues as public/jobs/index.html:12,93
const GROUP_COLORS: Record<string, string> = {
  anthropic: "var(--accent)",
  interp: "#5f7d6e",
  frontier: "#6b7a94",
};

export default function JobsMini({ data }: { data: JobsPreview }) {
  if (data.latest.length === 0 || data.byLab.length === 0) {
    throw new Error("JobsMini: empty preview data");
  }
  const max = Math.max(...data.byLab.map((l) => l.n));
  return (
    <div className={styles.jobsMini}>
      <ul className={styles.jobsList}>
        {data.latest.map((j, i) => (
          <li key={i} className={styles.jobRow}>
            <span
              className={styles.jobCo}
              style={{ background: GROUP_COLORS[j.group] ?? GROUP_COLORS.frontier }}
            >
              {j.company}
            </span>
            <span className={styles.jobTitle}>{j.title}</span>
            {j.comp && <span className={styles.jobComp}>{j.comp}</span>}
          </li>
        ))}
      </ul>
      <div className={styles.labBars} aria-label={`Open roles per lab across ${data.byLab.length} labs`}>
        {data.byLab.map((l) => (
          <div key={l.company} className={styles.labBarCol} title={`${l.company}: ${l.n} open`}>
            <div
              className={styles.labBar}
              style={{ height: `${Math.max(8, Math.round((l.n / max) * 100))}%` }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
