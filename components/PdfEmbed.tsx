import styles from "./PdfEmbed.module.css";

export default function PdfEmbed({
  src,
  title,
  heading,
}: {
  src: string;
  title: string;
  heading: string;
}) {
  return (
    <main className={styles.page}>
      <div className={styles.head}>
        <h1>{heading}</h1>
        <span className={styles.actions}>
          <a href={src} target="_blank" rel="noopener noreferrer">
            Open ↗
          </a>
          <a href={src} download>
            Download ↓
          </a>
        </span>
      </div>
      <object className={styles.frame} data={`${src}#view=FitH`} type="application/pdf" aria-label={title}>
        <p className={styles.fallback}>
          Your browser can&rsquo;t display the PDF inline.{" "}
          <a href={src} target="_blank" rel="noopener noreferrer">
            Open {title} in a new tab
          </a>
          .
        </p>
      </object>
    </main>
  );
}
