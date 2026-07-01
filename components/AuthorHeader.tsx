import styles from "./AuthorHeader.module.css";

const SOCIALS = [
  { href: "mailto:alexloftus2004@gmail.com", label: "Email", external: false },
  { href: "https://github.com/loftusa", label: "GitHub", external: true },
  { href: "https://x.com/AlexLoftus19", label: "Twitter", external: true },
  { href: "https://www.linkedin.com/in/alex-loftus", label: "LinkedIn", external: true },
  {
    href: "https://scholar.google.com/citations?user=_Njcmm8AAAAJ",
    label: "Scholar",
    external: true,
  },
];

export default function AuthorHeader() {
  return (
    <header className={styles.author}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        className={styles.avatar}
        src="/images/me.png"
        alt="Alex Loftus"
        width={88}
        height={88}
      />
      <div className={styles.meta}>
        <h1 className={styles.name}>Alex Loftus</h1>
        <p className={styles.tagline}>AI-safety research &amp; engineering · interpretability</p>
        <nav className={styles.socials} aria-label="Profiles">
          {SOCIALS.map((s) => (
            <a
              key={s.label}
              href={s.href}
              target={s.external ? "_blank" : undefined}
              rel={s.external ? "noopener noreferrer" : undefined}
            >
              {s.label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}
