"use client";

import Link from "next/link";
import styles from "./ProjectPreviews.module.css";

// Real, honest suggested prompts — the RAG bot answers these from my CV,
// papers, and site. Clicking one scrolls to the single chat widget at the
// top of the About tab and pre-fills it (see ChatWidget's `site-chat-focus`
// listener); no second chat box is rendered.
const PROMPTS = [
  "What are you working on now?",
  "Summarize your research.",
  "How do I get in touch?",
];

function focusChat(prompt?: string) {
  window.dispatchEvent(
    new CustomEvent("site-chat-focus", { detail: { prompt } })
  );
}

function ChatIcon() {
  return (
    <svg
      className={styles.chatIcon}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4.5 5.5h15a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H10l-4 3.5V15.5H4.5a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1Z" />
      <path d="M8 9.5h8M8 12h5" />
    </svg>
  );
}

export default function ChatCard() {
  return (
    <article className={styles.chatCard}>
      <div className={styles.chatMain}>
        <h3 className={styles.cardTitle}>
          <ChatIcon />
          Ask me anything
        </h3>
        <p className={styles.chatLede}>
          A retrieval-augmented chatbot, grounded in my CV, papers, and this
          site, answers questions about my work — try:
        </p>
        <div className={styles.chips}>
          {PROMPTS.map((q) => (
            <button
              key={q}
              type="button"
              className={styles.chip}
              onClick={() => focusChat(q)}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
      <div className={styles.chatAside}>
        <p className={styles.chatStat}>zai-glm-4.7 · RAG over my CV &amp; papers</p>
        <Link href="/chat/" className={styles.chatOpen}>
          open the chat →
        </Link>
      </div>
    </article>
  );
}
