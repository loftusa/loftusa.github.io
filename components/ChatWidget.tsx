"use client";

import { useEffect, useRef, useState } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { API_BASE } from "@/lib/api";
import styles from "./ChatWidget.module.css";

type Msg = { role: "user" | "bot"; text: string; html: string };

function getOrCreateUserId(): string {
  try {
    let id = localStorage.getItem("chat_user_id");
    if (!id) {
      id =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
              const r = (Math.random() * 16) | 0;
              const v = c === "x" ? r : (r & 0x3) | 0x8;
              return v.toString(16);
            });
      localStorage.setItem("chat_user_id", id);
    }
    return id;
  } catch {
    return "anon";
  }
}

export default function ChatWidget() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Warm the Fly machine so it's ready by the time the user types.
  useEffect(() => {
    fetch(API_BASE + "/health").catch(() => {});
  }, []);

  // The "Ask me anything" card lower on the page points back up to this one
  // widget: it dispatches `site-chat-focus` (optionally with a suggested
  // prompt) instead of rendering a second, confusing chat box.
  useEffect(() => {
    function onFocus(e: Event) {
      const prompt = (e as CustomEvent<{ prompt?: string }>).detail?.prompt;
      if (prompt) setInput(prompt);
      const el = inputRef.current;
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        el.focus({ preventScroll: true });
      }
    }
    window.addEventListener("site-chat-focus", onFocus);
    return () => window.removeEventListener("site-chat-focus", onFocus);
  }, []);

  function scrollToBottom() {
    const el = messagesRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;

    setInput("");
    setMessages((m) => [
      ...m,
      { role: "user", text, html: "" },
      { role: "bot", text: "…", html: "" },
    ]);
    setBusy(true);

    let botText = "";
    let first = true;

    function setLastBot(next: Partial<Msg>) {
      setMessages((m) => {
        const copy = m.slice();
        copy[copy.length - 1] = { role: "bot", text: "", html: "", ...next };
        return copy;
      });
    }

    try {
      const res = await fetch(API_BASE + "/chat?logging=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, user_id: getOrCreateUserId() }),
      });
      if (res.status === 429) throw new Error("rate_limit");
      if (!res.ok || !res.body) throw new Error("API returned " + res.status);

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;
      while (!done) {
        const { value, done: streamDone } = await reader.read();
        done = streamDone;
        if (value) {
          if (first) {
            botText = "";
            first = false;
          }
          botText += decoder.decode(value, { stream: !done });
          const html = DOMPurify.sanitize(await marked.parse(botText));
          setLastBot({ text: botText, html });
          scrollToBottom();
        }
      }
    } catch (err) {
      const message =
        (err as Error).message === "rate_limit"
          ? "This conversation has gotten pretty long! Please refresh the page to start a new one."
          : "Something went wrong. Please try again.";
      setLastBot({ text: message });
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  return (
    <div className={styles.container}>
      <hr className={styles.rule} />
      <form className={styles.form} onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          className={styles.input}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything about my work…"
          autoComplete="off"
          disabled={busy}
        />
        <button className={styles.send} type="submit" aria-label="Send" disabled={busy}>
          →
        </button>
      </form>
      <div className={styles.messages} ref={messagesRef}>
        {messages.map((m, i) => (
          <div
            key={i}
            className={`${styles.msg} ${m.role === "user" ? styles.user : styles.bot}`}
          >
            {m.role === "user" && <span className={styles.usrLabel}>you</span>}
            {m.html ? (
              <div className={styles.text} dangerouslySetInnerHTML={{ __html: m.html }} />
            ) : (
              <div className={styles.text}>{m.text}</div>
            )}
          </div>
        ))}
      </div>
      <p className={styles.privacy}>(conversations are logged)</p>
    </div>
  );
}
