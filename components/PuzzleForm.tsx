"use client";

import { useState } from "react";
import styles from "./PuzzleForm.module.css";

export default function PuzzleForm() {
  const [answer, setAnswer] = useState("");
  const [feedback, setFeedback] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (answer.trim() === "55834") {
      window.location.href = "/solution_page.html";
    } else {
      setFeedback(
        "Incorrect Input! Every time you get one wrong, a painting falls off the wall!"
      );
    }
  }

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <label htmlFor="answer" className={styles.label}>
        Enter your answer:
      </label>
      <div className={styles.row}>
        <input
          id="answer"
          name="answer"
          type="text"
          className={styles.input}
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          autoComplete="off"
        />
        <button type="submit" className={styles.button}>
          Submit
        </button>
      </div>
      {feedback && <p className={styles.feedback}>{feedback}</p>}
    </form>
  );
}
