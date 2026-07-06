import type { Metadata } from "next";
import ChatWidget from "@/components/ChatWidget";

export const metadata: Metadata = {
  title: "Chat",
  description: "Ask a retrieval-augmented chatbot about Alex Loftus's work.",
};

export default function ChatPage() {
  return (
    <main className="page-main">
      <div className="prose">
        <h1>Ask me anything</h1>
        <p>
          A retrieval-augmented chatbot, grounded in my CV, papers, and this
          site. Conversations are logged.
        </p>
      </div>
      <ChatWidget />
    </main>
  );
}
