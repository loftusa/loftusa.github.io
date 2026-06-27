// Single source for the FastAPI backend base URL.
// Build-time env wins (Vercel preview can point at staging); else hostname fallback
// preserves the exact behavior of the old vanilla-JS files.
const FALLBACK_PROD = "https://llm-resume-restless-thunder-9259.fly.dev";
const FALLBACK_LOCAL = "http://127.0.0.1:8000";

function hostnameFallback(): string {
  if (typeof window !== "undefined") {
    const h = window.location.hostname;
    if (h === "localhost" || h === "127.0.0.1") return FALLBACK_LOCAL;
  }
  return FALLBACK_PROD;
}

export const API_BASE: string =
  process.env.NEXT_PUBLIC_API_BASE && process.env.NEXT_PUBLIC_API_BASE.length > 0
    ? process.env.NEXT_PUBLIC_API_BASE
    : hostnameFallback();
