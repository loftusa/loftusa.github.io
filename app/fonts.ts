import { Fraunces, Newsreader } from "next/font/google";

// Display: Fraunces — a characterful "old-style with personality" variable serif.
// Used for the wordmark, headings, and editorial display moments.
export const fontDisplay = Fraunces({
  subsets: ["latin"],
  axes: ["opsz", "SOFT", "WONK"],
  style: ["normal", "italic"],
  variable: "--font-display",
  display: "swap",
});

// Body: Newsreader — a warm, literary text serif with real optical sizing.
export const fontBody = Newsreader({
  subsets: ["latin"],
  axes: ["opsz"],
  style: ["normal", "italic"],
  variable: "--font-body",
  display: "swap",
});
