import type { Metadata } from "next";
import { fontDisplay, fontBody } from "./fonts";
import "./tokens.css";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://alex-loftus.com"),
  title: {
    default: "Alex Loftus",
    template: "%s · Alex Loftus",
  },
  description:
    "Alex Loftus — ML / interpretability researcher. Research, writing, and interactive coauthorship networks.",
  alternates: {
    types: { "application/atom+xml": "/feed.xml" },
  },
  openGraph: {
    title: "Alex Loftus",
    description:
      "ML / interpretability researcher. Research, writing, and interactive coauthorship networks.",
    url: "https://alex-loftus.com",
    siteName: "Alex Loftus",
    type: "website",
  },
  twitter: {
    card: "summary",
    creator: "@AlexLoftus19",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${fontDisplay.variable} ${fontBody.variable}`}>
      <body>{children}</body>
    </html>
  );
}
