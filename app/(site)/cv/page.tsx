import type { Metadata } from "next";
import PdfEmbed from "@/components/PdfEmbed";

export const metadata: Metadata = {
  title: "CV",
  description: "Curriculum vitae — Alex Loftus.",
};

export default function CvPage() {
  return <PdfEmbed src="/files/cv.pdf" title="CV" heading="Curriculum Vitae" />;
}
