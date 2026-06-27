import type { MetadataRoute } from "next";
import { getAllPostsMeta } from "@/lib/content";

const BASE = "https://alex-loftus.com";

export default function sitemap(): MetadataRoute.Sitemap {
  const staticRoutes = [
    "/",
    "/cv/",
    "/publications/",
    "/year-archive/",
    "/networks/",
    "/networks/affiliations/",
    "/networks/analyses/",
    "/networks/affiliations/analyses/",
  ];
  const posts = getAllPostsMeta().map((p) => ({
    url: BASE + p.permalink,
    lastModified: p.date || undefined,
  }));
  return [...staticRoutes.map((r) => ({ url: BASE + r })), ...posts];
}
