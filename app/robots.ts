import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/", disallow: ["/happybirthday/"] },
    sitemap: "https://alex-loftus.com/sitemap.xml",
    host: "https://alex-loftus.com",
  };
}
