import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  trailingSlash: true,

  // The 4 /networks/ pages are self-contained vanilla-D3 apps assembled into
  // static HTML under public/_networks/ (see scripts/build_networks_html.mjs).
  async rewrites() {
    return [
      { source: "/networks", destination: "/_networks/papers.html" },
      { source: "/networks/affiliations", destination: "/_networks/affiliations.html" },
      { source: "/networks/analyses", destination: "/_networks/papers-analyses.html" },
      { source: "/networks/affiliations/analyses", destination: "/_networks/affiliations-analyses.html" },
      { source: "/talkmap", destination: "/talkmap/map.html" },
      // Bay Area rental scout — self-contained Leaflet page under public/houses/.
      { source: "/houses", destination: "/houses/index.html" },
      // STI risk data page — self-contained Tufte page under public/sti/ (noindex, unlisted).
      { source: "/sti", destination: "/sti/index.html" },
      // Partner preferences checklist — self-contained page under public/klist/ (noindex,
      // unlisted). Submissions POST to the Fly backend; /klist/admin is the bearer-gated viewer.
      { source: "/klist", destination: "/klist/index.html" },
      { source: "/klist/admin", destination: "/klist/admin.html" },
      // The 2 /perfumes/ pages are assembled from _pages/perfumes*.html into
      // public/_perfumes/ (see scripts/build_perfumes_html.mjs); assets already
      // live under public/assets/. atlas = fullscreen canvas app; analyses = text.
      { source: "/perfumes", destination: "/_perfumes/atlas.html" },
      { source: "/perfumes/analyses", destination: "/_perfumes/analyses.html" },
      // /red-teaming/* = the NDA "Mangrove" viewer. Source stays in the PRIVATE
      // loftusa/red-teaming repo and is deployed as a SEPARATE Vercel project
      // (aol-red-teaming.vercel.app, static docs/). Proxy to it so NO NDA content
      // ever lands in this public repo. Replaces the old GitHub-Pages project site
      // that went dark when the apex moved to Vercel. See memory: mangrove-website-split.
      { source: "/red-teaming", destination: "https://aol-red-teaming.vercel.app" },
      { source: "/red-teaming/:path*", destination: "https://aol-red-teaming.vercel.app/:path*" },
    ];
  },

  // URL parity with the old Jekyll site (redirect-from + the /coauthorship→/networks move).
  async redirects() {
    return [
      { source: "/about", destination: "/", permanent: true },
      { source: "/about.html", destination: "/", permanent: true },
      { source: "/resume", destination: "/cv/", permanent: true },
      { source: "/coauthorship", destination: "/networks/", permanent: true },
      { source: "/coauthorship/affiliations", destination: "/networks/affiliations/", permanent: true },
      { source: "/coauthorship/analyses", destination: "/networks/analyses/", permanent: true },
      {
        source: "/coauthorship/affiliations/analyses",
        destination: "/networks/affiliations/analyses/",
        permanent: true,
      },
      { source: "/wordpress/blog-posts", destination: "/year-archive/", permanent: true },
      { source: "/talks", destination: "/publications/", permanent: true },
      { source: "/teaching", destination: "/", permanent: true },
      { source: "/portfolio", destination: "/", permanent: true },
      { source: "/talkmap.html", destination: "/talkmap/map.html", permanent: true },
    ];
  },
};

export default nextConfig;
