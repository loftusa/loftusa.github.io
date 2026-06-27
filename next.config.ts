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
