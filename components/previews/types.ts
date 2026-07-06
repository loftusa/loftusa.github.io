export type HousesPreview = {
  meta: { n_scouted: number; price_min: number; price_max: number; price_med: number; generated: string };
  anchors: { lat: number; lon: number; label: string }[];
  listings: { lat: number; lon: number; fit: number; pdisp: string; hood: string; driver: string }[];
};

export type JobsPreview = {
  meta: { open: number; n_labs: number; generated: string };
  latest: { company: string; group: string; title: string; comp: string | null; date: string }[];
  byLab: { company: string; n: number }[];
};

export type NetworksPreview = {
  meta: { n_nodes: number; n_links: number };
  communities: { id: number; label: string }[];
  nodes: { label: string; community: number; x: number; y: number }[];
  links: [number, number][];
};

export type PreviewsData = {
  houses: HousesPreview;
  jobs: JobsPreview;
  networks: NetworksPreview;
};
