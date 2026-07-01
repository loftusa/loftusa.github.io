// Format an ISO date in UTC so a "2024-10-10" frontmatter date never slips a day
// due to the server/browser local timezone.
export function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function yearOf(iso: string): string {
  if (!iso) return "Undated";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "Undated" : String(d.getUTCFullYear());
}
