// app/stocks/page.tsx
import StocksTufte from "../../components/StocksTufte";

export const revalidate = 0; // no cache; always fresh

export default async function StocksPage() {
  const res = await fetch(`/api/stocks`, { cache: "no-store" });
  const { weeklyChanges } = await res.json(); // [{ticker, change}]
  const updated = new Date().toLocaleString();

  return (
    <div className="p-6">
      <div className="mb-2 text-sm text-gray-500">
        Last updated: {updated} · <a className="underline" href="/api/stocks">/api/stocks</a>
      </div>
      <StocksTufte data={weeklyChanges} />
    </div>
  );
}
