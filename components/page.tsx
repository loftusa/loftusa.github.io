// app/stocks/page.tsx
import StocksTufte from "@/components/StocksTufte";

export const revalidate = 0; // no cache; always fresh

export default async function Page() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL}/api/stocks`, { cache: "no-store" });
  const { weeklyChanges } = await res.json(); // [{ticker, change}]
  return <StocksTufte data={weeklyChanges} />;
}
