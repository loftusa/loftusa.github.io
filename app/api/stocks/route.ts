// app/api/stocks/route.ts
import { NextResponse } from "next/server";
import { getTickers } from "../../../lib/holdings";

async function fetchStooqSeries(ticker: string) {
  const url = `https://stooq.com/q/d/l/?s=${ticker}.us&i=d`;
  const resp = await fetch(url, { cache: "no-store" });
  const csv = await resp.text();
  const lines = csv.trim().split(/\n+/).slice(1);
  const rows = lines.map((l) => {
    const [date, , , , close] = l.split(",");
    return { date, close: Number(close) };
  });
  return rows.sort((a, b) => a.date.localeCompare(b.date));
}

export async function GET() {
  const tickers = getTickers();
  const weeklyChanges: { ticker: string; change: number }[] = [];

  for (const t of tickers) {
    try {
      const rows = await fetchStooqSeries(t);
      const n = rows.length;
      const last = rows[n - 1]?.close;
      const prev = rows[n - 6]?.close ?? rows[0]?.close;
      const change = last && prev ? ((last - prev) / prev) * 100 : 0;
      weeklyChanges.push({ ticker: t, change });
    } catch (e) {
      weeklyChanges.push({ ticker: t, change: 0 });
    }
  }

  return NextResponse.json({ weeklyChanges });
}
