// app/api/stocks/route.ts
import { NextResponse } from "next/server";

// Simple, reliable: fetch last ~6 daily closes from Stooq (no key) and compute ~5D change.
// For stronger SLAs, swap to Finnhub/Tiingo/Polygon later.
const TICKERS = ["APLD","CRM","IREN","EQT","GOOG","INTC","ASML","CORZ","NVDA","CEG","MOD","VST","TSM","META","AVGO","OKLO","PLTR"];

async function stooqSeries(ticker: string) {
  const url = `https://stooq.com/q/d/l/?s=${ticker}.us&i=d`;
  const csv = await fetch(url).then(r => r.text());
  const lines = csv.trim().split(/\n+/).slice(1);
  const rows = lines.map(l => {
    const [date,, , , close] = l.split(",");
    return { date, close: Number(close) };
  });
  return rows.sort((a, b) => a.date.localeCompare(b.date));
}

export async function GET() {
  const out: { ticker: string; change: number }[] = [];
  for (const t of TICKERS) {
    try {
      const rows = await stooqSeries(t);
      const n = rows.length;
      const last = rows[n - 1]?.close;
      const prev = rows[n - 6]?.close ?? rows[0]?.close;
      const change = (last && prev) ? ((last - prev) / prev) * 100 : 0;
      out.push({ ticker: t, change });
    } catch {
      // If a symbol is missing on Stooq, keep it but with 0 change; you can log internally
      out.push({ ticker: t, change: 0 });
    }
  }
  return NextResponse.json({ weeklyChanges: out });
}
