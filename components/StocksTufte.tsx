// components/StocksTufte.tsx
import React, { useMemo } from "react";

type Row = { ticker: string; change: number };

export default function StocksTufte({ data }: { data: Row[] }) {
  const sorted = useMemo(() => [...data].sort((a, b) => b.change - a.change), [data]);
  const extent = useMemo(() => {
    const vals = data.map(d => d.change);
    const pad = 2;
    return { min: Math.min(-20, Math.min(...vals) - pad), max: Math.max(20, Math.max(...vals) + pad) };
  }, [data]);
  const scale = (v: number) => {
    const { min, max } = extent;
    const pct = (Math.max(min, Math.min(max, v)) - min) / (max - min || 1);
    return `${pct * 100}%`;
  };

  return (
    <div className="p-6 text-gray-900">
      <div className="mb-1 text-2xl font-semibold">Last Week Price Change (5-Day %)</div>
      <div className="mb-6 text-sm opacity-70 max-w-3xl">
        Minimalist, Tufte-style comparison. Sorted best→worst.
      </div>
      <div className="rounded-2xl shadow-sm border p-4">
        <div className="mb-2 text-base font-medium">Ranking (Dot Plot)</div>
        <div className="text-xs opacity-60 -mt-2 mb-1">
          Scale: {extent.min.toFixed(1)}% to {extent.max.toFixed(1)}%
        </div>
        <div className="space-y-2">
          {sorted.map((d, idx) => {
            const good = d.change >= 0;
            return (
              <div key={d.ticker} className="flex items-center gap-3">
                <div className="w-10 shrink-0 tabular-nums font-semibold">{idx + 1}.</div>
                <div className="w-16 shrink-0 font-medium">{d.ticker}</div>
                <div className="relative h-7 grow border-y">
                  <div className="absolute left-1/2 top-0 bottom-0 border-l" />
                  <div
                    className="absolute -translate-x-1/2 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full"
                    style={{ left: scale(d.change), background: good ? "#0a7d32" : "#b00020" }}
                  />
                </div>
                <div className={`w-20 text-right tabular-nums ${good ? "text-emerald-700" : "text-rose-700"}`}>
                  {`${d.change.toFixed(2)}%`}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
