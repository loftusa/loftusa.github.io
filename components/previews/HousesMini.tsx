"use client";

import { useMemo, useRef, useState } from "react";
import type { HousesPreview } from "./types";
import styles from "./ProjectPreviews.module.css";

// Same driver hues as public/houses/index.html:15
const DRIVER_COLORS: Record<string, string> = {
  nature: "#4f8a55", quiet: "#5f7189", nice: "#7b6091", social: "#bd8a3a",
  value: "#3f867e", commute: "#8a8378", aesthetic: "#b0654d",
};

const W = 200;
const H = 128;
const PAD = 10;

export default function HousesMini({ data }: { data: HousesPreview }) {
  if (data.listings.length === 0) throw new Error("HousesMini: empty listings");
  const boxRef = useRef<HTMLDivElement | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; i: number } | null>(null);

  const { sx, sy } = useMemo(() => {
    const pts = [...data.listings, ...data.anchors];
    const lats = pts.map((p) => p.lat);
    const lons = pts.map((p) => p.lon);
    const latMin = Math.min(...lats);
    const latMax = Math.max(...lats);
    const lonMin = Math.min(...lons);
    const lonMax = Math.max(...lons);
    return {
      sx: (lon: number) => PAD + ((lon - lonMin) / (lonMax - lonMin || 1)) * (W - 2 * PAD),
      // north up: larger latitude -> smaller y
      sy: (lat: number) => H - PAD - ((lat - latMin) / (latMax - latMin || 1)) * (H - 2 * PAD),
    };
  }, [data]);

  // Anchor the tooltip to the dot's rendered rect (immune to viewBox letterboxing).
  function showTip(i: number, e: React.PointerEvent<SVGCircleElement>) {
    const box = boxRef.current;
    if (!box) return;
    const b = box.getBoundingClientRect();
    const c = e.currentTarget.getBoundingClientRect();
    setTip({ x: c.x + c.width / 2 - b.x, y: c.y - b.y, i });
  }

  const tipListing = tip != null ? data.listings[tip.i] : null;
  return (
    <div className={styles.svgWrap}>
      <div className={styles.svgBox} ref={boxRef}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Dot map of shortlisted Bay Area rentals">
          {data.anchors.map((a) => (
            <text key={a.label} x={sx(a.lon)} y={sy(a.lat)} className={styles.anchorMark}>▲</text>
          ))}
          {data.listings.map((l, i) => (
            <circle
              key={i}
              cx={sx(l.lon)}
              cy={sy(l.lat)}
              r={1.8 + (l.fit / 10) * 3.2}
              fill={DRIVER_COLORS[l.driver] ?? DRIVER_COLORS.commute}
              fillOpacity={tip?.i === i ? 1 : 0.72}
              className={styles.dot}
              onPointerEnter={(e) => showTip(i, e)}
              onPointerLeave={() => setTip(null)}
            />
          ))}
        </svg>
        {tipListing && tip && (
          <div className={styles.tip} style={{ left: tip.x, top: tip.y }}>
            <div className={styles.tipName}>{tipListing.hood}</div>
            <div className={styles.tipSub}>{tipListing.pdisp} · fit {tipListing.fit}</div>
          </div>
        )}
      </div>
      <p className={styles.caption}>the rental shortlist, mapped — hover a dot</p>
    </div>
  );
}
