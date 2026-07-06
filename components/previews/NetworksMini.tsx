"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { relax, settled, type SimNode } from "@/lib/force-sim.mjs";
import type { NetworksPreview } from "./types";
import styles from "./ProjectPreviews.module.css";

// Community -> site categorical tokens (0 EleutherAI, 1 David Bau, 2 Vogelstein)
const COMMUNITY_COLORS = ["var(--cat-blue)", "var(--cat-sienna)", "var(--cat-sage)"];

const W = 200;
const H = 132;
const PAD = 8;

function scaleNodes(data: NetworksPreview): SimNode[] {
  const xs = data.nodes.map((n) => n.x);
  const ys = data.nodes.map((n) => n.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  return data.nodes.map((n) => {
    const x = PAD + ((n.x - xMin) / (xMax - xMin || 1)) * (W - 2 * PAD);
    const y = PAD + ((n.y - yMin) / (yMax - yMin || 1)) * (H - 2 * PAD);
    return { x, y, x0: x, y0: y };
  });
}

export default function NetworksMini({ data }: { data: NetworksPreview }) {
  if (data.nodes.length === 0) throw new Error("NetworksMini: empty nodes");
  const nodesRef = useRef<SimNode[]>([]);
  const [, setTick] = useState(0);
  const [tip, setTip] = useState<{ x: number; y: number; i: number } | null>(null);
  const dragRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const reducedRef = useRef(false);

  // Initialize once (and if data identity ever changes)
  useMemo(() => {
    nodesRef.current = scaleNodes(data);
  }, [data]);

  useEffect(() => {
    reducedRef.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  function loop() {
    relax(nodesRef.current, data.links, { iterations: 2 });
    setTick((t) => t + 1);
    const done = dragRef.current == null && settled(nodesRef.current, 0.15);
    rafRef.current = done ? null : requestAnimationFrame(loop);
    if (done) {
      // snap exactly home so idle state is byte-identical to first render
      for (const n of nodesRef.current) { n.x = n.x0; n.y = n.y0; }
      setTick((t) => t + 1);
    }
  }

  function toLocal(e: React.PointerEvent): { x: number; y: number } {
    const svg = svgRef.current!;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: W / 2, y: H / 2 };
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return { x: p.x, y: p.y };
  }

  // Anchor the tooltip to the node's rendered rect (immune to viewBox letterboxing).
  function showTip(i: number, e: React.PointerEvent<SVGCircleElement>) {
    if (dragRef.current != null) return; // no tooltip while dragging
    const box = boxRef.current;
    if (!box) return;
    const b = box.getBoundingClientRect();
    const c = e.currentTarget.getBoundingClientRect();
    setTip({ x: c.x + c.width / 2 - b.x, y: c.y - b.y, i });
  }

  function onNodeDown(i: number, e: React.PointerEvent) {
    if (reducedRef.current) return;
    e.preventDefault();
    setTip(null);
    dragRef.current = i;
    nodesRef.current[i].pinned = true;
    (e.target as Element).setPointerCapture(e.pointerId);
    if (rafRef.current == null) rafRef.current = requestAnimationFrame(loop);
  }
  function onNodeMove(e: React.PointerEvent) {
    const i = dragRef.current;
    if (i == null) return;
    const p = toLocal(e);
    nodesRef.current[i].x = Math.max(2, Math.min(W - 2, p.x));
    nodesRef.current[i].y = Math.max(2, Math.min(H - 2, p.y));
  }
  function onNodeUp() {
    const i = dragRef.current;
    if (i == null) return;
    nodesRef.current[i].pinned = false;
    dragRef.current = null;
  }

  const nodes = nodesRef.current;
  const tipNode = tip != null ? data.nodes[tip.i] : null;
  const tipGroup = tipNode ? data.communities.find((c) => c.id === tipNode.community)?.label : null;

  return (
    <div className={styles.svgWrap}>
      <div className={styles.svgBox} ref={boxRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label="Co-authorship network of researchers Alex has worked with"
        >
          {data.links.map(([s, t], i) => (
            <line
              key={i}
              x1={nodes[s].x} y1={nodes[s].y}
              x2={nodes[t].x} y2={nodes[t].y}
              className={styles.netLink}
            />
          ))}
          {nodes.map((n, i) => (
            <circle
              key={i}
              cx={n.x} cy={n.y} r={tip?.i === i ? 3.6 : 2.4}
              fill={COMMUNITY_COLORS[data.nodes[i].community] ?? "var(--muted)"}
              fillOpacity={0.85}
              className={dragRef.current === i ? `${styles.netNode} ${styles.netNodeDragging}` : styles.netNode}
              onPointerDown={(e) => onNodeDown(i, e)}
              onPointerMove={onNodeMove}
              onPointerUp={onNodeUp}
              onPointerCancel={onNodeUp}
              onPointerEnter={(e) => showTip(i, e)}
              onPointerLeave={() => setTip(null)}
            />
          ))}
        </svg>
        {tip && tipNode && (
          <div className={styles.tip} style={{ left: tip.x, top: tip.y }}>
            <div className={styles.tipName}>{tipNode.label}</div>
            {tipGroup && <div className={styles.tipSub}>{tipGroup}</div>}
          </div>
        )}
      </div>
      <p className={styles.caption}>co-authors, linked by papers — drag a node</p>
    </div>
  );
}
