// Minimal deterministic spring relaxation for the front-page networks mini.
// The graph behaves like an elastic sheet anchored at home positions (x0, y0):
// each link tries to preserve its original offset vector, each unpinned node
// is pulled back toward home. No RNG, no repulsion (homes prevent collapse).
export function relax(nodes, links, { iterations = 1, spring = 0.06, home = 0.1 } = {}) {
  const n = nodes.length;
  const fx = new Float64Array(n);
  const fy = new Float64Array(n);
  for (let it = 0; it < iterations; it++) {
    fx.fill(0);
    fy.fill(0);
    for (const [s, t] of links) {
      const a = nodes[s];
      const b = nodes[t];
      // deviation of the current offset from the original offset
      const dx = (b.x - a.x) - (b.x0 - a.x0);
      const dy = (b.y - a.y) - (b.y0 - a.y0);
      fx[s] += spring * dx; fy[s] += spring * dy;
      fx[t] -= spring * dx; fy[t] -= spring * dy;
    }
    for (let i = 0; i < n; i++) {
      const node = nodes[i];
      if (node.pinned) continue;
      node.x += fx[i] + home * (node.x0 - node.x);
      node.y += fy[i] + home * (node.y0 - node.y);
    }
  }
}

export function settled(nodes, eps = 0.05) {
  return nodes.every(
    (n) => n.pinned || (Math.abs(n.x - n.x0) < eps && Math.abs(n.y - n.y0) < eps),
  );
}
