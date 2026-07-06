import { test } from "node:test";
import assert from "node:assert/strict";
import { relax, settled } from "./force-sim.mjs";

const mk = (x, y, over = {}) => ({ x, y, x0: x, y0: y, ...over });

test("a graph at rest stays at rest", () => {
  const nodes = [mk(0, 0), mk(10, 0)];
  relax(nodes, [[0, 1]], { iterations: 5 });
  assert.equal(nodes[0].x, 0);
  assert.equal(nodes[1].x, 10);
  assert.ok(settled(nodes));
});

test("pinned nodes never move", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 30, y: 40 }), mk(10, 0)];
  relax(nodes, [[0, 1]], { iterations: 3 });
  assert.equal(nodes[0].x, 30);
  assert.equal(nodes[0].y, 40);
});

test("neighbors follow a displaced pinned node", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 20 }), mk(10, 0)];
  relax(nodes, [[0, 1]], { iterations: 10 });
  assert.ok(nodes[1].x > 10, `expected neighbor pulled right, got ${nodes[1].x}`);
});

test("after unpinning, everything relaxes back home", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 20, y: 15 }), mk(10, 0)];
  relax(nodes, [[0, 1]], { iterations: 10 });
  nodes[0].pinned = false;
  relax(nodes, [[0, 1]], { iterations: 400 });
  assert.ok(Math.abs(nodes[0].x - 0) < 0.01, `node0.x=${nodes[0].x}`);
  assert.ok(Math.abs(nodes[1].x - 10) < 0.01, `node1.x=${nodes[1].x}`);
  assert.ok(settled(nodes, 0.02));
});

test("positions stay finite under a large displacement", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 1e4 }), mk(1, 0), mk(2, 0)];
  relax(nodes, [[0, 1], [1, 2]], { iterations: 200 });
  for (const n of nodes) {
    assert.ok(Number.isFinite(n.x) && Number.isFinite(n.y));
  }
});

test("settled respects eps and ignores pinned nodes", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 100 }), mk(10, 0)];
  assert.ok(settled(nodes, 0.01), "pinned displacement should not count");
  nodes[1].x = 10.5;
  assert.ok(!settled(nodes, 0.01));
  assert.ok(settled(nodes, 1));
});
