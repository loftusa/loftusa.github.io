import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

const FitMath = require("../../public/houses/fitmath.js");
const ctx = { window: {} };
vm.createContext(ctx);
vm.runInContext(readFileSync(new URL("../../public/houses/data.js", import.meta.url), "utf8"), ctx);
const D = ctx.window.HOUSES_DATA;
assert.ok(D.listings.length > 0, "data.js loaded");

const W = FitMath.normWeights(D.meta.fit_weights.alex);

// 1) Parity: default weights reproduce shipped fits (rows rated under current weights)
let checked = 0;
for (const x of D.listings) {
  if (!("gym" in x.scores)) { assert.ok(x.pinned, `${x.id} missing gym but not pinned`); continue; }
  assert.equal(FitMath.fitOf(x, W), x.fit, `parity ${x.id}`);
  checked++;
}
assert.ok(checked >= D.listings.length - 6, "parity covered nearly all listings");

// 2) normWeights: all-zero -> equal; single nonzero -> 1; sums to 1
const eq = FitMath.normWeights(Object.fromEntries(FitMath.COMP_KEYS.map(k => [k, 0])));
for (const k of FitMath.COMP_KEYS) assert.ok(Math.abs(eq[k] - 1 / 7) < 1e-9);
const one = FitMath.normWeights({ commute: 50, nice: 0, aesthetic: 0, nature: 0, value: 0, soft: 0, gym: 0 });
assert.equal(one.commute, 1);
const nw = FitMath.normWeights({ commute: 30, nice: 10, aesthetic: 10, nature: 10, value: 10, soft: 10, gym: 20 });
assert.ok(Math.abs(Object.values(nw).reduce((a, b) => a + b, 0) - 1) < 1e-9);

// 3) soft mapping + missing-score neutrality
const apt = { bucket: "apt", scores: { quiet: 9, social: 1 } };
const room = { bucket: "room", scores: { quiet: 9, social: 1 } };
assert.equal(FitMath.compScores(apt).soft, 9);
assert.equal(FitMath.compScores(room).soft, 1);
assert.equal(FitMath.compScores(apt).gym, 5);
assert.equal(FitMath.compScores(apt).nice, 5);

// 4) loved bonus + cap at 10
const hot = { bucket: "apt", loved: true, dual_commute: 10,
  scores: { quiet: 10, social: 0, nice: 10, nature: 10, value: 10, commute: 10, aesthetic: 10, gym: 10 } };
assert.equal(FitMath.fitOf(hot, W), 10, "capped at 10");
const mid = { bucket: "apt", loved: true, dual_commute: 5,
  scores: { quiet: 5, social: 0, nice: 5, nature: 5, value: 5, commute: 5, aesthetic: 5, gym: 5 } };
assert.equal(FitMath.fitOf(mid, W), 5.8, "loved bonus +0.8 applied");

// 5) rankBoard: cap filter, topN cutoff, "all", sorted desc
const cands = D.listings.filter(x => !x.gone);
const rb = FitMath.rankBoard(cands, W, null, 5);
assert.equal(rb.boardIds.size, Math.min(5, cands.length));
assert.equal(rb.ranked.length, cands.length);
const fits = rb.ranked.map(x => rb.fits.get(x.id));
for (let i = 1; i < fits.length; i++) assert.ok(fits[i - 1] >= fits[i], "sorted desc");
const cheap = FitMath.rankBoard(cands, W, 1500, "all");
assert.ok(cheap.ranked.length > 0, "some listings under $1500");
assert.ok([...cheap.boardIds].every(id => cands.find(x => x.id === id).price <= 1500));
const none = FitMath.rankBoard(cands, W, 1, "all");
assert.equal(none.boardIds.size, 0, "cap below min price -> empty board");
const all = FitMath.rankBoard(cands, W, null, "all");
assert.equal(all.boardIds.size, cands.length);

// 6) driverOf: at-the-mean listing is neutral; forced winner detected
const means = FitMath.fieldMeans(cands);
const flat = { bucket: "apt", dual_commute: means.commute, scores: Object.fromEntries(
  FitMath.COMP_KEYS.map(k => [k === "soft" ? "quiet" : k, means[k]])) };
assert.equal(FitMath.driverOf(flat, W, means), null, "at-the-mean listing is neutral");
const gymmy = { bucket: "apt", dual_commute: means.commute, scores: Object.fromEntries(
  FitMath.COMP_KEYS.map(k => [k === "soft" ? "quiet" : k, k === "gym" ? 10 : means[k]])) };
assert.equal(FitMath.driverOf(gymmy, W, means), "gym", "gym outlier wins");

console.log(`parity.test.mjs OK — ${checked} parity rows, ${cands.length} candidates`);
