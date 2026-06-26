# Panel contract — /coauthorship/analyses/

One panel = one method = one agent = exactly THREE files. You may not create, edit, or
delete anything else. No git commands. No `pip install`. No writes into `_derived/`.

```
experiments/coauthorship/analyses/<slug>.py    # compute (PEP-723 header, run with `uv run <slug>.py`)
assets/data/analyses/<slug>.json               # output of running your .py (committed)
assets/js/analyses/<slug>.js                   # render module (plain IIFE — NO ES modules)
```

Paths are relative to the repo root (your cwd is the worktree root). The shell page, the
toolbar, all CSS, and the loader already exist — your JS registers into them.

## Audience & voice (IMPORTANT)

The readers are the researchers IN this network — ML interpretability people (Bau lab,
EleutherAI) and computational researchers. The page is partly an **explainer of graph
statistics for ML people**. Your `prose.how` section teaches your method in 3-6 plain
sentences, anchored to ML concepts they already know (embeddings, alignment across
checkpoints/seeds, ablations, train/test, logits, attention maps). One good analogy beats
three equations. Example register: "OMNI embeds every year's graph into ONE shared space —
the same trick as aligning latent spaces across model checkpoints, so a person's movement
between years is meaningful, not an artifact of each year being embedded separately."
Jargon terms are fine ONLY if introduced by the analogy. The full technical name + citation
goes in `prose.method` (the collapsible "For the curious" footnote), nowhere else.

## Python compute contract

Template (copy this header exactly; add per-method deps like `hyppo` or `POT` if needed):

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""<slug> — one line. Run: cd experiments/coauthorship/analyses && uv run <slug>.py"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"     # papers.json / yearly.json / layers.json / tfidf.json (read-only)
OUT = REPO / "assets/data/analyses" / "<slug>.json"
```

Envelope (all four keys REQUIRED):

```python
payload = {
  "slug": "<slug>",                  # must equal your filename stem
  "title": "<toolbar title>",
  "headline": "…one finding sentence with the key number in <strong>…</strong>…",
  "data": { ... },                   # free-form, everything your render() needs
}
```

Hard rules:
- `headline` is a SENTENCE stating the finding (Tufte: the title states the conclusion).
  Plain HTML, `<strong>` around key numbers. No stat cards.
- ALL randomness seeded (`numpy.random.default_rng(0)`, `random_state=0`). Re-running your
  script must produce byte-identical output (verify will diff it).
- Every node id you ship inside `data` must exist in `GRAPH["nodes"]` — assert it.
- `blob = json.dumps(payload, separators=(",", ":"))`; `assert len(blob) < 300_000`
  (target < 100 KB). Write minified.
- End with `print(f"[<slug>] OK {len(blob)/1024:.0f}KB — " + <headline stripped of tags>)`.
- Liberal asserts on intermediate shapes; fail fast and loud.

## Shared derived data (read-only, in `_derived/`)

- `papers.json`: `[{key, title, year, citations, sources:["s2"|"openalex",…], members:[node ids
  present in the shipped graph], n_authors_total, big:bool}]` — every distinct paper by a
  listed person, cross-source deduped, name-keyed IDENTICALLY to the shipped graph.
  `big` = paper has > 25 authors (the shipped graph only draws list↔list edges from those).
- `yearly.json`: `{vertex_order:[node ids], years:[…], per_year:{yr:[[i,j,w],…]},
  cumulative:{yr:[[i,j,w],…]}}` — i,j index vertex_order; w = #distinct papers (version-deduped);
  same edge rules as the shipped graph (big-paper rule, EDGE_DROP, junk filters).
- `layers.json`: `{vertex_order:[…], s2:[[i,j,w],…], oa:[[i,j,w],…]}` — single-source slices.
- `tfidf.json`: `{ids:[…], vocab:[…], rows:[[ [term_idx, weight],…],…]}` — title TF-IDF per
  list member with ≥1 paper.

If you need something not here, compute it privately inside your own script. Do NOT write
to `_derived/`.

## JS render contract

Your file is a plain IIFE (script-tag loaded, any order). EXACT registration pattern:

```js
/* assets/js/analyses/<slug>.js */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("<slug>", {
    prose: {
      intro:  "<p>2-3 sentences: the question this panel asks, in plain language.</p>",
      how:    "<p>3-6 sentences: HOW the method works, for an ML-literate reader, with one interp/ML analogy. May use 2-3 <p> tags.</p>",
      method: "<p>One paragraph: formal method name, key reference (authors, venue, year), caveats. Jargon lives here only.</p>"
    },
    render: function (el, data, shared) {
      // d3 (v7) is global — reference it ONLY inside render, never at IIFE top level.
      // el = your panel's .m-viz div, already sized and emptied. Redraw from scratch every call
      // (the shell clears el and re-invokes on resize). Read el.clientWidth, never window.
      // data = your <slug>.json envelope (the whole object; your payload is data.data).
    }
  });
})();
```

`shared` API (built by the shell — use it, don't rebuild it):
- `shared.colors` — `{community:["#4c6b8a","#a6611a","#5a7d5a"], other:"#b3a98f", bg:"#faf8f3",
  ink:"#2b2b2b", muted:"#8c867b", hair:"#e3ddcf", src:{both:"#5a7d5a", s2:"#a6611a", oa:"#4c6b8a"}}`.
  Community index = community id (0=EleutherAI, 1=David Bau, 2=Joshua Vogelstein).
  Magnitude = ink-opacity ramps, NOT new hues.
- `shared.labelOf(id)`, `shared.communityOf(id)`, `shared.isList(id)`, `shared.colorOf(id)`,
  `shared.nodes` (Map id → {label, community, is_list, degree}), `shared.communities`.
- `shared.minimap(el, colorFn, opts)` — draws the standard 200×160 network mini-map (the page's
  visual anchor; same layout every panel) colored by your `colorFn(nodeId) → css color | null`
  (null = periphery grey). opts: `{width, height, radiusFn(id), opacityFn(id), edgeOpacity}`.
  Returns `{svg, highlight(id|null)}`. USE THIS instead of drawing your own mini network.
- `shared.tooltip.show(html, evt)` / `shared.tooltip.hide()` — the one shared tooltip card.
- `shared.fmt.num/pct/sig`, `shared.esc(s)` (escape ALL data-derived strings you put in HTML).

Rendering rules:
- No `<style>` tags, no `document.createElement("style")`, no font-family, no page-level CSS.
  Inline attribute styles on your own SVG/divs are fine. Use the pinned classes for text:
  `m-axis` (11px ticks), `m-label` (12px direct labels w/ halo), `m-note` (muted small).
- Direct labels over legends. Hairline axes (`shared.colors.hair`) or none. No gridlines
  heavier than #efe9dc. No pies, no 3D, no gradients-for-decoration.
- D3 GOTCHA (learned on the sister page): a pending `.transition().style(...)` silently
  overwrites later direct `.style(...)` sets — call `selection.interrupt()` before applying
  a sustained highlight/dim.
- Interactive hover: use `shared.tooltip` + your minimap's `highlight(id)` so hovering a
  chart row highlights the same person on the mini-map.

## Definition of done (ALL must pass before you finish)

1. `cd experiments/coauthorship/analyses && uv run <slug>.py` exits 0, prints the OK line.
2. Re-run it: output byte-identical (`git diff --stat` shows your json unchanged on 2nd run).
3. `node --check assets/js/analyses/<slug>.js` passes.
4. Self-review against this contract (registration shape, prose keys incl. `how`, no
   forbidden patterns: `grep -nE "<style|createElement\(.style.\)|font-family" your.js` is clean,
   and `d3` appears only inside `render`).
5. Your final message: the headline sentence + any honest data caveats (verify agents will
   adversarially re-check; do not oversell).
