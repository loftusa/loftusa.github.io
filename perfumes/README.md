# The Perfume Atlas — `/perfumes/`

An interactive map of **24,050 fragrances** arranged so that perfumes which *smell alike sit close
together*. Colour = scent family, point size = how loved (review count), and the nodes bloom into the
actual bottle photographs as you zoom in. Click any perfume to trace its weighted **scent-twins**.
Companion field guide at `/perfumes/analyses/`.

Live: https://alex-loftus.com/perfumes/

## How similarity is defined

Each perfume becomes a sparse vector over its **notes** (1,671 of them) and **accords** (84):

- a note is weighted by **pyramid position** — base ×3, middle ×2, top ×1 (the base lasts longest and
  defines the drydown) — and by **IDF**, `log(N / docs_with_note)`, so a shared *oud* counts far more
  than a shared *musk*.
- accords get a positional weight (Fragrantica lists them strongest-first) × IDF.
- the two unit-normalised blocks are concatenated as `√0.7·notes ⊕ √0.3·accords`, so the cosine of the
  concatenation is exactly `0.7·cos(notes) + 0.3·cos(accords)`. One clean metric.

Validation it's real: Sauvage's nearest neighbours come out as its known budget clones (Armaf Club de
Nuit Urban Elixir, etc.); Angel lands in *Caramel/Chocolate*, Tobacco Vanille in *Tobacco/Whiskey*.

## Pipeline

```
fra_cleaned.csv  ──>  build_atlas.py  ──>  assets/data/perfumes-atlas.json     (positions, meta, 14 families)
(Fragrantica)                              assets/data/perfumes-neighbors.json  (top-8 scent-twins + weights)
                                           assets/data/perfume-analyses.json    (family stats, dupes, loners, eras)
                                                  │
                                                  ├─ vectors (IDF + pyramid notes ⊕ accords)
                                                  ├─ UMAP(cosine)            -> 2-D position
                                                  ├─ kNN(cosine)             -> weighted edges
                                                  └─ accord vote             -> 14 scent families
```

Data source: the public Fragrantica `fra_cleaned.csv` (≈24k perfumes with notes split by pyramid level,
5 accords, rating + review count). Fragrantica itself is Cloudflare-blocked, so live scraping is not used;
the bottle thumbnails are hot-linked from its CDN (`fimgs.net`, derived from the perfume id in the url).

### Rebuild

```bash
# the source CSV (≈6.7 MB, no auth) — a public mirror of the Fragrantica dataset:
curl -sL https://raw.githubusercontent.com/abdullah-makhokhar/fragrance-generator/main/data/fra_cleaned.csv -o /tmp/fra_cleaned.csv

uv run --with umap-learn,polars,scikit-learn,scipy,numpy,click \
  python perfumes/build_atlas.py --csv /tmp/fra_cleaned.csv --out assets/data
```

Deterministic (UMAP `random_state=42`); ~50s. The 14 scent families and their colours are the curated
`MACRO` table at the top of `build_atlas.py` — every Fragrantica accord maps to exactly one (asserted).

## Front-end

| file | role |
|------|------|
| `_pages/perfumes.html`           | the map page (permalink `/perfumes/`, `layout: fullscreen`) |
| `_pages/perfumes-analyses.html`  | the field guide (permalink `/perfumes/analyses/`, `layout: bare`) |
| `assets/js/perfumes-atlas.js`    | d3-zoom + canvas renderer; lazy bottle images, hover card, twin-edges, search |
| `assets/css/perfumes-atlas.css`  | the map's elegant/feminine Tufte styling |
| `assets/css/perfumes-analyses.css` | the field-guide styling |

The renderer draws 24k points on a single `<canvas>` (culled + d3-quadtree hit-testing); a node shows its
photograph once its on-screen radius passes ~11 px, capped to the most-reviewed ~420 per frame so panning
stays smooth. Images are hot-linked without `crossOrigin` (the CDN sends no CORS header) — fine for
display, we never read pixels back. `window.PerfumeAtlas` exposes `findByName / flyToNode / select`.

Rebuilding the data requires a `bundle exec jekyll build` to copy it into `_site/`.
