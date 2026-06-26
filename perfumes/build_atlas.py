"""Build the perfume similarity atlas for /perfumes/.

Pipeline (all ~24k Fragrantica perfumes):
  1. clean + parse notes (top/middle/base pyramid) and the 5 main accords
  2. build a single cosine feature space that encodes
         sim = 0.7 * cos(note vectors) + 0.3 * cos(accord vectors)
     by concatenating two unit-normalised, sqrt-weighted sub-vectors. Notes are
     weighted by pyramid level (base lasts longest) x IDF (rare shared notes count more).
  3. UMAP -> 2D position (smell-alikes land near each other)
  4. kNN -> each perfume's top-k scent twins (the weighted graph edges)
  5. assign each perfume to one of 14 curated scent families by an accord vote
     (accord-less perfumes inherit the majority family of their kNN twins)
  6. emit compact columnar JSON for the canvas front-end

Run:  uv run --with umap-learn,polars,scikit-learn,scipy,numpy,click \
          python perfumes/build_atlas.py --csv <fra_cleaned.csv> --out assets/data
"""
from __future__ import annotations
import json, math, re, time
from collections import Counter
from pathlib import Path

import click
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

LEVEL_WT = {
    "Top": 1.0,
    "Middle": 2.0,
    "Base": 3.0,
}  # base notes define the drydown / "soul"
ALPHA_NOTES = 0.7  # vs 0.3 for the coarse accords
K_EDGES = 8  # scent twins stored per perfume
PID_RE = re.compile(r"-(\d+)\.html?$")
# single source of truth for the two URL templates the front-ends consume (via meta / analyses json)
IMG_TMPL = "https://fimgs.net/mdimg/perfume-thumbs/375x500.{pid}.jpg"
FRAG_TMPL = "https://www.fragrantica.com/perfume/-{pid}.html"

# Fourteen curated scent families — the fragrance world's actual categories, ordered as a
# gentle wheel (fresh -> floral -> sweet -> warm -> dark -> green). Each colour *means*
# something (sage=green, aqua=aquatic, gold=amber, browns=oud/leather), Tufte-style, and the
# whole set is muted/feminine so it harmonises on cream. Every Fragrantica accord maps to one.
MACRO = [
    (
        "White Florals",
        "#dca7b4",
        ["white floral", "floral", "tuberose", "yellow floral"],
    ),
    ("Rose", "#c4647a", ["rose"]),
    (
        "Powder & Musk",
        "#b9a6d3",
        ["powdery", "iris", "violet", "aldehydic", "musky", "soapy"],
    ),
    ("Citrus & Cologne", "#e6c24f", ["citrus", "fresh"]),
    (
        "Green & Aromatic",
        "#8caa72",
        [
            "green",
            "herbal",
            "aromatic",
            "lavender",
            "conifer",
            "camphor",
            "terpenic",
            "savory",
            "cannabis",
        ],
    ),
    (
        "Aquatic & Marine",
        "#6fb0c2",
        ["aquatic", "marine", "ozonic", "salty", "sand", "mineral"],
    ),
    (
        "Fruity & Boozy",
        "#e0895f",
        [
            "fruity",
            "tropical",
            "coconut",
            "cherry",
            "champagne",
            "wine",
            "rum",
            "vodka",
            "alcohol",
            "sour",
            "coca-cola",
        ],
    ),
    (
        "Sweet Gourmand",
        "#cf9466",
        [
            "sweet",
            "vanilla",
            "caramel",
            "chocolate",
            "cacao",
            "honey",
            "almond",
            "nutty",
            "lactonic",
            "coffee",
            "creamy",
            "gourmand",
            "beeswax",
        ],
    ),
    (
        "Spice",
        "#c15f43",
        [
            "warm spicy",
            "fresh spicy",
            "soft spicy",
            "cinnamon",
            "anis",
            "spicy",
            "bitter",
        ],
    ),
    (
        "Woods & Moss",
        "#95934f",
        ["woody", "patchouli", "earthy", "mossy", "metallic", "clay", "paper", "oily"],
    ),
    ("Amber & Resin", "#e0992f", ["amber", "balsamic", "oriental"]),
    ("Oud", "#6b4c39", ["oud"]),
    (
        "Leather & Smoke",
        "#776c66",
        ["leather", "smoky", "tobacco", "animalic", "whiskey"],
    ),
    (
        "Unconventional",
        "#8f9295",
        [
            "gasoline",
            "asphault",
            "vinyl",
            "plastic",
            "rubber",
            "industrial glue",
            "brown scotch tape",
            "hot iron",
        ],
    ),
]
ACCORD2FAM = {a: fid for fid, (_n, _c, accs) in enumerate(MACRO) for a in accs}


def prettify(slug: str) -> str:
    """jean-paul-gaultier -> Jean Paul Gaultier ; keep small words lower where natural."""
    if not slug:
        return ""
    words = re.split(r"[-\s]+", slug.strip())
    small = {
        "de",
        "la",
        "le",
        "du",
        "des",
        "et",
        "of",
        "the",
        "a",
        "by",
        "for",
        "von",
        "und",
    }
    out = []
    for i, w in enumerate(words):
        if not w:
            continue
        if i and w.lower() in small:
            out.append(w.lower())
        elif w.isupper() or any(c.isdigit() for c in w):
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def norm_gender(g: str) -> int:
    g = (g or "").lower()
    if "unisex" in g:
        return 2
    if "women" in g or "female" in g:
        return 0
    if "men" in g or "male" in g:
        return 1
    return 3


GENDER_LABELS = ["women", "men", "unisex", "—"]


def parse_list(s: str) -> list[str]:
    if s is None:
        return []
    # � is the utf8-lossy replacement char for bytes lost on decode (e.g. ® in "physcool®")
    out = []
    for x in str(s).split(","):
        x = x.replace("�", "").strip().lower()
        if x and x != "unknown":
            out.append(x)
    return out


@click.command()
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", default="assets/data", type=click.Path())
@click.option(
    "--min-notes",
    default=3,
    help="drop perfumes with fewer total notes (too sparse to place)",
)
def main(csv_path: str, out_dir: str, min_notes: int):
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. load + clean ----------------------------------------------------
    df = pl.read_csv(
        csv_path, separator=";", infer_schema_length=8000, encoding="utf8-lossy"
    )
    df = df.with_columns(
        [
            pl.col("Rating Count")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("rc"),
            pl.col("Rating Value")
            .str.replace(",", ".")
            .cast(pl.Float64, strict=False)
            .alias("rv"),
            pl.col("Year").cast(pl.Int64, strict=False).alias("yr"),
        ]
    )
    pids, names, brands, years, genders, ratings, rcs = [], [], [], [], [], [], []
    note_bags, accord_lists, note_disp = [], [], []
    n_bad_pid = 0
    for row in df.iter_rows(named=True):
        m = PID_RE.search(row["url"] or "")
        if not m:
            n_bad_pid += 1
            continue
        bag = Counter()
        ordered_notes = []
        for lvl in ("Top", "Middle", "Base"):
            for note in parse_list(row[lvl]):
                bag[note] += LEVEL_WT[lvl]
                if note not in ordered_notes:
                    ordered_notes.append(note)
        if sum(1 for _ in bag) < min_notes:
            continue
        accords = [row[f"mainaccord{i}"] for i in range(1, 6)]
        accords = [
            a.strip().lower()
            for a in accords
            if a and str(a).strip() and str(a).strip() != "unknown"
        ]
        pids.append(int(m.group(1)))
        names.append(prettify(row["Perfume"]))
        brands.append(prettify(row["Brand"]))
        years.append(
            int(row["yr"]) if row["yr"] and 1900 < (row["yr"] or 0) < 2100 else None
        )
        genders.append(norm_gender(row["Gender"]))
        ratings.append(round(row["rv"], 2) if row["rv"] is not None else None)
        rcs.append(int(row["rc"]))
        note_bags.append(bag)
        accord_lists.append(accords)
        note_disp.append(", ".join(ordered_notes[:8]))
    n = len(pids)
    assert n > 10000, f"expected ~24k perfumes, got {n}"
    print(
        f"[clean] kept {n} perfumes  (dropped {n_bad_pid} with no pid, "
        f"{df.height - n - n_bad_pid} too sparse)  in {time.time()-t0:.1f}s"
    )

    # ---- 2. build the combined cosine feature space -------------------------
    # note vocabulary + IDF
    note_df = Counter()
    for bag in note_bags:
        for note in bag:
            note_df[note] += 1
    note_vocab = {note: i for i, note in enumerate(note_df)}
    note_idf = {note: math.log(n / dfc) for note, dfc in note_df.items()}
    data, ri, ci = [], [], []
    for i, bag in enumerate(note_bags):
        for note, w in bag.items():
            ri.append(i)
            ci.append(note_vocab[note])
            data.append(w * note_idf[note])
    Xn = normalize(
        csr_matrix((data, (ri, ci)), shape=(n, len(note_vocab)), dtype=np.float32)
    )

    # accord vocabulary + IDF; accords are listed strongest-first -> positional weight
    acc_df = Counter()
    for accs in accord_lists:
        for a in accs:
            acc_df[a] += 1
    acc_vocab = {a: i for i, a in enumerate(acc_df)}
    acc_idf = {a: math.log(n / dfc) for a, dfc in acc_df.items()}
    data, ri, ci = [], [], []
    for i, accs in enumerate(accord_lists):
        for rank, a in enumerate(accs):
            ri.append(i)
            ci.append(acc_vocab[a])
            data.append((5 - rank) * acc_idf[a])
    Xa = normalize(
        csr_matrix((data, (ri, ci)), shape=(n, len(acc_vocab)), dtype=np.float32)
    )

    # cos(concat) == ALPHA*cos(notes) + (1-ALPHA)*cos(accords) when both blocks are unit norm.
    # A perfume can have notes but zero accords (Xa row all-zero) -> combined norm = sqrt(ALPHA),
    # so re-normalise the stacked matrix to guarantee unit rows (and a meaningful cosine).
    X = normalize(
        hstack([Xn * math.sqrt(ALPHA_NOTES), Xa * math.sqrt(1 - ALPHA_NOTES)]).tocsr()
    )
    rownorm = np.sqrt(X.multiply(X).sum(axis=1)).A1
    assert np.allclose(
        rownorm[rownorm > 0], 1.0, atol=1e-4
    ), "combined rows must be unit norm"
    print(
        f"[vectors] notes={len(note_vocab)} accords={len(acc_vocab)} "
        f"nnz={X.nnz}  in {time.time()-t0:.1f}s"
    )

    # ---- 3. UMAP ------------------------------------------------------------
    import umap

    tu = time.time()
    emb = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    ).fit_transform(X)
    emb = np.asarray(emb, dtype=np.float32)
    # centre + scale to a tidy [-100,100]-ish box (front-end re-fits to viewport anyway)
    emb -= emb.mean(axis=0)
    emb /= (np.abs(emb).max() + 1e-9) / 100.0
    print(f"[umap] {emb.shape} in {time.time()-tu:.1f}s")

    # ---- 4. kNN scent twins (graph edges) -----------------------------------
    tk = time.time()
    nn = NearestNeighbors(n_neighbors=K_EDGES + 4, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    # filter self per-row: with exact-duplicate vectors (an EDT/EDP listed identically) the self
    # point can land off column 0, so a plain idx[:,1:] would keep it. Build clean top-k twins.
    nbr = np.zeros((n, K_EDGES), dtype=np.int32)
    wt = np.zeros((n, K_EDGES), dtype=np.float32)
    for i in range(n):
        js, ds = [], []
        for j, d in zip(idx[i], dist[i]):
            if int(j) == i:
                continue
            js.append(int(j))
            ds.append(float(d))
            if len(js) == K_EDGES:
                break
        while len(js) < K_EDGES:  # pad tiny isolated rows
            js.append(js[-1] if js else i)
            ds.append(ds[-1] if ds else 1.0)
        nbr[i] = js
        wt[i] = 1.0 - np.array(ds, dtype=np.float32)
    print(f"[knn] k={K_EDGES} in {time.time()-tk:.1f}s")

    # ---- 5. scent families (curated macro-families from accords) -------------
    # every accord must map to a family, else a perfume could be silently mis-coloured
    unmapped = [a for a in acc_vocab if a not in ACCORD2FAM]
    assert not unmapped, f"unmapped accords: {unmapped}"
    # assign each perfume to the family it scores highest on (accords are listed
    # strongest-first -> rank weight 5,4,3,2,1)
    fam_of = np.full(n, -1, dtype=np.int32)
    for i, accs in enumerate(accord_lists):
        score = np.zeros(len(MACRO))
        for rank, a in enumerate(accs):
            score[ACCORD2FAM[a]] += 5 - rank
        if score.max() > 0:
            fam_of[i] = int(score.argmax())
    # the handful with no accords inherit the majority family of their kNN scent twins
    missing = np.where(fam_of < 0)[0]
    for i in missing:
        votes = Counter(int(fam_of[j]) for j in nbr[i] if fam_of[j] >= 0)
        if votes:
            fam_of[i] = votes.most_common(1)[0][0]
    fam_of[fam_of < 0] = 0
    print(
        f"[families] {len(MACRO)} macro-families; {n - len(missing)} by accord, "
        f"{len(missing)} by twin-vote"
    )

    families = []
    for fid, (fname, color, _accs) in enumerate(MACRO):
        members = np.where(fam_of == fid)[0]
        if len(members) == 0:
            continue
        loc_acc, loc_note = Counter(), Counter()
        for i in members:
            loc_acc.update(accord_lists[i])
            loc_note.update(note_bags[i].keys())
        arche = int(
            members[np.argmax([rcs[i] for i in members])]
        )  # most-reviewed = family's face
        families.append(
            {
                "id": fid,
                "name": fname,
                "color": color,
                "size": int(len(members)),
                "accords": [a for a, _ in loc_acc.most_common(5)],
                "notes": [no for no, _ in loc_note.most_common(6)],
                "archetype": arche,
            }
        )
    print("[families] " + " | ".join(f"{f['name']}:{f['size']}" for f in families))

    # ---- 6. fun facts for the (non-technical) companion page -----------------
    pop = [i for i in range(n) if rcs[i] >= 500]
    # global scent twins: the most-similar pair of well-reviewed perfumes from *different houses*
    # (the fun "this smells just like that" discovery; excludes a brand's own flankers + dupes)
    best = (-1.0, None)
    for i in pop:
        for j, w in zip(nbr[i], wt[i]):
            j = int(j)
            if (
                rcs[j] >= 500
                and best[0] < w < 0.9999
                and brands[i] != brands[j]
                and names[i] != names[j]
            ):
                best = (float(w), (i, j))
    # the loner: most-reviewed perfume whose closest twin is the least similar
    loner = min(pop, key=lambda i: wt[i][0]) if pop else 0
    facts = {
        "n": n,
        "n_popular": len(pop),
        "n_families": len(families),
        "twins": {
            "sim": round(best[0], 3),
            "a": best[1][0] if best[1] else 0,
            "b": best[1][1] if best[1] else 0,
        },
        "loner": {"i": loner, "sim": round(float(wt[loner][0]), 3)},
        "biggest_family": max(families, key=lambda f: f["size"])["id"]
        if families
        else 0,
    }

    # ---- 6b. analyses (self-contained data for the companion page) ----------
    def disp(i):
        return {
            "pid": pids[i],
            "name": names[i],
            "brand": brands[i],
            "fam": int(fam_of[i]),
        }

    # enriched per-family stats: gender lean, era, rating, signature notes, a bottle gallery
    fam_stats = []
    for f in families:
        members = np.where(fam_of == f["id"])[0]
        g = Counter(genders[i] for i in members)
        yrs = [years[i] for i in members if years[i]]
        rts = [ratings[i] for i in members if ratings[i]]
        gallery = sorted(members, key=lambda i: -rcs[i])[:8]
        fam_stats.append(
            {
                "id": f["id"],
                "name": f["name"],
                "color": f["color"],
                "size": f["size"],
                "pct": round(100 * f["size"] / n, 1),
                "archetype": disp(f["archetype"]),
                "notes": f["notes"],
                "gender": [
                    round(g.get(k, 0) / len(members), 3) for k in (0, 1, 2)
                ],  # women/men/unisex
                "median_year": int(np.median(yrs)) if yrs else None,
                "avg_rating": round(float(np.mean(rts)), 2) if rts else None,
                "gallery": [disp(int(i)) for i in gallery],
            }
        )

    # surprising twins: most-similar cross-house pairs, each perfume used once (a varied gallery)
    seen_pair, used, pairs = set(), Counter(), []
    cand = []
    for i in pop:
        for j, w in zip(nbr[i], wt[i]):
            j = int(j)
            if (
                rcs[j] >= 400
                and brands[i] != brands[j]
                and names[i] != names[j]
                and w < 0.9999
            ):
                key = (min(i, j), max(i, j))
                if key not in seen_pair:
                    seen_pair.add(key)
                    cand.append((float(w), key[0], key[1]))
    cand.sort(reverse=True)
    for w, i, j in cand:
        if used[i] or used[j]:
            continue
        used[i] += 1
        used[j] += 1
        pairs.append({"sim": round(w, 3), "a": disp(i), "b": disp(j)})
        if len(pairs) >= 14:
            break

    # loners: well-reviewed perfumes whose nearest twin is the least similar (one per name)
    loner_rows, seen_nm = [], set()
    for i in sorted(pop, key=lambda i: wt[i][0]):
        if names[i] in seen_nm:
            continue
        seen_nm.add(names[i])
        loner_rows.append(
            {**disp(i), "sim": round(float(wt[i][0]), 3), "notes": note_disp[i]}
        )
        if len(loner_rows) >= 10:
            break

    # how scent has changed: family share by decade (1950s-2020s)
    decades = list(range(1950, 2030, 10))
    dec_counts = {d: [0] * len(MACRO) for d in decades}
    for i in range(n):
        y = years[i]
        if y:
            d = (y // 10) * 10
            if d in dec_counts:
                dec_counts[d][int(fam_of[i])] += 1
    analyses = {
        "n": n,
        "n_popular": len(pop),
        "img": IMG_TMPL,
        "frag": FRAG_TMPL,
        "facts": facts,
        "families": fam_stats,
        "twins": pairs,
        "loners": loner_rows,
        "decades": {
            "labels": decades,
            "counts": [dec_counts[d] for d in decades],
            "fam_names": [m[0] for m in MACRO],
            "fam_colors": [m[1] for m in MACRO],
        },
    }
    (out / "perfume-analyses.json").write_text(
        json.dumps(analyses, separators=(",", ":"), ensure_ascii=False)
    )

    # ---- write --------------------------------------------------------------
    atlas = {
        "meta": {
            "n": n,
            "k": K_EDGES,
            "alpha_notes": ALPHA_NOTES,
            "img": IMG_TMPL,
            "frag": FRAG_TMPL,
            "gender_labels": GENDER_LABELS,
            "source": "Fragrantica (fra_cleaned)",
            "facts": facts,
        },
        "families": families,
        "pid": pids,
        "name": names,
        "brand": brands,
        "year": years,
        "gender": genders,
        "rating": ratings,
        "reviews": rcs,
        "fam": fam_of.tolist(),
        "x": [round(float(v), 2) for v in emb[:, 0]],
        "y": [round(float(v), 2) for v in emb[:, 1]],
        "notes": note_disp,
        "accords": [", ".join(a) for a in accord_lists],
    }
    (out / "perfumes-atlas.json").write_text(
        json.dumps(atlas, separators=(",", ":"), ensure_ascii=False)
    )
    neighbors = {
        "k": K_EDGES,
        "nbr": [[int(j) for j in nbr[i]] for i in range(n)],
        "w": [[round(float(w), 3) for w in wt[i]] for i in range(n)],
    }
    (out / "perfumes-neighbors.json").write_text(
        json.dumps(neighbors, separators=(",", ":"))
    )
    a_mb = (out / "perfumes-atlas.json").stat().st_size / 1e6
    n_mb = (out / "perfumes-neighbors.json").stat().st_size / 1e6
    print(
        f"[write] atlas={a_mb:.1f}MB neighbors={n_mb:.1f}MB  families={len(families)}"
    )
    print(f"[done] {n} perfumes in {time.time()-t0:.1f}s")
    # quick human sanity check
    print("\nfamilies:")
    for f in families:
        a = f["archetype"]
        print(
            f"  {f['id']:2d} n={f['size']:5d}  {f['name']:28s}  e.g. {names[a]} / {brands[a]}"
        )


if __name__ == "__main__":
    main()
