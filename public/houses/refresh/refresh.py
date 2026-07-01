#!/usr/bin/env python3
"""Daily refresh pipeline for alex-loftus.com/houses.

Two modes:
  --pull   Pull live Craigslist rentals (sapi JSON API), dedupe, filter to Alex's
           zone/budget, attach neighborhood vibe + dual-anchor commute priors,
           select ~50 diverse candidates, scrape each one's photo gallery AND
           posting body text -> writes houses/refresh/shortlist.json.
           Exits non-zero (fails loud) if Craigslist returns too little data, so a
           blocked run never produces a broken/empty map.
  --build  Merge shortlist.json + ratings.json (per-listing LLM scores written by
           the rating agent) with the dual-anchor commute model -> writes
           houses/data.js (window.HOUSES_DATA). Drops dead/commercial/low-fit,
           dedupes by URL, ranks by fit, builds neighborhood cards. Exits non-zero
           if too few listings survive.

Usage (the routine agent does this):
    python3 houses/refresh/refresh.py --pull
    # ... agent reads shortlist.json, writes ratings.json ...
    python3 houses/refresh/refresh.py --build

Alex's brief: SF Bay Area, moving soon. Budget <=$3000 (aim ~$2000). Studio fine OR
housemates for networking (tech founders/CEOs, AI-lab & elite-university crowd —
NOT artists/creative collectives). Has a car; needs easy access to BOTH downtown SF
and downtown Berkeley (FAR Labs), <=~35 min. Loves nature/trees; loves Mill
Valley / Sausalito / Marin; also fine in the city.
"""
import argparse
import collections
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SHORTLIST = os.path.join(HERE, "shortlist.json")
RATINGS = os.path.join(HERE, "ratings.json")
PULL_STATS = os.path.join(HERE, "pull_stats.json")
DATA_JS = os.path.abspath(os.path.join(HERE, "..", "data.js"))

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
SAPI = "https://sapi.craigslist.org/web/v8/postings/search/full"

# Downtown SF (Financial District) commute anchor
DT_LAT, DT_LON = 37.7935, -122.3970

# (name, lat, lon, radius_mi) search centers covering the whole zone
CENTERS = [
    ("sf_inner_bay", 37.78, -122.42, 12),  # SF + Daly City + inner East Bay + Sausalito
    ("east_bay", 37.845, -122.27, 6),  # Berkeley/Oakland/Albany/Emeryville/Piedmont
    ("north_bay", 37.90, -122.52, 9),  # Sausalito/Mill Valley/Tiburon/Corte Madera
]
# (category, min_price, max_price, bucket)
QUERIES = [
    ("apa", 1300, 3100, "apt"),  # apartments/studios/1BR (quiet/private option)
    ("roo", 800, 2100, "room"),  # rooms & shares (housemate/networking option)
]

# Minimum kept listings before we trust a pull; below this we assume the datacenter
# IP got blocked/rate-limited and refuse to build a degraded page.
MIN_KEPT = 30
MIN_SHOWN = 15

IMG_RE = re.compile(
    r"https://images\.craigslist\.org/[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_600x450\.jpg"
)
DEAD_RE = re.compile(
    r"This posting has been (deleted|flagged)|has expired|<title>[^<]*(removed|deleted)",
    re.I,
)
BODY_RE = re.compile(r'<section id="postingbody">(.*?)</section>', re.S)
TAG_RE = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------- #
# --pull
# --------------------------------------------------------------------------- #
def haversine_mi(lat1, lon1, lat2, lon2):
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def curl_json(url):
    out = subprocess.run(
        [
            "curl",
            "-sS",
            "--max-time",
            "30",
            "-A",
            UA,
            "-H",
            "Accept: application/json",
            url,
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0 or not out.stdout:
        print(
            f"  curl failed rc={out.returncode} err={out.stderr[:120]}", file=sys.stderr
        )
        return None
    try:
        return json.loads(out.stdout)
    except Exception as e:
        print(f"  json parse failed: {e}", file=sys.stderr)
        return None


def parse_item(it, locs):
    pid = it[0]
    price = it[3] if len(it) > 3 else None
    geo = it[4] if len(it) > 4 else ""
    lat = lon = hood = None
    if isinstance(geo, str) and "~" in geo:
        p = geo.split("~")
        lp = p[0].split(":")
        if len(lp) > 1:
            try:
                hi = int(lp[1])
                if 0 < hi < len(locs):
                    hood = locs[hi]
            except Exception:
                pass
        try:
            lat = float(p[1])
            lon = float(p[2])
        except Exception:
            pass
    title = slug = token = pdisp = beds = None
    imgs = []
    for el in it[6:]:
        if isinstance(el, str):
            title = el
        elif isinstance(el, list) and el:
            t = el[0]
            if t == 4:
                imgs = el[1:]
            elif t == 6:
                slug = el[1] if len(el) > 1 else None
            elif t == 10:
                pdisp = el[1] if len(el) > 1 else None
            elif t == 13:
                token = el[1] if len(el) > 1 else None
            elif t == 5:
                beds = el[1] if len(el) > 1 else None
    img = None
    if imgs:
        ref = imgs[0]
        if isinstance(ref, str) and ":" in ref:
            core = ref.split(":", 1)[1]  # keep host suffix: 00T0T_8I6TRnex3fH_0nm0hw
            img = f"https://images.craigslist.org/{core}_600x450.jpg"
    url = (
        f"https://www.craigslist.org/view/d/{slug}/{token}" if slug and token else None
    )
    if beds is None and isinstance(title, str):
        m = re.search(r"(\d)\s*br", title.lower())
        if m:
            beds = int(m.group(1))
        elif "studio" in title.lower():
            beds = 0
    return dict(
        pid=pid,
        price=price,
        pdisp=pdisp,
        beds=beds,
        hood=hood,
        lat=lat,
        lon=lon,
        title=title,
        url=url,
        img=img,
        nimg=len(imgs),
    )


def pull_raw():
    seen = {}
    for cname, lat, lon, dist in CENTERS:
        for cat, pmin, pmax, bucket in QUERIES:
            params = {
                "batch": "1-0-360-0-0",
                "cc": "US",
                "lang": "en",
                "searchPath": cat,
                "min_price": pmin,
                "max_price": pmax,
                "lat": lat,
                "lon": lon,
                "search_distance": dist,
                "sort": "date",
                "availabilityMode": 0,
            }
            url = SAPI + "?" + urllib.parse.urlencode(params)
            d = curl_json(url)
            if not d:
                print(f"[{cname}/{cat}] FAILED")
                continue
            data = d["data"]
            locs = data["decode"]["locationDescriptions"]
            items = data["items"]
            n_new = 0
            for raw in items:
                r = parse_item(raw, locs)
                if not r["url"] or r["price"] is None:
                    continue
                r["bucket"] = bucket
                if r["pid"] not in seen:
                    seen[r["pid"]] = r
                    n_new += 1
            print(
                f"[{cname}/{cat}] pulled {len(items)}; +{n_new} new; unique={len(seen)}"
            )
            time.sleep(1.2)  # be polite
    return list(seen.values())


# keyword -> (region, offpeak_drive_min_to_downtownSF, nature, quiet, nice, social) 0-5
RULES = [
    ("tiburon", ("NorthBay", 33, 5, 5, 5, 1)),
    ("belvedere", ("NorthBay", 33, 5, 5, 5, 1)),
    ("mill valley", ("NorthBay", 28, 5, 5, 5, 2)),
    ("sausalito", ("NorthBay", 23, 5, 4, 5, 2)),
    ("corte madera", ("NorthBay", 32, 4, 4, 4, 1)),
    ("larkspur", ("NorthBay", 34, 4, 4, 4, 1)),
    ("greenbrae", ("NorthBay", 34, 4, 4, 4, 1)),
    ("kentfield", ("NorthBay", 40, 5, 5, 5, 1)),
    ("ross", ("NorthBay", 40, 5, 5, 5, 1)),
    ("san rafael", ("NorthBay", 40, 3, 3, 3, 2)),
    ("petaluma", ("NorthBay", 55, 3, 3, 3, 1)),
    ("santa rosa", ("NorthBay", 60, 3, 3, 3, 1)),
    ("north bay", ("NorthBay", 35, 4, 4, 4, 1)),
    ("marin", ("NorthBay", 35, 5, 4, 4, 1)),
    ("pacifica", ("Peninsula", 30, 5, 4, 3, 1)),
    ("brisbane", ("Peninsula", 18, 3, 4, 3, 1)),
    ("daly city", ("Peninsula", 18, 2, 3, 3, 1)),
    ("south san francisco", ("Peninsula", 22, 2, 3, 3, 1)),
    ("south san fran", ("Peninsula", 22, 2, 3, 3, 1)),
    ("san bruno", ("Peninsula", 28, 2, 3, 3, 1)),
    ("westlake", ("Peninsula", 16, 2, 3, 3, 1)),
    ("emeryville", ("EastBay", 24, 3, 3, 4, 3)),
    ("berkeley north", ("EastBay", 36, 5, 5, 5, 2)),
    ("berkeley hills", ("EastBay", 36, 5, 5, 5, 2)),
    ("rockridge", ("EastBay", 32, 4, 4, 5, 4)),
    ("claremont", ("EastBay", 33, 5, 4, 5, 3)),
    ("montclair", ("EastBay", 34, 5, 5, 5, 1)),
    ("piedmont", ("EastBay", 32, 4, 4, 5, 2)),
    ("temescal", ("EastBay", 30, 3, 3, 4, 4)),
    ("oakland north", ("EastBay", 30, 3, 3, 4, 4)),
    ("lake merrit", ("EastBay", 28, 3, 3, 4, 4)),
    ("lake merritt", ("EastBay", 28, 3, 3, 4, 4)),
    ("oakland downtown", ("EastBay", 26, 2, 2, 3, 4)),
    ("old oakland", ("EastBay", 26, 2, 2, 3, 4)),
    ("oakland west", ("EastBay", 24, 2, 2, 2, 3)),
    ("oakland east", ("EastBay", 34, 3, 2, 2, 2)),
    ("oakland hills", ("EastBay", 36, 5, 5, 4, 1)),
    ("oakland estuary", ("EastBay", 28, 3, 3, 3, 2)),
    ("oakland", ("EastBay", 30, 3, 3, 3, 3)),
    ("alameda", ("EastBay", 30, 3, 4, 4, 2)),
    ("albany", ("EastBay", 36, 3, 4, 4, 2)),
    ("el cerrito", ("EastBay", 38, 3, 4, 3, 2)),
    ("richmond", ("EastBay", 36, 2, 2, 2, 1)),
    ("berkeley", ("EastBay", 33, 4, 4, 4, 4)),
    ("nob hill", ("SF", 8, 1, 2, 4, 4)),
    ("russian hill", ("SF", 9, 2, 3, 5, 4)),
    ("polk gulch", ("SF", 8, 1, 2, 3, 4)),
    ("tenderloin", ("SF", 7, 1, 1, 1, 2)),
    ("soma", ("SF", 6, 1, 2, 3, 4)),
    ("south beach", ("SF", 7, 2, 3, 4, 4)),
    ("downtown", ("SF", 6, 1, 2, 3, 4)),
    ("civic", ("SF", 7, 1, 2, 2, 3)),
    ("van ness", ("SF", 8, 1, 2, 3, 3)),
    ("hayes valley", ("SF", 9, 2, 3, 5, 5)),
    ("marina", ("SF", 13, 3, 3, 5, 5)),
    ("cow hollow", ("SF", 13, 3, 3, 5, 5)),
    ("pacific heights", ("SF", 12, 2, 4, 5, 3)),
    ("pac hts", ("SF", 12, 2, 4, 5, 3)),
    ("pac heights", ("SF", 12, 2, 4, 5, 3)),
    ("laurel", ("SF", 13, 3, 4, 5, 2)),
    ("presidio", ("SF", 14, 5, 4, 5, 2)),
    ("noe valley", ("SF", 14, 4, 5, 5, 3)),
    ("castro", ("SF", 12, 2, 3, 5, 5)),
    ("upper market", ("SF", 13, 2, 3, 4, 4)),
    ("mission", ("SF", 11, 2, 2, 4, 5)),
    ("potrero", ("SF", 10, 3, 3, 4, 3)),
    ("bernal", ("SF", 13, 3, 4, 4, 3)),
    ("alamo", ("SF", 11, 3, 3, 5, 4)),
    ("nopa", ("SF", 11, 3, 3, 4, 4)),
    ("haight", ("SF", 12, 4, 3, 4, 4)),
    ("twin peaks", ("SF", 16, 4, 4, 4, 1)),
    ("diamond", ("SF", 16, 4, 4, 4, 1)),
    ("forest hill", ("SF", 17, 5, 5, 5, 1)),
    ("west portal", ("SF", 18, 4, 5, 5, 2)),
    ("inner sunset", ("SF", 16, 4, 4, 5, 3)),
    ("sunset", ("SF", 18, 3, 4, 4, 2)),
    ("parkside", ("SF", 18, 3, 4, 4, 1)),
    ("inner richmond", ("SF", 15, 4, 4, 4, 3)),
    ("richmond / seacliff", ("SF", 15, 5, 4, 5, 2)),
    ("seacliff", ("SF", 15, 5, 4, 5, 2)),
    ("excelsior", ("SF", 16, 2, 3, 3, 2)),
    ("outer mission", ("SF", 15, 2, 3, 3, 2)),
    ("ingleside", ("SF", 16, 2, 3, 3, 2)),
    ("sfsu", ("SF", 16, 2, 3, 3, 2)),
    ("ccsf", ("SF", 16, 2, 3, 3, 2)),
    ("portola", ("SF", 15, 2, 3, 3, 1)),
    ("bayview", ("SF", 13, 2, 2, 2, 1)),
    ("northwest san francisco", ("SF", 13, 3, 3, 4, 2)),
    ("city of san francisco", ("SF", 12, 2, 3, 3, 3)),
    ("san francisco", ("SF", 12, 2, 3, 3, 3)),
]
DEFAULT_PRIOR = ("Other", 30, 3, 3, 3, 2)


def priors(hood):
    h = (hood or "").lower().strip()
    # SF Richmond district vs East Bay Richmond city disambiguation.
    if "richmond" in h and ("seacliff" in h or "inner" in h):
        return ("SF", 15, 5, 4, 5, 2)
    # SF Marina/Cow Hollow contain the substring "marin", so the ordered scan
    # would hit the NorthBay "marin" rule first — guard them explicitly.
    # (Regression: this bug shipped once and mislabeled Fort Mason as NorthBay.)
    if "marina" in h or "cow hollow" in h:
        return ("SF", 13, 3, 3, 5, 5)
    for kw, p in RULES:
        if kw in h:
            return p
    return DEFAULT_PRIOR


def price_fit(r):
    p = r["price"]
    if r["bucket"] == "apt":
        if p <= 2200:
            return 5.0 - max(0, (1500 - p)) / 1500 * 1.0
        return max(0, 5.0 - (p - 2200) / 800 * 3.0)
    if p <= 1400:
        return 5.0
    return max(0, 5.0 - (p - 1400) / 700 * 3.0)


def commute_fit(dmin):
    if dmin <= 20:
        return 5.0
    if dmin >= 40:
        return 0.0
    return 5.0 * (40 - dmin) / 20


def seed_fit(r):
    soft = r["quiet"] if r["bucket"] == "apt" else r["social"]
    return round(
        0.27 * r["nice"]
        + 0.23 * r["nature"]
        + 0.18 * soft
        + 0.17 * price_fit(r)
        + 0.15 * commute_fit(r["drive_min"]),
        3,
    )


def norm_hood(h):
    return re.sub(r"\s+", " ", (h or "").lower().strip())


def select_shortlist(rows):
    """Filter to zone/budget, attach priors + seed fit, select ~50 diverse."""
    for r in rows:
        if r["lat"] and r["lon"]:
            r["mi_to_dt"] = round(haversine_mi(r["lat"], r["lon"], DT_LAT, DT_LON), 1)
        else:
            r["mi_to_dt"] = None
    keep = []
    for r in rows:
        if r["mi_to_dt"] is None or r["mi_to_dt"] > 14:
            continue
        if r["bucket"] == "apt" and r["price"] > 3050:
            continue
        if r["bucket"] == "room" and r["price"] > 2100:
            continue
        if r["price"] < 700 or r["nimg"] == 0:
            continue
        keep.append(r)

    for r in keep:
        reg, dmin, nat, quiet, nice, soc = priors(r["hood"])
        r["region"], r["drive_min"] = reg, dmin
        r["nature"], r["quiet"], r["nice"], r["social"] = nat, quiet, nice, soc
        r["fit"] = seed_fit(r)

    pool = [r for r in keep if r["drive_min"] <= 38]
    pool.sort(key=lambda r: -r["fit"])

    region_target = {"SF": 24, "EastBay": 18, "NorthBay": 9, "Peninsula": 7, "Other": 2}
    per_hood_cap = 3
    sel, by_region, by_hood, by_bucket = (
        [],
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    for r in pool:
        reg, nh = r["region"], norm_hood(r["hood"])
        if by_region[reg] >= region_target.get(reg, 3):
            continue
        if by_hood[nh] >= per_hood_cap:
            continue
        tot = len(sel)
        if tot >= 8:
            frac_apt = by_bucket["apt"] / tot
            if r["bucket"] == "apt" and frac_apt > 0.66:
                continue
            if r["bucket"] == "room" and (1 - frac_apt) > 0.55:
                continue
        sel.append(r)
        by_region[reg] += 1
        by_hood[nh] += 1
        by_bucket[r["bucket"]] += 1
        if len(sel) >= 55:
            break
    return keep, sel


def scrape_page(r):
    """Fetch a listing page; extract full photo gallery + posting body text."""
    try:
        out = subprocess.run(
            ["curl", "-sSL", "--max-time", "25", "-A", UA, r["url"]],
            capture_output=True,
            text=True,
            timeout=30,
        )
        html = out.stdout
    except Exception as e:
        return r["id"], {"imgs": [], "body": "", "status": f"error:{e}"}
    if not html:
        return r["id"], {"imgs": [], "body": "", "status": "empty"}
    dead = bool(DEAD_RE.search(html))
    seen, imgs = set(), []
    for u in IMG_RE.findall(html):
        if u not in seen:
            seen.add(u)
            imgs.append(u)
    body = ""
    m = BODY_RE.search(html)
    if m:
        body = TAG_RE.sub(" ", m.group(1))
        body = re.sub(r"\s+", " ", body).strip()[:1600]
    return r["id"], {
        "imgs": imgs[:24],
        "body": body,
        "status": "dead" if dead else ("ok" if imgs else "noimg"),
    }


def do_pull():
    print("== PULL ==")
    rows = pull_raw()
    if len(rows) < MIN_KEPT:
        sys.exit(
            f"FATAL: only {len(rows)} raw listings pulled from Craigslist "
            f"(expected many). The datacenter IP was likely blocked/rate-limited. "
            f"Refusing to build a degraded page."
        )
    keep, sel = select_shortlist(rows)
    if len(sel) < MIN_KEPT:
        sys.exit(
            f"FATAL: only {len(sel)} in-zone candidates after filtering "
            f"(< {MIN_KEPT}). Aborting rather than shipping a thin map."
        )
    # stable ids, then scrape galleries + bodies in parallel
    for i, r in enumerate(sel):
        r["id"] = f"L{i + 1:02d}"
    print(f"scraping {len(sel)} listing pages (photos + body)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        scraped = dict(ex.map(scrape_page, sel))
    n_dead = n_body = 0
    for r in sel:
        info = scraped.get(r["id"], {})
        gallery = info.get("imgs") or ([r["img"]] if r["img"] else [])
        r["imgs"] = gallery
        r["img"] = gallery[0] if gallery else r["img"]
        r["nimg"] = len(gallery)
        r["body"] = info.get("body", "")
        r["scrape_status"] = info.get("status", "?")
        if info.get("status") == "dead":
            n_dead += 1
        if r["body"]:
            n_body += 1
    json.dump(sel, open(SHORTLIST, "w"), indent=1)
    json.dump(
        {
            "n_raw": len(rows),
            "n_kept": len(keep),
            "n_shortlist": len(sel),
            "pulled": datetime.date.today().isoformat(),
        },
        open(PULL_STATS, "w"),
    )
    print(
        f"wrote {SHORTLIST}: {len(sel)} candidates "
        f"(raw={len(rows)} kept={len(keep)}) | bodies={n_body} dead={n_dead}"
    )
    print("by region:", dict(collections.Counter(r["region"] for r in sel)))
    print("by bucket:", dict(collections.Counter(r["bucket"] for r in sel)))


# --------------------------------------------------------------------------- #
# --build
# --------------------------------------------------------------------------- #
def berk_drive(hood, region):
    """Off-peak driving minutes to downtown Berkeley (Shattuck/University)."""
    h = (hood or "").lower()
    if "emeryville" in h:
        return 9
    if "rockridge" in h or "claremont" in h:
        return 8
    if "temescal" in h or "oakland north" in h:
        return 8
    if "piedmont" in h or "montclair" in h:
        return 13
    if "lake merrit" in h:
        return 12
    if "old oakland" in h or "oakland downtown" in h:
        return 13
    if "oakland west" in h:
        return 12
    if "oakland east" in h:
        return 18
    if "oakland hills" in h or "mills" in h:
        return 18
    if "estuary" in h:
        return 15
    if "albany" in h or "el cerrito" in h:
        return 9
    if "alameda" in h:
        return 20
    if "berkeley north" in h or "berkeley hills" in h:
        return 10
    if "berkeley" in h:
        return 6
    if region == "EastBay" or "east bay" in h or region == "Other":
        return 14
    if "tiburon" in h or "belvedere" in h:
        return 45
    if "mill valley" in h:
        return 42
    if "sausalito" in h:
        return 38
    if "corte madera" in h or "larkspur" in h or "greenbrae" in h:
        return 38
    if "san rafael" in h or "kentfield" in h or "ross" in h:
        return 33
    if region == "NorthBay":
        return 40
    if "pacifica" in h:
        return 48
    if "daly city" in h or "brisbane" in h or "westlake" in h:
        return 38
    if "south san fran" in h or "san bruno" in h:
        return 40
    if region == "Peninsula":
        return 40
    if "seacliff" in h or "richmond" in h:
        return 35
    if "sunset" in h or "parkside" in h:
        return 36
    if "forest hill" in h or "west portal" in h or "twin peaks" in h or "diamond" in h:
        return 36
    if "bayview" in h:
        return 26
    return 29  # central SF default


def cscore(m):  # minutes -> 0..10 (<=18 min is perfect)
    return max(0.0, min(10.0, 10 - max(0, m - 18) / 4.2))


def gs(scores, k, d=5):
    v = scores.get(k)
    return d if v is None else v


LOVED = ("mill valley", "sausalito")
FERRY_SF = ("sausalito", "mill valley", "tiburon", "larkspur", "corte madera")

SEARCHLINKS = [
    {
        "group": "San Francisco",
        "links": [
            {
                "label": "Craigslist SF · apts $1.5–3k",
                "url": "https://sfbay.craigslist.org/search/sfc/apa?min_price=1500&max_price=3000&availabilityMode=0&sort=date",
            },
            {
                "label": "Craigslist SF · rooms/shares",
                "url": "https://sfbay.craigslist.org/search/sfc/roo?max_price=2000&availabilityMode=0&sort=date",
            },
            {
                "label": "Zillow · SF rentals ≤$3k",
                "url": "https://www.zillow.com/san-francisco-ca/rentals/?searchQueryState=%7B%22filterState%22%3A%7B%22mp%22%3A%7B%22max%22%3A3000%7D%7D%7D",
            },
            {
                "label": "Apartments.com · SF ≤$3k",
                "url": "https://www.apartments.com/san-francisco-ca/under-3000/",
            },
        ],
    },
    {
        "group": "East Bay (Berkeley / Oakland)",
        "links": [
            {
                "label": "Craigslist East Bay · apts",
                "url": "https://sfbay.craigslist.org/search/eby/apa?min_price=1400&max_price=3000&availabilityMode=0&sort=date",
            },
            {
                "label": "Craigslist East Bay · rooms",
                "url": "https://sfbay.craigslist.org/search/eby/roo?max_price=1800&availabilityMode=0&sort=date",
            },
            {
                "label": "Apartments.com · Berkeley ≤$3k",
                "url": "https://www.apartments.com/berkeley-ca/under-3000/",
            },
        ],
    },
    {
        "group": "North Bay (Sausalito / Marin)",
        "links": [
            {
                "label": "Craigslist North Bay · apts",
                "url": "https://sfbay.craigslist.org/search/nby/apa?min_price=1500&max_price=3100&availabilityMode=0&sort=date",
            },
            {
                "label": "Apartments.com · Sausalito",
                "url": "https://www.apartments.com/sausalito-ca/",
            },
        ],
    },
    {
        "group": "Peninsula (Daly City / Pacifica)",
        "links": [
            {
                "label": "Craigslist Peninsula · apts",
                "url": "https://sfbay.craigslist.org/search/pen/apa?min_price=1400&max_price=3000&availabilityMode=0&sort=date",
            },
        ],
    },
    {
        "group": "Aggregators & rooms/housemates",
        "links": [
            {
                "label": "HotPads map ≤$3k",
                "url": "https://hotpads.com/san-francisco-ca/apartments-for-rent?price=0,3000",
            },
            {
                "label": "PadMapper (map, all sources)",
                "url": "https://www.padmapper.com/apartments/san-francisco-ca?box=-122.55,37.70,-122.35,37.83&maxPrice=3000",
            },
            {
                "label": "Zumper · SF ≤$3k",
                "url": "https://www.zumper.com/apartments-for-rent/san-francisco-ca?max-price=3000",
            },
            {
                "label": "SpareRoom (housemates)",
                "url": "https://www.spareroom.com/roommates/san_francisco",
            },
        ],
    },
]


def build_neighborhoods(listings):
    """Deterministic neighborhood cards: group shown listings by hood, average the
    LLM component scores. Keeps groups with >=2 listings, top ~10 by avg fit."""
    groups = collections.defaultdict(list)
    for x in listings:
        groups[norm_hood(x["hood"])].append(x)
    cards = []
    for nh, xs in groups.items():
        if len(xs) < 2 or not nh:
            continue

        def avg(k):
            vals = [gs(x["scores"], k) for x in xs]
            return round(sum(vals) / len(vals), 1)

        name = max((x["hood"] for x in xs), key=lambda h: len(h or ""))
        cards.append(
            {
                "name": name,
                "region": xs[0]["region"],
                "n": len(xs),
                "avg_fit": round(sum(x["fit"] for x in xs) / len(xs), 1),
                "drive_berk": berk_drive(name, xs[0]["region"]),
                "scores": {k: avg(k) for k in ["nature", "quiet", "nice", "social"]},
            }
        )
    cards.sort(key=lambda c: -c["avg_fit"])
    return cards[:10]


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# US phone: optional +1, then 3-3-4 with common separators/parens.
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?\(?([2-9]\d{2})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})(?!\d)"
)


def extract_contact(body):
    """Pull a poster-supplied email / phone out of the listing body, if present.

    Craigslist hides its own relay email behind the reply flow, but many posters
    paste a real email/phone into the post so people can contact them directly.
    Returns (email_or_None, phone_or_None). Craigslist relay + image/no-reply
    addresses are skipped.
    """
    if not body:
        return None, None
    email = None
    for m in EMAIL_RE.findall(body):
        low = m.lower()
        if any(
            bad in low
            for bad in (
                "craigslist.org",
                "reply.craigslist",
                "example.",
                "@2x",
                ".png",
                ".jpg",
            )
        ):
            continue
        email = m
        break
    phone = None
    pm = PHONE_RE.search(body)
    if pm:
        phone = f"({pm.group(1)}) {pm.group(2)}-{pm.group(3)}"
    return email, phone


def do_build():
    print("== BUILD ==")
    if not os.path.exists(SHORTLIST):
        sys.exit(f"FATAL: {SHORTLIST} missing — run --pull first.")
    if not os.path.exists(RATINGS):
        sys.exit(
            f"FATAL: {RATINGS} missing — the rating agent must write it before --build."
        )
    sel = {r["id"]: r for r in json.load(open(SHORTLIST))}
    rated = json.load(open(RATINGS))
    ratings = {rr["id"]: rr for rr in rated}
    print(f"shortlist={len(sel)} rated={len(ratings)}")

    listings = []
    for lid, r in sel.items():
        rt = ratings.get(lid)
        if rt is None:
            continue  # agent didn't rate it -> skip
        if rt.get("live") is False or rt.get("commercial") is True:
            continue
        if (rt.get("fit") or 0) <= 2:
            continue
        # scores may be nested under "scores" or flat on the object
        src = rt.get("scores", rt)
        scores = {
            k: src.get(k)
            for k in [
                "nature",
                "quiet",
                "nice",
                "social",
                "value",
                "commute",
                "aesthetic",
            ]
        }
        rationale = rt.get("rationale") or rt.get("why") or ""
        gallery = r.get("imgs") or ([r["img"]] if r.get("img") else [])
        email, phone = extract_contact(r.get("body", ""))
        listings.append(
            {
                "id": lid,
                "price": r["price"],
                "pdisp": r.get("pdisp") or f"${r['price']:,}",
                "beds": r.get("beds"),
                "bucket": r["bucket"],
                "hood": r["hood"],
                "region": r["region"],
                "lat": r["lat"],
                "lon": r["lon"],
                "url": r["url"],
                "img": gallery[0] if gallery else r.get("img"),
                "imgs": gallery,
                "nimg": len(gallery),
                "drive_min": r["drive_min"],
                "title": r.get("title", ""),
                "fit": rt.get("fit"),
                "scores": scores,
                "rationale": rationale,
                "contact_email": email,
                "contact_phone": phone,
            }
        )

    # dual-anchor commute + final fit (blends LLM component scores w/ factual dual commute)
    for x in listings:
        x["drive_sf"] = x["drive_min"]
        x["drive_berk"] = berk_drive(x["hood"], x["region"])
        worst = max(x["drive_sf"], x["drive_berk"])
        avg = (x["drive_sf"] + x["drive_berk"]) / 2
        x["dual_commute"] = round(0.65 * cscore(worst) + 0.35 * cscore(avg), 1)
        x["ferry_sf"] = any(k in (x["hood"] or "").lower() for k in FERRY_SF)
        x["loved"] = any(k in (x["hood"] or "").lower() for k in LOVED)
        s = x["scores"]
        soft = gs(s, "quiet") if x["bucket"] == "apt" else gs(s, "social")
        # aesthetic = how good the place looks in its photos (weighted so listings
        # with ugly/low-effort photos don't rank high on looks alone). gs() defaults
        # missing scores to 5 (neutral) for backward compat with pre-aesthetic ratings.
        fit = (
            0.17 * gs(s, "nice")
            + 0.15 * gs(s, "nature")
            + 0.13 * soft
            + 0.13 * gs(s, "value")
            + 0.26 * x["dual_commute"]
            + 0.16 * gs(s, "aesthetic")
        )
        if x["loved"]:
            fit += 0.8
        x["fit"] = round(min(10.0, fit), 1)
        s["commute"] = round(x["dual_commute"], 1)

    # dedupe by URL (keep higher fit)
    by_url = {}
    for x in sorted(listings, key=lambda x: -(x["fit"] or 0)):
        by_url.setdefault(x["url"], x)
    listings = sorted(by_url.values(), key=lambda x: -(x["fit"] or 0))

    if len(listings) < MIN_SHOWN:
        sys.exit(
            f"FATAL: only {len(listings)} listings survived rating/filtering "
            f"(< {MIN_SHOWN}). Not overwriting data.js with a thin page."
        )

    # top picks: genuinely good fit, keep region variety
    seen_reg = {}
    for x in listings:
        reg = x["region"]
        if x["fit"] >= 7 and seen_reg.get(reg, 0) < 2 and sum(seen_reg.values()) < 6:
            x["pick"] = True
            seen_reg[reg] = seen_reg.get(reg, 0) + 1
        else:
            x["pick"] = False

    neighborhoods = build_neighborhoods(listings)
    today = datetime.date.today().isoformat()
    stats = {}
    if os.path.exists(PULL_STATS):
        stats = json.load(open(PULL_STATS))
    meta = {
        "generated": today,
        "n_scouted": stats.get("n_kept", len(sel)),
        "n_shortlist": stats.get("n_shortlist", len(sel)),
        "n_shown": len(listings),
        "source": f"Craigslist (live API), refreshed {today}",
        "anchors": "downtown SF + downtown Berkeley (FAR Labs)",
        "price_min": min(x["price"] for x in listings),
        "price_max": max(x["price"] for x in listings),
        "price_med": int(statistics.median(x["price"] for x in listings)),
        "move_by": "~late July 2026",
        "mode": "FINAL",
    }
    data = {
        "meta": meta,
        "listings": listings,
        "neighborhoods": neighborhoods,
        "searchlinks": SEARCHLINKS,
    }
    with open(DATA_JS, "w") as f:
        f.write(
            "window.HOUSES_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
        )
    picks = [x for x in listings if x["pick"]]
    print(
        f"wrote {DATA_JS}: shown={len(listings)} picks={len(picks)} neighborhoods={len(neighborhoods)}"
    )
    print("top 6:", [(x["id"], x["hood"], x["fit"]) for x in listings[:6]])


def main():
    ap = argparse.ArgumentParser(description="Daily refresh for alex-loftus.com/houses")
    ap.add_argument("--pull", action="store_true", help="pull + shortlist + scrape")
    ap.add_argument("--build", action="store_true", help="merge ratings -> data.js")
    args = ap.parse_args()
    if args.pull:
        do_pull()
    elif args.build:
        do_build()
    else:
        ap.error("specify --pull or --build")


if __name__ == "__main__":
    main()
