"""Refresh pipeline for alex-loftus.com/jobs — Frontier AI Jobs board.

Two layers share one payload (data.js):
  PUBLIC  — every open role at the tracked frontier-AI labs, refreshed daily,
            categorized, with comp/location/remote/NEW tracking. No LLM cost.
  SCORED  — roles matching the profile lane additionally carry fit scores,
            offer probabilities and prep decks (rate.py, Claude API). This is
            Alex's layer and the live demo of the paid personal tier
            (see README_PRO.md / make_user.py).

Phases (mirrors public/houses/refresh/refresh.py):

  --pull   Fetch every configured ATS board (Greenhouse / Ashby / Lever JSON
           APIs, no auth). Keep ALL roles; mark lane roles; diff against the
           previous data.js; write shortlist.json = lane roles that still
           need LLM scoring. Fails loud if Anthropic < 200 jobs so a
           blocked/broken pull can never publish a degraded board.

  --build  Merge ratings.json (written by rate.py) into the previous payload,
           compute deterministic fit for scored roles, rank, write data.js.

  --sweep  Re-pull boards: update open/closed on known roles AND append
           newly-posted roles (unscored — they get scored at the next full
           refresh). No LLM.
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data.js")
SHORTLIST = os.path.join(HERE, "shortlist.json")
RATINGS = os.path.join(HERE, "ratings.json")
PULL_STATS = os.path.join(HERE, "pull_stats.json")
PREP_BANK = os.path.join(HERE, "prep_bank.json")
LIVE_INDEX = os.path.join(HERE, "live_index.json")

TODAY = date.today().isoformat()
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) jobs-refresh/1.0"

# ---------------------------------------------------------------- boards ----
# group: anthropic | interp (interp/safety orgs) | frontier.
# Redwood Research + EleutherAI have no public ATS JSON API as of 2026-07 —
# revisit occasionally.
BOARDS = [
    {
        "company": "Anthropic",
        "ats": "greenhouse",
        "slug": "anthropic",
        "group": "anthropic",
    },
    {"company": "Goodfire", "ats": "greenhouse", "slug": "goodfire", "group": "interp"},
    {
        "company": "Apollo Research",
        "ats": "lever",
        "slug": "apolloresearch",
        "group": "interp",
    },
    {"company": "METR", "ats": "lever", "slug": "metr", "group": "interp"},
    {
        "company": "DeepMind",
        "ats": "greenhouse",
        "slug": "deepmind",
        "group": "frontier",
    },
    {"company": "OpenAI", "ats": "ashby", "slug": "openai", "group": "frontier"},
    {"company": "xAI", "ats": "greenhouse", "slug": "xai", "group": "frontier"},
    {"company": "Mistral", "ats": "lever", "slug": "mistral", "group": "frontier"},
]

# Roles must smell like the profile's lane before we spend LLM tokens on them.
PREFILTER = re.compile(
    r"program\s+(manager|management|lead)|technical\s+program|\bTPM\b"
    r"|research\s+(manager|program|operations|lead|communicat)"
    r"|interpretab|sabotage|\beval(s|uation)?\b|red[\s-]?team|alignment"
    r"|model\s+(safety|welfare|behavior)|safety\s+(research|program|institute)"
    r"|chief\s+of\s+staff|science\s+communicat|technical\s+writer"
    r"|developer\s+relations|solutions?\s+(architect|engineer)|forward\s+deployed",
    re.I,
)

MIN_FIT_NON_ANTHROPIC = 6.5  # below this, non-Anthropic roles leave the scored view
TOMBSTONE_DAYS = 30  # closed scored roles pruned after this long

# ------------------------------------------------------------- categories ---
# First match wins — safety/policy before the broad research/engineering nets.
CATEGORY_RULES = [
    (
        "Safety",
        r"safety|alignment|red[\s-]?team|interpretab|safeguard|welfare|trust\b"
        r"|responsible|preparedness|dangerous capab",
    ),
    (
        "Policy",
        r"policy|government|public affairs|societal|communications|press|media",
    ),
    ("Research", r"research|scientist"),
    (
        "Engineering",
        r"engineer|infrastructure|developer|software|\bsre\b|platform|systems"
        r"|hardware|silicon|network|security|\bdata\b|machine learning|\bml\b"
        r"|technical program|\btpm\b|forward deployed|solutions? architect",
    ),
    ("Product", r"product|design"),
    (
        "GTM",
        r"sales|account|go.to.market|\bgtm\b|marketing|growth|partner"
        r"|business development|customer|revenue|success",
    ),
    (
        "Ops",
        r"recruit|sourcer|people|talent|workplace|executive assistant|office"
        r"|finance|legal|counsel|accounting|procurement|admin|program manager"
        r"|chief of staff|operations|equity|real estate",
    ),
]
CATEGORY_RES = [(name, re.compile(pat, re.I)) for name, pat in CATEGORY_RULES]


def categorize(title):
    for name, rx in CATEGORY_RES:
        if rx.search(title):
            return name
    return "Other"


# ------------------------------------------------------------ fit formula ---
# Role-shape weights = what Alex wants (2026-06-17): lead people, present to
# stakeholders, be social, stay technical under his interp background.
FIT_W = {
    "people": 0.24,
    "stage": 0.20,
    "social": 0.12,
    "technical": 0.22,
    "domain": 0.22,
}


def fit_score(scores):
    f = sum(FIT_W[k] * float(scores.get(k, 0)) for k in FIT_W)
    return min(10.0, round(f, 1))


# ----------------------------------------------------------------- payload --
def load_prev_payload():
    """Previous payload from data.js, or None on first run."""
    if not os.path.exists(DATA):
        return None
    src = open(DATA).read()
    m = re.search(r"window\.JOBS_DATA\s*=\s*(\{.*\});?\s*$", src, re.S)
    assert m, "data.js exists but has no window.JOBS_DATA blob"
    return json.loads(m.group(1))


def payload_meta(jobs, day1=False, swept=None):
    # NEW = first seen today and still open — survives same-day sweeps/builds.
    new_ids = (
        []
        if day1
        else [j["id"] for j in jobs if j.get("first_seen") == TODAY and not j.get("closed")]
    )
    scored = [j for j in jobs if j.get("scores")]
    open_jobs = [j for j in jobs if not j.get("closed")]
    return {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "date": TODAY,
        **({"swept": swept} if swept else {}),
        "total": len(jobs),
        "open": len(open_jobs),
        "scored": len(scored),
        "shown": sum(1 for j in scored if not j.get("hidden") and not j.get("closed")),
        "new_today": len(new_ids),
        "new_ids": new_ids,
        "fit_weights": FIT_W,
        "companies": sorted({j["company"] for j in jobs}),
        "categories": sorted({j.get("category", "Other") for j in jobs}),
    }


def write_payload(payload):
    with open(DATA, "w") as f:
        f.write(
            "// Data for alex-loftus.com/jobs — generated by refresh/refresh.py.\n"
            "window.JOBS_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
        )
    print(
        f"wrote {os.path.relpath(DATA, HERE)} "
        f"({len(payload['jobs'])} roles, {os.path.getsize(DATA)//1024} KB)"
    )


# ------------------------------------------------------------- fetch/parse --
def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


TAG_RE = re.compile(r"<[^>]+>")
COMP_RE = re.compile(r"\$(\d{2,3}),(\d{3})\s*(?:—|–|−|-|to)\s*\$(\d{2,3}),(\d{3})")
COMP_K_RE = re.compile(r"\$(\d{2,4})[Kk]\s*(?:—|–|−|-|to)\s*\$(\d{2,4})[Kk]")


def strip_html(s):
    # Greenhouse ?content=true entity-escapes the HTML once, and the HTML's own
    # text nodes contain entities (&amp;mdash; -> &mdash;) — so unescape, strip
    # tags, then unescape again for the text-node layer.
    return html.unescape(TAG_RE.sub(" ", html.unescape(s or ""))).replace("\xa0", " ")


def parse_comp(text):
    m = COMP_RE.search(text or "")
    if m:
        lo, hi = int(m.group(1) + m.group(2)), int(m.group(3) + m.group(4))
    else:
        m = COMP_K_RE.search(text or "")
        if not m:
            return None
        lo, hi = int(m.group(1)) * 1000, int(m.group(2)) * 1000
    return [lo, hi] if 40_000 <= lo <= hi <= 2_000_000 else None


def _norm(company, group, rid, title, url, locations, remote, published, jd):
    jd_full = re.sub(r"\s+", " ", jd or "").strip()
    jd = jd_full[:7000]  # cap what we ship to the rater; comp parses the full text
    title = title.strip()
    return {
        "id": rid,
        "company": company,
        "group": group,
        "title": title,
        "category": categorize(title),
        "lane": bool(group == "anthropic" or PREFILTER.search(title)),
        "url": url,
        "locations": locations,
        "remote": bool(remote),
        "comp": parse_comp(jd_full),
        "published": (published or "")[:10],
        "jd_text": jd,
    }


def fetch_greenhouse(b):
    d = _get_json(
        f"https://boards-api.greenhouse.io/v1/boards/{b['slug']}/jobs?content=true"
    )
    out = []
    for j in d.get("jobs", []):
        loc = (j.get("location") or {}).get("name", "") or ""
        locs = [s.strip() for s in re.split(r"[|;]", loc) if s.strip()]
        out.append(
            _norm(
                b["company"],
                b["group"],
                f"gh-{b['slug']}-{j['id']}",
                j["title"],
                j["absolute_url"],
                locs,
                "remote" in loc.lower(),
                j.get("first_published") or j.get("updated_at"),
                strip_html(j.get("content", "")),
            )
        )
    return out


def fetch_ashby(b):
    d = _get_json(f"https://api.ashbyhq.com/posting-api/job-board/{b['slug']}")
    out = []
    for j in d.get("jobs", []):
        if not j.get("isListed", True):
            continue
        locs = [j.get("location") or ""] + [
            s.get("location", "") for s in j.get("secondaryLocations", [])
        ]
        locs = [s for s in locs if s]
        out.append(
            _norm(
                b["company"],
                b["group"],
                f"ab-{b['slug']}-{j['id']}",
                j["title"],
                j.get("jobUrl") or j.get("applyUrl"),
                locs,
                j.get("isRemote") or j.get("workplaceType") == "Remote",
                j.get("publishedAt"),
                j.get("descriptionPlain", ""),
            )
        )
    return out


def fetch_lever(b):
    d = _get_json(f"https://api.lever.co/v0/postings/{b['slug']}?mode=json")
    out = []
    for j in d:
        cats = j.get("categories") or {}
        locs = [cats.get("location") or ""] + (cats.get("allLocations") or [])
        locs = sorted({s for s in locs if s})
        pub = j.get("createdAt")
        pub = (
            datetime.fromtimestamp(pub / 1000, timezone.utc).date().isoformat()
            if pub
            else ""
        )
        wt = (j.get("workplaceType") or "").lower()
        out.append(
            _norm(
                b["company"],
                b["group"],
                f"lv-{b['slug']}-{j['id']}",
                j["text"],
                j["hostedUrl"],
                locs,
                wt == "remote" or "remote" in (cats.get("location") or "").lower(),
                pub,
                (j.get("descriptionPlain") or "")
                + " "
                + (j.get("additionalPlain") or ""),
            )
        )
    return out


FETCHERS = {"greenhouse": fetch_greenhouse, "ashby": fetch_ashby, "lever": fetch_lever}

GENERIC_KEYS = (
    "company",
    "group",
    "title",
    "category",
    "lane",
    "url",
    "locations",
    "remote",
    "comp",
    "published",
)


def failed_companies(stats):
    """Companies whose board fetch failed this run — their roles must not be
    closed/tombstoned off the board (an API outage is not a closed req)."""
    return {c for c, v in stats.items() if isinstance(v, str) and not c.startswith("_")}


def pull_boards():
    """Fetch all boards in parallel. A dead board logs and skips."""
    results, stats = [], {}

    def one(b):
        try:
            roles = FETCHERS[b["ats"]](b)
            stats[b["company"]] = len(roles)
            return roles
        except Exception as e:  # noqa: BLE001 — log+skip
            print(f"[{b['company']}] FAILED: {e}", file=sys.stderr)
            stats[b["company"]] = f"FAILED: {e}"
            return []

    with ThreadPoolExecutor(max_workers=8) as ex:
        for roles in ex.map(one, BOARDS):
            results.extend(roles)
    n_anthropic = sum(1 for r in results if r["company"] == "Anthropic")
    assert n_anthropic >= 200, (
        f"Anthropic pull returned only {n_anthropic} jobs — "
        "refusing to publish a degraded board"
    )
    stats["_total"] = len(results)
    stats["_lane"] = sum(1 for r in results if r["lane"])
    return results, stats


# ---------------------------------------------------------------- phases ----
def cmd_pull():
    prev = load_prev_payload()
    prev_jobs = {j["id"]: j for j in (prev or {}).get("jobs", [])}

    live, stats = pull_boards()
    need = [
        r
        for r in live
        if r["lane"]
        and (
            r["id"] not in prev_jobs
            or prev_jobs[r["id"]].get("scored_title", prev_jobs[r["id"]].get("title"))
            != r["title"]
            or "scores" not in prev_jobs[r["id"]]
            or (
                prev_jobs[r["id"]].get("fit", 0) >= 6.0
                and "prep" not in prev_jobs[r["id"]]
            )
        )
    ]
    with open(SHORTLIST, "w") as f:
        json.dump(need, f, indent=1)
    # live_index: --build uses this to set open/closed without re-fetching
    with open(LIVE_INDEX, "w") as f:
        json.dump({r["id"]: {k: r[k] for k in GENERIC_KEYS} for r in live}, f)
    stats.update(_to_score=len(need), _date=TODAY)
    with open(PULL_STATS, "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    print(f"shortlist: {len(need)} lane roles need scoring")


def cmd_build():
    prev = load_prev_payload()
    prev_jobs = {j["id"]: j for j in (prev or {}).get("jobs", [])}
    live = json.load(open(LIVE_INDEX))
    ratings = {}
    if os.path.exists(RATINGS):
        ratings = {r["id"]: r for r in json.load(open(RATINGS))["ratings"]}

    jobs = []
    for rid, meta in live.items():
        rec = prev_jobs.get(rid, {})
        rated = ratings.get(rid)
        if rated:
            rec = {**rec, **{k: rated[k] for k in rated if k != "id"}}
        first_seen = rec.get("first_seen") or TODAY
        rec.update(id=rid, first_seen=first_seen, last_seen=TODAY, closed=False, **meta)
        rec.pop("closed_on", None)  # back on the board — stale close date resets
        if rec.get("scores"):
            rec["fit"] = fit_score(rec["scores"])
            rec["hidden"] = (
                rec.get("group", "anthropic") != "anthropic"
                and rec["fit"] < MIN_FIT_NON_ANTHROPIC
            )
        jobs.append(rec)

    # tombstones: scored roles that vanished from their board (kept for the
    # personal status pipeline; unscored vanished roles simply drop)
    stats = json.load(open(PULL_STATS)) if os.path.exists(PULL_STATS) else {}
    failed = failed_companies(stats)
    cutoff = (date.today() - timedelta(days=TOMBSTONE_DAYS)).isoformat()
    for rid, rec in prev_jobs.items():
        if rid in live:
            continue
        if rec.get("company") in failed:
            jobs.append(rec)  # board unreachable this run — keep state as-is
            continue
        if not rec.get("scores"):
            continue  # genuinely vanished + unscored: drop from the board
        rec["closed"] = True
        rec.setdefault("closed_on", rec.get("last_seen", TODAY))
        if rec["closed_on"] >= cutoff:
            jobs.append(rec)

    finish_payload(jobs, day1=prev is None)


def finish_payload(jobs, day1=False, swept=None):
    """Rank scored roles, order the list, assert sanity, write data.js."""
    scored = [j for j in jobs if j.get("scores")]
    unscored = [j for j in jobs if not j.get("scores")]
    scored.sort(key=lambda j: (-j["fit"], -(j.get("prob") or 0)))
    unscored.sort(key=lambda j: (j.get("published") or "", j["id"]), reverse=True)
    for i, j in enumerate(scored):
        j["rank"] = i + 1
    jobs = scored + unscored

    n_shown = sum(1 for j in scored if not j.get("hidden") and not j.get("closed"))
    assert n_shown >= 15, f"only {n_shown} scored roles visible — refusing to publish"
    assert len(jobs) >= 800, f"only {len(jobs)} total roles — a board is missing"

    payload = {"meta": payload_meta(jobs, day1=day1, swept=swept), "jobs": jobs}
    # ship the prep bank inside the payload so the page can show STAR gists /
    # hard-Q rebuttals on drill cards
    if os.path.exists(PREP_BANK):
        payload["bank"] = json.load(open(PREP_BANK))
    write_payload(payload)


def cmd_sweep():
    """No LLM: refresh open/closed on known roles, append newly-posted ones."""
    prev = load_prev_payload()
    assert prev, "sweep needs an existing data.js"
    live, stats = pull_boards()
    failed = failed_companies(stats)
    live_by_id = {r["id"]: r for r in live}
    known = {j["id"] for j in prev["jobs"]}
    changed = 0
    for j in prev["jobs"]:
        if j.get("company") in failed:
            continue  # board unreachable this run — do not touch these roles
        was_closed = j.get("closed", False)
        if j["id"] in live_by_id:
            j["closed"] = False
            j.pop("closed_on", None)
            j["last_seen"] = TODAY
            for k in GENERIC_KEYS:
                j[k] = live_by_id[j["id"]][k]
        else:
            j["closed"] = True
            j.setdefault("closed_on", j.get("last_seen", TODAY))
        changed += was_closed != j["closed"]
    fresh_ids = [rid for rid in live_by_id if rid not in known]
    fresh = [
        {
            **{k: live_by_id[rid][k] for k in GENERIC_KEYS},
            "id": rid,
            "first_seen": TODAY,
            "last_seen": TODAY,
            "closed": False,
        }
        for rid in fresh_ids
    ]
    print(f"sweep: {changed} open/closed changes, {len(fresh)} new roles appended")
    finish_payload(prev["jobs"] + fresh, swept=TODAY)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pull", action="store_true")
    g.add_argument("--build", action="store_true")
    g.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.pull:
        cmd_pull()
    elif a.build:
        cmd_build()
    else:
        cmd_sweep()
