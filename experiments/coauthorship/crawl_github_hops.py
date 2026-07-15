# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "click"]
# ///
"""GitHub leg of the affiliation hop layer (ToS-clean public API only; no LinkedIn).

Two steps, both cached under raw/github/ so re-runs are cheap and reviewable:
1. Resolve members -> GitHub handles by name search, accepted only with corroboration
   (profile company/bio mentions one of their mapped orgs, or their blog matches the
   homepage in seeds.json). Written to hop_sources/github_handles.json for hand review.
2. For each mapped company/community room with a real GitHub org (GH_ROOMS), public
   org membership is the co-event: members of the map who are publicly in the org vouch
   for the other public members. Outside people ranked by followers, top-K per room.

    cd experiments/coauthorship && gh auth status   # token via `gh auth token`
    uv run crawl_github_hops.py [--refresh]

Writes hop_sources/github.json in build_hops.py's merge schema; run build_hops.py after.
"""
import json
import re
import subprocess
import time
from pathlib import Path

import click
import httpx

HERE = Path(__file__).resolve().parent
AFF = HERE.parents[1] / "public" / "assets" / "data" / "affiliations.json"
SEEDS = HERE / "seeds.json"
CACHE = HERE / "raw" / "github"
OUT_DIR = HERE / "hop_sources"
HANDLES_OUT = OUT_DIR / "github_handles.json"
HOPS_OUT = OUT_DIR / "github.json"

# our org slug -> GitHub org login. Hand-verified to exist with a public-member list small
# enough that co-membership still means "you'd plausibly know each other" (microsoft at
# ~4500 public members carries no social signal and is deliberately absent).
GH_ROOMS: dict[str, str] = {
    "eleutherai": "EleutherAI",
    "openai": "openai",
    "anthropic": "anthropics",
    "deepmind": "google-deepmind",
    "scale-ai": "scaleapi",
    "nomic-ai": "nomic-ai",
    "nvidia": "NVIDIA",
}
MAX_ROOM = 200           # refuse rooms bigger than this (signal dies with size)
FANOUT_CAP = 10          # outside people kept per (vouching member, room)
REPO_CAP = 12            # most-starred org repos checked for co-contribution
CONTRIB_PAGES = 3        # how deep in a repo's contributor list a co-event can sit (×100)

ni = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def gh_token() -> str:
    return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                          check=True).stdout.strip()


def cached_get(client: httpx.Client, path: str, key: str, refresh: bool, **params):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{key}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    for attempt in range(6):
        r = client.get(f"https://api.github.com/{path}", params=params)
        if r.status_code in (403, 429):     # secondary rate limit (search is 30/min)
            wait = int(r.headers.get("retry-after") or 0) or 15 * (attempt + 1)
            print(f"  rate-limited on {path}, sleeping {wait}s")
            time.sleep(wait)
            continue
        break
    if r.status_code == 404:
        data = None
    else:
        r.raise_for_status()
        data = r.json()
    f.write_text(json.dumps(data, ensure_ascii=False))
    return data


def resolve_handles(client, members: dict[str, set[str]], homepages: dict[str, str],
                    refresh: bool) -> dict[str, dict]:
    """member id -> {login, name, evidence} — only when the profile corroborates."""
    out = {}
    for pid, org_words in sorted(members.items()):
        res = cached_get(client, "search/users", f"search_{ni(pid).replace(' ', '_')}",
                         refresh, q=f"{pid} in:name", per_page=5)
        for item in (res or {}).get("items", []):
            u = cached_get(client, f"users/{item['login']}", f"user_{item['login']}", refresh)
            if not u:
                continue
            hay = ni(f"{u.get('company') or ''} {u.get('bio') or ''}")
            hit = next((w for w in org_words if w and w in hay), None)
            blog = (u.get("blog") or "").lower().rstrip("/")
            home = (homepages.get(pid) or "").lower().rstrip("/")
            same_site = bool(blog and home and (blog in home or home in blog))
            if ni(u.get("name") or "") == ni(pid) and (hit or same_site):
                out[pid] = {"login": u["login"], "name": u.get("name"),
                            "company": u.get("company"), "blog": u.get("blog"),
                            "evidence": f"org match: {hit}" if hit else f"homepage match: {blog}"}
                break
    return out


@click.command()
@click.option("--refresh", is_flag=True, help="bypass the raw/github cache")
def main(refresh: bool) -> None:
    aff = json.loads(AFF.read_text())
    org_ids = {o["id"] for o in aff["orgs"]}
    assert set(GH_ROOMS) <= org_ids, sorted(set(GH_ROOMS) - org_ids)
    label_of = {o["id"]: o["label"] for o in aff["orgs"]}

    # corroboration vocabulary: each member's mapped org labels, plus seed homepages
    member_orgs: dict[str, set[str]] = {p["id"]: set() for p in aff["people"]}
    for l in aff["links"]:
        member_orgs[l["person"]].add(ni(re.sub(r"\(.*?\)", "", label_of[l["org"]])))
    homepages = {ni(e["name"]): e.get("homepage") or "" for e in json.loads(SEEDS.read_text())}

    client = httpx.Client(timeout=30, headers={
        "Authorization": f"Bearer {gh_token()}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "affiliation-hops (alex-loftus.com)",
    })

    handles = resolve_handles(client, member_orgs, homepages, refresh)
    OUT_DIR.mkdir(exist_ok=True)
    HANDLES_OUT.write_text(json.dumps(handles, indent=1, ensure_ascii=False) + "\n")
    print(f"resolved {len(handles)}/{len(member_orgs)} members to GitHub handles "
          f"-> {HANDLES_OUT.name} (hand-reviewable)")
    login2member = {h["login"].lower(): pid for pid, h in handles.items()}

    # (login, org_slug) -> accumulated evidence across both legs; one link per pair ships
    ev_map: dict[tuple[str, str], dict] = {}

    def add_event(login: str, org_slug: str, vouchers: list[str], top: str) -> None:
        ev = ev_map.setdefault((login.lower(), org_slug), {"via": set(), "n": 0, "top": ""})
        ev["via"] |= set(vouchers)
        ev["n"] += 1
        ev["top"] = ev["top"] or top

    for org_slug, gh_org in sorted(GH_ROOMS.items()):
        # leg 1 — co-contribution: member and outsider committed to the same org repo
        repos = []
        for page in range(1, 6):               # up to 500 repos, then the star-sort picks
            batch = cached_get(client, f"orgs/{gh_org}/repos", f"repos_{gh_org}_p{page}",
                               refresh, per_page=100, type="public", page=page) or []
            repos += batch
            if len(batch) < 100:
                break
        repos = sorted((r for r in repos if not r.get("fork") and not r.get("archived")),
                       key=lambda r: (-(r.get("stargazers_count") or 0), r["name"]))[:REPO_CAP]
        n_repo_events = 0
        for repo in repos:
            contribs = []
            for page in range(1, CONTRIB_PAGES + 1):
                batch = cached_get(client, f"repos/{gh_org}/{repo['name']}/contributors",
                                   f"contrib_{gh_org}_{repo['name']}_p{page}", refresh,
                                   per_page=100, page=page)
                contribs += batch or []
                if not batch or len(batch) < 100:
                    break
            contribs = [c for c in contribs
                        if c.get("login") and c.get("type") != "Bot"
                        and not c["login"].endswith("[bot]")]
            vouchers = sorted({login2member[c["login"].lower()] for c in contribs
                               if c["login"].lower() in login2member})
            if not vouchers:
                continue
            outside = sorted((c for c in contribs if c["login"].lower() not in login2member),
                             key=lambda c: (-(c.get("contributions") or 0), c["login"].lower()))
            for c in outside[:FANOUT_CAP]:
                add_event(c["login"], org_slug, vouchers,
                          f"co-contributor on github.com/{gh_org}/{repo['name']} "
                          f"({c['contributions']} commits)")
                n_repo_events += 1

        # leg 2 — public org membership
        roster, page = [], 1
        while True:
            batch = cached_get(client, f"orgs/{gh_org}/public_members",
                               f"org_{gh_org}_p{page}", refresh, per_page=100, page=page)
            roster += batch or []
            if not batch or len(batch) < 100 or len(roster) > MAX_ROOM:
                break
            page += 1
        m_vouchers = sorted({login2member[m["login"].lower()] for m in roster
                             if m["login"].lower() in login2member})
        if 0 < len(roster) <= MAX_ROOM and m_vouchers:
            for m in roster:
                if m["login"].lower() not in login2member:
                    add_event(m["login"], org_slug, m_vouchers,
                              f"public member of github.com/{gh_org} ({m['login']})")
        room = sorted({lg for (lg, o) in ev_map if o == org_slug})
        print(f"  {gh_org}: {n_repo_events} co-contribution events "
              f"({len(repos)} repos), membership {'ok' if m_vouchers else 'unvouched'} "
              f"({len(roster)} public) -> {len(room)} outside candidates")

    # per (member, room) fan-out cap, by co-event count — mirrors build_hops's openalex cap
    keep: set[tuple[str, str]] = set()
    by_member_org: dict[tuple[str, str], list] = {}
    for (lg, org_slug), ev in ev_map.items():
        for m in ev["via"]:
            by_member_org.setdefault((m, org_slug), []).append((-ev["n"], lg))
    for (m, org_slug), cands in by_member_org.items():
        for _, lg in sorted(cands)[:FANOUT_CAP]:
            keep.add((lg, org_slug))

    people: dict[str, dict] = {}
    links: list[dict] = []
    for (lg, org_slug) in sorted(keep):
        ev = ev_map[(lg, org_slug)]
        u = cached_get(client, f"users/{lg}", f"user_{lg}", refresh)
        if not u:
            continue
        label = u.get("name") or u["login"]
        if not ni(label):                      # junk profile names ("?", "...") -> the login
            label = u["login"]
        if ni(label) in member_orgs:           # member under an unresolved alt account
            continue
        hid = "h:gh:" + lg
        people.setdefault(hid, {
            "id": hid, "label": label,
            "initials": "".join(w[0] for w in label.split()[:2]).upper() or "?",
        })
        links.append({
            "person": hid, "org": org_slug, "via": sorted(ev["via"]), "n": ev["n"],
            "years": "", "top": ev["top"], "src": "github",
        })

    out = {"people": sorted(people.values(), key=lambda p: p["id"]),
           "links": sorted(links, key=lambda l: (l["person"], l["org"]))}
    HOPS_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {HOPS_OUT.relative_to(HERE)}: {len(out['people'])} outside people, "
          f"{len(out['links'])} links — now run build_hops.py")


if __name__ == "__main__":
    main()
