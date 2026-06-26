# /// script
# requires-python = ">=3.10"
# ///
"""BFS sanity check for the page's shortest-path finder, run against the built data.

Mirrors shortestPath() in assets/js/coauthorship-network.js: BFS over genuine shared-paper
edges — the drawn links PLUS the path_links bridge-route edges (also real co-authorships,
pre-computed for isolated people), never ghost anchor links. Asserts symmetry, edge-validity
of every step, direct coauthors at 1 hop, that isolated people with a bridge route are
reachable (consistent with clicking them), and that no-paper people are honestly unreachable.

Run: uv run test_shortest_path_data.py
"""
import json
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "assets/data/coauthorship.json"

data = json.loads(DATA.read_text())
nodes = data["nodes"]
links = data["links"] + data.get("path_links", [])
adj: dict[str, set[str]] = {n["id"]: set() for n in nodes}
for l in links:
    adj[l["source"]].add(l["target"])
    adj[l["target"]].add(l["source"])
edge_set = {frozenset((l["source"], l["target"])) for l in links}


def bfs(a: str, b: str) -> list[str] | None:
    if a == b:
        return None
    parent: dict[str, str | None] = {a: None}
    q = deque([a])
    while q and b not in parent:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    if b not in parent:
        return None
    ids = []
    cur: str | None = b
    while cur is not None:
        ids.append(cur)
        cur = parent[cur]
    return ids[::-1]


listed = [n for n in nodes if n.get("is_list")]
assert listed, "no listed people?"

# direct coauthors are 1 hop
l0 = links[0]
p = bfs(l0["source"], l0["target"])
assert p is not None and len(p) - 1 == 1, "direct coauthor must be 1 hop"

connected = disconnected = 0
max_len, max_pair = 0, None
for i in range(len(listed)):
    for j in range(i + 1, len(listed)):
        a, b = listed[i]["id"], listed[j]["id"]
        ab, ba = bfs(a, b), bfs(b, a)
        la = None if ab is None else len(ab) - 1
        lb = None if ba is None else len(ba) - 1
        assert la == lb, f"asymmetric: {a} <> {b}: {la} vs {lb}"
        if ab is None:
            disconnected += 1
            continue
        connected += 1
        for k in range(len(ab) - 1):  # every step must be a real edge
            assert frozenset((ab[k], ab[k + 1])) in edge_set, f"non-edge step in {a}->{b}"
        if la > max_len:
            max_len, max_pair = la, (a, b)

# an isolated person WITH a bridge route must reach its target in <= the route's length
# (<=, not ==: the combined graph may shortcut through another person's bridge edges)
paths = data.get("paths", {})
hub = max(listed, key=lambda n: n["degree"])
for n in listed:
    if n["degree"] or n["id"] not in paths:
        continue
    p = paths[n["id"]]
    got = bfs(n["id"], p["target"])
    assert got is not None, f"{n['id']} has a bridge route but pair BFS finds no path"
    assert len(got) - 1 <= p["len"], f"{n['id']}: BFS {len(got) - 1} hops > bridge {p['len']}"

# a no-papers / no-route person must be honestly unreachable
iso = next((n for n in listed if n["degree"] == 0 and n["id"] not in paths), None)
if iso:
    assert bfs(iso["id"], hub["id"]) is None, f"{iso['id']} must be disconnected"

print(f"OK: {len(listed)} listed people · {connected} connected pairs · {disconnected} disconnected pairs")
print(f"max distance among listed pairs: {max_len} hops ({max_pair[0]} <-> {max_pair[1]})")
