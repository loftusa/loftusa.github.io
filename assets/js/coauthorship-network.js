/* Co-authorship network for /coauthorship/
 * Listed researchers are anchors; the "Reach" slider reveals the outside people who connect
 * them (shortest co-authorship paths up to k hops, pre-computed in coauthorship.json).
 * Nodes are avatars: a photo if assets/images/coauthors/<id>.jpg exists (node.photo), else a
 * monogram on the community colour. Positions are pre-computed and held fixed. */
(function () {
  "use strict";

  const PALETTE = ["#4c6b8a", "#a6611a", "#5a7d5a", "#8a6d9b", "#b08968", "#9b6a6a"];
  const OTHER = "#b3a98f";                       // has papers, but no co-authorship link shown
  const color = (c) => (c < 0 ? OTHER : PALETTE[c % PALETTE.length]);
  // people with no indexed papers at all: drawn as open (hollow) circles — absence of publications
  // reads as absence of fill — visually distinct from the solid OTHER above.
  const NOPAPER_FILL = "#faf8f3", NOPAPER_STROKE = "#cdc4b0", NOPAPER_TEXT = "#a59d8b";
  const fillFor = (d) => (d.no_papers ? NOPAPER_FILL : color(d.community));
  // provenance: each node/edge is attested by Semantic Scholar, OpenAlex, or both (cross-referenced)
  const SRC = { both: "Semantic Scholar + OpenAlex", s2: "Semantic Scholar only", oa: "OpenAlex only" };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  // a node's papers list, built once and cached (hover fires repeatedly; the list never changes).
  // Papers arrive most-cited first from the build; show the citation count (and year) for each.
  const PAPER_CAP = 14;
  function papersHtml(d) {
    if (d._ph !== undefined) return d._ph;
    const ps = d.papers || [];
    if (!ps.length) return (d._ph = "");
    const rows = ps.slice(0, PAPER_CAP).map((p) => {
      const meta = [p.year, p.cites ? `${p.cites} cite${p.cites === 1 ? "" : "s"}` : ""]
        .filter(Boolean).join(" · ");
      return `<li>${esc(p.title)}${meta ? ` <span class="t-yr">${meta}</span>` : ""}</li>`;
    }).join("");
    const more = ps.length > PAPER_CAP ? `<li class="t-more">+${ps.length - PAPER_CAP} more</li>` : "";
    return (d._ph = `<ul class="t-papers">${rows}${more}</ul>`);
  }
  const hopDesc = (h) =>
    h === 1 ? "Everyone listed; lines mark direct co-authorships (a shared paper)."
      : h === 2 ? "…plus one shared co-author bridging two listed people."
        : `…plus chains through ${h - 1} intermediate people.`;

  // ---- corrections backend (see experiments/chat_api.py + experiments/coauthorship/overrides.py) ----
  // network-identity.js (loaded first on every /coauthorship/* page) owns the API base
  const LOCAL = /localhost|127\.0\.0\.1/.test(location.hostname);
  const API_BASE = (window.NetworkIdentity && window.NetworkIdentity.API_BASE) ||
    (LOCAL ? "http://127.0.0.1:8000" : "https://llm-resume-restless-thunder-9259.fly.dev");

  const initialsOf = (label) => {
    const parts = (label || "").split(/\s+/).filter((p) => p && /[a-z]/i.test(p[0]));
    if (!parts.length) return (label || "").slice(0, 2).toUpperCase();
    return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
  };

  // Apply the live correction overlay (merged pending edits) on top of the static JSON, BEFORE the
  // graph is built — so pending crowd corrections are visible to everyone within seconds, not only
  // after the nightly rebuild bakes them in. This is the JS mirror of overrides.apply_overrides();
  // keep the two in sync (same op order + semantics).
  function applyOverlay(data, ov) {
    if (!ov) return;
    const ni = (s) => (s || "").toString().replace(/\s+/g, " ").trim().toLowerCase();   // node id key
    const nt = (s) => (s || "").toString().replace(/\s+/g, " ").trim().toLowerCase();   // title key
    const pair = (a, b) => [ni(a), ni(b)].sort();
    const same = (l, p) => { const q = [ni(l.source), ni(l.target)].sort(); return q[0] === p[0] && q[1] === p[1]; };
    const idIndex = new Map(data.nodes.map((n) => [n.id, n]));
    const find = (id) => idIndex.get(id) || data.nodes.find((n) => ni(n.id) === ni(id));
    data.path_links = data.path_links || []; data.paths = data.paths || {};
    data.communities = data.communities || []; data.unconnected = data.unconnected || [];
    data.unresolved = data.unresolved || [];

    // 1. remove_nodes — drop node + every edge/path touching it
    const drop = new Set((ov.remove_nodes || []).map(ni));
    if (drop.size) {
      const real = new Set(data.nodes.filter((n) => drop.has(ni(n.id))).map((n) => n.id));
      data.nodes = data.nodes.filter((n) => !real.has(n.id));
      data.links = data.links.filter((l) => !real.has(l.source) && !real.has(l.target));
      data.path_links = data.path_links.filter((l) => !real.has(l.source) && !real.has(l.target));
      for (const k of Object.keys(data.paths))
        if (real.has(k) || data.paths[k].path.some((x) => real.has(x))) delete data.paths[k];
      data.unconnected = data.unconnected.filter((nm) => !drop.has(ni(nm)));
      data.unresolved = data.unresolved.filter((nm) => !drop.has(ni(nm)));
    }
    // 2. remove_edges
    const rmE = (ov.remove_edges || []).map((e) => pair(e[0], e[1]));
    if (rmE.length) data.links = data.links.filter((l) => !rmE.some((p) => same(l, p)));
    // 2b. remove_papers — drop a title from the edge + both endpoints' lists
    for (const it of (ov.remove_papers || [])) {
      const p = pair(it.between[0], it.between[1]), tk = nt(it.title);
      for (const l of data.links) if (same(l, p)) l.papers = (l.papers || []).filter((t) => nt(t) !== tk);
      for (const id of it.between) { const n = find(id); if (n) n.papers = (n.papers || []).filter((pp) => nt(pp.title) !== tk); }
    }
    data.links = data.links.filter((l) => l.papers && l.papers.length);   // empty edge is no edge
    // 3. add_papers — find-or-create an edge between two existing nodes
    for (const it of (ov.add_papers || [])) {
      const a = find(it.between[0]), b = find(it.between[1]); if (!(a && b)) continue;
      const p = pair(it.between[0], it.between[1]), title = (it.title || "").replace(/\s+/g, " ").trim();
      let link = data.links.find((l) => same(l, p));
      if (!link) { link = { source: p[0], target: p[1], weight: 0, minhop: 1, sources: "manual", papers: [] }; data.links.push(link); }
      if (!link.papers.some((t) => nt(t) === nt(title))) link.papers.push(title);
      for (const n of [a, b]) { n.papers = n.papers || []; if (!n.papers.some((pp) => nt(pp.title) === nt(title))) n.papers.push({ title, year: it.year ?? null }); }
    }
    // 4. paper_rename — rename a title everywhere it appears
    const ren = {}; for (const [k, v] of Object.entries(ov.paper_rename || {})) ren[nt(k)] = v;
    if (Object.keys(ren).length) {
      for (const n of data.nodes) for (const pp of (n.papers || [])) if (ren[nt(pp.title)] !== undefined) pp.title = ren[nt(pp.title)];
      for (const coll of [data.links, data.path_links]) for (const l of coll) l.papers = (l.papers || []).map((t) => (ren[nt(t)] !== undefined ? ren[nt(t)] : t));
    }
    // 5. node field patches
    for (const [id, label] of Object.entries(ov.node_label || {})) { const n = find(id); if (n) { n.label = label; n.initials = initialsOf(label); } }
    const commIds = new Set(data.communities.map((c) => c.id));
    for (const [id, c] of Object.entries(ov.node_community || {})) { const n = find(id); if (n) { n.community = +c; if (!commIds.has(+c)) { data.communities.push({ id: +c, label: "group " + c }); commIds.add(+c); } } }
    for (const [id, urls] of Object.entries(ov.node_url || {})) { const n = find(id); if (n) { if ("openalex" in urls) n.openalex = urls.openalex; if ("oa_url" in urls) n.oa_url = urls.oa_url; } }
    for (const [id, f] of Object.entries(ov.node_photo || {})) { const n = find(id); if (n) n.photo = String(f).startsWith("/") ? f : "/assets/images/coauthors/" + f; }
    // 6. recompute shared_papers; weights stay as built (fractional 1/n_authors — papers.length
    // would stomp them). Links minted by overrides get an approximate fractional weight
    // (~1/3 per paper, median team size); the nightly rebuild replays overrides exactly.
    for (const l of data.links) if (!l.weight) l.weight = Math.max(0.2, (l.papers || []).length / 3);
    const titles = new Map(data.nodes.map((n) => [n.id, new Set()]));
    for (const l of data.links) for (const e of [l.source, l.target]) if (titles.has(e)) (l.papers || []).forEach((t) => titles.get(e).add(nt(t)));
    for (const n of data.nodes) if (!n.no_papers && !n.path_only) n.shared_papers = titles.get(n.id).size;
  }

  // Pending joiners (affiliation event log, "join the map" form) appear here instantly too —
  // minted as temp no-papers nodes, exactly the shape the nightly bake will create durably.
  // Mirrors the careers map (coauthorship-affiliations.js), which mints from the same overlay.
  function mintJoiners(data, affOv) {
    if (!affOv || !affOv.join) return;
    // node-id normalization: the build folds punctuation runs to single spaces, lowercase
    // (e.g. "Leo McKee-Reid" -> "leo mckee reid"); the overlay's join keys are already in this form
    const nk = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    const have = new Set(data.nodes.map((n) => nk(n.id)));
    // the build lays the no-paper folk on a radius-1.3 rim circle, evenly spaced; seed each
    // temp joiner on that same rim, midway between existing slots, so it joins the row naturally
    const rimN = data.nodes.filter((n) => n.no_papers).length || 1;
    let j = 0;
    for (const [pid, join] of Object.entries(affOv.join)) {
      const id = nk(pid);
      if (!id || have.has(id)) continue;
      have.add(id);
      const name = (join && join.name) ? String(join.name).replace(/\s+/g, " ").trim() : id;
      const ang = (2 * Math.PI / rimN) * (j++ + 0.5);
      data.nodes.push({
        id, label: name, initials: initialsOf(name),
        is_list: true, minhop: 0, shared_papers: 0, degree: 0, community: -1,
        x: 1.3 * Math.cos(ang), y: 1.3 * Math.sin(ang),
        openalex: (join && join.scholar_url) || null, oa_url: null,
        sources: null, photo: null, no_papers: true, papers: [],
      });
      data.unresolved = data.unresolved || [];
      if (!data.unresolved.includes(name)) data.unresolved.push(name);   // footnote stays truthful
    }
  }

  Promise.all([
    d3.json("/assets/data/coauthorship.json"),
    // 4s cap: a hung/slow API must not stall first paint — the graph falls back to the static JSON
    fetch(API_BASE + "/coauthorship/overlay", { signal: AbortSignal.timeout(4000) })
      .then((r) => (r.ok ? r.json() : null)).catch(() => null),
    fetch(API_BASE + "/affiliations/overlay", { signal: AbortSignal.timeout(4000) })
      .then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]).then(([data, overlay, affOverlay]) => {
    mintJoiners(data, affOverlay);   // before applyOverlay, so corrections can patch temp nodes too
    applyOverlay(data, overlay);
    const nodes = data.nodes.map((d) => Object.assign({}, d));
    const links = data.links.map((d) => Object.assign({}, d));
    // fallback-path edges: hidden until you click an isolated person, then they trace the route
    // out to the nearest listed author (computed by the build's bridge crawl).
    const pathLinks = (data.path_links || []).map((d) => Object.assign({}, d, { path_only: true }));
    const allLinks = links.concat(pathLinks);
    const visEdgeNodes = () => {                           // ids of nodes sitting on a currently-drawn edge
      const s = new Set();
      allLinks.forEach((l) => { if (l._vis) { s.add(l.source); s.add(l.target); } });
      return s;
    };
    const paths = data.paths || {};                       // isolated id -> {path:[ids], target, len}
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const KMAX = (data.meta && data.meta.k_max) || 4;

    // adjacency (full graph) + each node's reveal hop (min incident edge hop)
    const adj = new Map(nodes.map((n) => [n.id, new Set()]));
    nodes.forEach((n) => (n.revealHop = Infinity));
    links.forEach((l) => {
      adj.get(l.source).add(l.target);
      adj.get(l.target).add(l.source);
      byId.get(l.source).revealHop = Math.min(byId.get(l.source).revealHop, l.minhop);
      byId.get(l.target).revealHop = Math.min(byId.get(l.target).revealHop, l.minhop);
    });
    // path edges contribute to adjacency (for hover/highlight) but NOT to revealHop — an isolated
    // person stays "isolated" (revealHop Infinity) until you open their route.
    pathLinks.forEach((l) => { adj.get(l.source).add(l.target); adj.get(l.target).add(l.source); });
    const pairKey = (a, b) => (a < b ? a + " " + b : b + " " + a);
    const pathLinkByPair = new Map(pathLinks.map((l) => [pairKey(l.source, l.target), l]));
    function routeFor(id) {                                // {ids:[...], links:[...], target, len} | null
      const p = paths[id];
      if (!p) return null;
      const ls = [];
      for (let i = 0; i + 1 < p.path.length; i++) ls.push(pathLinkByPair.get(pairKey(p.path[i], p.path[i + 1])));
      return { ids: p.path, links: ls.filter(Boolean), target: p.target, len: p.len };
    }
    let activeRoute = null;                                // route currently revealed (or null)

    // pair-finder adjacency: drawn edges PLUS bridge-route edges — both are genuine shared-paper
    // co-authorships (bridge edges are just routes pre-computed for isolated people) — but never
    // the ghost anchor links, which aren't co-authorships at all. Including bridge edges keeps the
    // finder consistent with clicking an isolated person: both trace the same real chain.
    const realAdj = new Map(nodes.map((n) => [n.id, new Set()]));
    allLinks.forEach((l) => { realAdj.get(l.source).add(l.target); realAdj.get(l.target).add(l.source); });
    // links last: where a pair has both a drawn edge and a bridge copy, light the drawn one
    const realLinkByPair = new Map(pathLinks.concat(links).map((l) => [pairKey(l.source, l.target), l]));
    function shortestPath(a, b) {                          // BFS — {ids, links, target, len, pair} | null
      if (a === b) return null;
      const parent = new Map([[a, null]]);
      let frontier = [a];
      while (frontier.length && !parent.has(b)) {
        const next = [];
        for (const id of frontier) for (const nb of realAdj.get(id)) {
          if (parent.has(nb)) continue;
          parent.set(nb, id); next.push(nb);
        }
        frontier = next;
      }
      if (!parent.has(b)) return null;
      const ids = [];
      for (let id = b; id !== null; id = parent.get(id)) ids.push(id);
      ids.reverse();
      const ls = [];
      for (let i = 0; i + 1 < ids.length; i++) ls.push(realLinkByPair.get(pairKey(ids[i], ids[i + 1])));
      return { ids, links: ls, target: b, len: ids.length - 1, pair: true };
    }

    // ---- scales / geometry ----
    const maxDeg = d3.max(nodes, (d) => d.degree) || 1;
    const r = d3.scaleSqrt().domain([0, maxDeg]).range([12, 27]);
    // edge thickness encodes fractional co-authorship weight (each paper counts 1/n_authors —
    // Stella Biderman's suggestion), sqrt-scaled from the lightest tie to the heaviest so a
    // single mega-paper tie stays legible while standing duos read clearly thicker.
    const maxWeight = d3.max(allLinks, (d) => d.weight) || 1;
    const minWeight = d3.min(allLinks, (d) => d.weight) || 1;
    const lw = d3.scaleSqrt().domain([minWeight, maxWeight]).range([0.7, 5]);
    const hubDeg = maxDeg * 0.5;
    const labelled = (d) => d.is_list || d.degree >= hubDeg;
    const SPAN = 1300;
    nodes.forEach((n) => { n.x = n.x * SPAN; n.y = n.y * SPAN; });

    // listed people with no direct list<->list path: woven into the network, not stranded at the rim.
    // A never-drawn "ghost" link ties each to the listed author its bridge route reaches, so the live
    // force layout nestles it right beside that author; the dashed route itself reveals on click.
    const isolated = nodes.filter((n) => n.is_list && n.degree === 0);
    const bridgeTarget = new Map();                        // isolated id -> listed author its route reaches
    isolated.forEach((n) => {
      n.isolated = true;
      const p = paths[n.id];
      if (p) { bridgeTarget.set(n.id, p.target); adj.get(n.id).add(p.target); adj.get(p.target).add(n.id); }
    });
    // The whole roster is shown from hop 1 (the listed people are the point of the page); the slider
    // only reveals the outside people who connect them. A listed person with no *visible* co-author at
    // the current hop would otherwise drift to the rim, so anchor each one to its nearest fellow listed
    // person (BFS over the full adjacency) with a never-drawn ghost link — keeping the roster one tidy
    // cluster. This subsumes the old isolated-only ghost links.
    function nearestListed(start) {
      const seen = new Set([start]); let frontier = [start];
      while (frontier.length) {
        const next = [];
        for (const id of frontier) for (const nb of adj.get(id)) {
          if (seen.has(nb)) continue;
          if (byId.get(nb).is_list) return nb;
          seen.add(nb); next.push(nb);
        }
        frontier = next;
      }
      return null;
    }
    // hub = highest-degree listed person; people with no co-authorship path at all (the no-indexed-
    // papers folk) anchor here so they sit as a halo beside the cluster instead of drifting off-canvas.
    const hub = nodes.filter((n) => n.is_list).reduce((a, b) => (a.degree >= b.degree ? a : b)).id;
    const homeAnchor = new Map(nodes.filter((n) => n.is_list)
      .map((n) => [n.id, nearestListed(n.id) || (n.id !== hub ? hub : null)]));

    // ---- svg scaffold ----
    const root = d3.select("#graph");
    const svg = root.append("svg");
    const g = svg.append("g");
    const defs = svg.append("defs");
    const linkG = g.append("g").attr("stroke", "#9a958c").attr("stroke-linecap", "round");
    const nodeG = g.append("g");
    const labelG = g.append("g");
    const tooltip = d3.select("#tooltip");

    // solid = co-authorship attested by BOTH indices; dashed = a single index only (bridge routes
    // get the long dash). Single-source edges are the ones cross-referencing couldn't corroborate.
    const link = linkG.selectAll("line").data(allLinks).join("line")
      .attr("stroke-width", (d) => lw(d.weight))
      .attr("stroke-dasharray", (d) => (d.path_only ? "4 3" : d.sources && d.sources !== "both" ? "2 3" : null));
    // wide invisible hit lines so thin edges are easy to hover; the slim visible lines stay as-is
    const hit = linkG.selectAll("line.hit").data(allLinks).join("line").attr("class", "hit")
      .attr("stroke", "transparent").attr("stroke-width", 12).style("cursor", "pointer")
      .on("mouseover", edgeHover).on("mousemove", moveTip).on("mouseout", edgeOut)
      .on("click", (e, d) => { if (editMode && !d.path_only) { e.stopPropagation(); openEdgeEditor(d); } });

    // node groups (avatars)
    const node = nodeG.selectAll("g.node").data(nodes).join("g").attr("class", "node")
      .style("cursor", "pointer")
      .on("mouseover", hover).on("mousemove", moveTip).on("mouseout", unhover)
      .on("click", (e, d) => (editMode ? openNodeEditor(d) : selectPerson(d.id)))
      .call(d3.drag().on("start", dragStart).on("drag", dragged).on("end", dragEnd));

    node.append("circle").attr("class", "ava").attr("r", (d) => r(d.degree))
      .attr("fill", fillFor)
      .attr("stroke", (d) => (d.no_papers ? NOPAPER_STROKE : "#faf8f3"))
      .attr("stroke-width", (d) => (d.no_papers ? 1.4 : 1.6))
      .attr("stroke-dasharray", (d) => (d.no_papers ? "3 2.5" : null));

    // optional photo fill (falls back to the monogram if it fails to load)
    node.filter((d) => d.photo).each(function (d) {
      const rad = r(d.degree), cid = "clip-" + d.id.replace(/\W+/g, "-");
      defs.append("clipPath").attr("id", cid).append("circle").attr("r", rad);
      d3.select(this).append("image").attr("href", d.photo)
        .attr("x", -rad).attr("y", -rad).attr("width", rad * 2).attr("height", rad * 2)
        .attr("preserveAspectRatio", "xMidYMid slice").attr("clip-path", `url(#${cid})`)
        .on("error", function () { d3.select(this).remove(); });
    });

    node.append("text").attr("class", "ava-text").text((d) => d.initials)
      .style("font-size", (d) => Math.min(r(d.degree) * 0.85, 13) + "px")
      .style("fill", (d) => (d.no_papers ? NOPAPER_TEXT : null))   // dark monogram on the hollow fill
      .style("display", (d) => (d.photo ? "none" : null));

    const label = labelG.selectAll("text").data(nodes).join("text")
      .attr("class", "label").attr("text-anchor", "middle")
      .attr("dy", (d) => -r(d.degree) - 4).text((d) => d.label);

    // ---- live force layout over the currently-visible subset ----
    // Each hop re-packs only the visible people, so every level is tight and legible;
    // new nodes ease in from their pre-computed positions.
    const linkForce = d3.forceLink([]).id((d) => d.id)
      .distance((l) => 62 + 18 / Math.sqrt(l.weight)).strength(0.45);
    const sim = d3.forceSimulation([])
      .force("link", linkForce)
      .force("charge", d3.forceManyBody().strength(-330).distanceMax(650))
      .force("collide", d3.forceCollide((d) => r(d.degree) + 5).strength(0.9))
      .force("x", d3.forceX(0).strength(0.06))
      .force("y", d3.forceY(0).strength(0.06))
      .on("tick", tick)
      .on("end", () => {
        if (activeRoute && activeRoute.pair) fitToRoute();
        else if (selected) centerOnNode(byId.get(selected));
        else fitVisible();
      });

    function tick() {
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      label.attr("x", (d) => d.x).attr("y", (d) => d.y);
      link.attr("x1", (d) => byId.get(d.source).x).attr("y1", (d) => byId.get(d.source).y)
          .attr("x2", (d) => byId.get(d.target).x).attr("y2", (d) => byId.get(d.target).y);
      hit.attr("x1", (d) => byId.get(d.source).x).attr("y1", (d) => byId.get(d.source).y)
         .attr("x2", (d) => byId.get(d.target).x).attr("y2", (d) => byId.get(d.target).y);
    }

    function restartSim() {
      const prev = new Set(sim.nodes().map((n) => n.id));
      const vn = nodes.filter((d) => d._vis);
      vn.forEach((n, i) => {                               // seed new nodes near where they attach
        if (prev.has(n.id)) return;
        if (n.isolated && bridgeTarget.has(n.id)) {        // start beside the author it bridges to
          const t = byId.get(bridgeTarget.get(n.id));
          n.x = t.x + (i % 7 - 3); n.y = t.y + (i % 5 - 2);
        } else {
          const nb = [...adj.get(n.id)].map((id) => byId.get(id)).filter((m) => prev.has(m.id));
          if (nb.length) { n.x = d3.mean(nb, (m) => m.x) + (i % 7 - 3); n.y = d3.mean(nb, (m) => m.y) + (i % 5 - 2); }
        }
      });
      const vl = allLinks.filter((l) => l._vis)
        .map((l) => ({ source: l.source, target: l.target, weight: l.weight }));
      const openIso = activeRoute ? activeRoute.ids[0] : null;   // its real chain defines its geometry
      const onVisEdge = visEdgeNodes();                     // listed people already held by a drawn edge
      nodes.forEach((n) => {                                // anchor every other visible listed person
        if (!n._vis || !n.is_list || n.id === openIso || onVisEdge.has(n.id)) return;
        const t = homeAnchor.get(n.id);
        if (t && byId.get(t)._vis) vl.push({ source: n.id, target: t, weight: 1 });
      });
      sim.nodes(vn);
      linkForce.links(vl);
      sim.alpha(0.85).restart();
    }

    // ---- zoom / pan ----
    const zoom = d3.zoom().scaleExtent([0.15, 5]).on("zoom", (e) => g.attr("transform", e.transform));
    svg.call(zoom).on("dblclick.zoom", null);
    svg.on("dblclick", () => fitVisible());

    function fitVisible(dur = 650) {
      const vis = nodes.filter((d) => d._vis);
      if (!vis.length) return;
      // Frame the connected network mass, not the stray singletons. Isolated / no-paper people are
      // always shown and drift to the canvas edges via their ghost links; framing them too would
      // shrink the real cluster to a dot (this is what made the default view open too zoomed-out).
      // So fit to nodes that sit on a VISIBLE edge, then drop the few that still stray beyond 3·MAD
      // of the median on either axis (a robust outlier filter). Fall back to all visible if too few.
      const onEdge = visEdgeNodes();
      const connected = vis.filter((d) => onEdge.has(d.id));
      const base = connected.length >= 3 ? connected : vis;
      const robust = (sel) => {
        const med = d3.median(base, sel);
        const mad = d3.median(base, (d) => Math.abs(sel(d) - med)) || 0;
        return { med, lo: med - 3 * mad, hi: med + 3 * mad };
      };
      const rx = robust((d) => d.x), ry = robust((d) => d.y);
      const core = base.filter((d) => d.x >= rx.lo && d.x <= rx.hi && d.y >= ry.lo && d.y <= ry.hi);
      frameNodes(core.length >= 3 ? core : base, 80, 2.4, dur);
    }

    function frameNodes(fit, pad, maxS, dur) {           // zoom/pan so these nodes fill the canvas
      const w = svg.node().clientWidth, h = svg.node().clientHeight;
      const x0 = d3.min(fit, (d) => d.x), x1 = d3.max(fit, (d) => d.x);
      const y0 = d3.min(fit, (d) => d.y), y1 = d3.max(fit, (d) => d.y);
      const s = Math.min(maxS, 0.95 / Math.max((x1 - x0 + pad) / w, (y1 - y0 + pad) / h));
      svg.transition().duration(dur).call(zoom.transform,
        d3.zoomIdentity.translate((w - s * (x0 + x1)) / 2, (h - s * (y0 + y1)) / 2).scale(s));
    }

    // ---- hop reveal ----
    let hop = 1, selected = null;
    const pathPair = { from: null, to: null, armed: null };   // shortest-path finder endpoints
    const slider = d3.select("#hops").attr("max", KMAX);

    // the core 1-hop group = listed people + their direct co-authors; once the graph reaches
    // out to 2+ hops, everyone beyond that core is drawn dimmer so the center stays the focus.
    const isCore = (n) => n.is_list || n.revealHop <= 1;
    const restOpacity = (n) => (!n._vis ? 0 : (hop >= 2 && !isCore(n) ? 0.45 : 1));

    function applyVisibility() {
      nodes.forEach((n) => (n._vis = !!n.is_list));   // whole roster always shown; only connectors reveal by hop
      links.forEach((l) => {
        l._vis = l.minhop <= hop;
        if (l._vis) byId.get(l.source)._vis = byId.get(l.target)._vis = true;
      });
      pathLinks.forEach((l) => (l._vis = false));
      if (activeRoute) {                               // reveal the open route's connectors + edges
        activeRoute.ids.forEach((id) => (byId.get(id)._vis = true));
        activeRoute.links.forEach((l) => (l._vis = true));
      }

      link.style("display", (d) => (d._vis ? null : "none"))
        .style("stroke-opacity", (d) => (d._vis ? (d.path_only ? 0.55 : 0.32) : 0));
      hit.style("display", (d) => (d._vis ? null : "none"));   // only visible edges are hoverable
      node.style("display", (d) => (d._vis ? null : "none"));
      node.filter((d) => d._vis).interrupt().style("opacity", 0)
        .transition().duration(400).style("opacity", restOpacity);
      label.style("display", (d) => (d._vis && labelled(d) ? null : "none"))
        .style("opacity", restOpacity);
      d3.selectAll("#people li").classed("hidden-now", function () {
        const n = byId.get(this.dataset.id);
        return n && !n._vis;
      });
    }

    function setHop(h, refit = true) {
      hop = Math.max(1, Math.min(KMAX, h));
      slider.property("value", hop);
      d3.select("#hop-k").text(hop + (hop === 1 ? " hop" : " hops"));
      d3.select("#hop-desc").text(hopDesc(hop));
      applyVisibility();
      restartSim();
      reapplyFocus();
      if (refit) fitVisible(300);
    }
    slider.on("input", function () {                       // pair paths survive hop changes; routes don't
      selected = null;
      if (!(activeRoute && activeRoute.pair)) { activeRoute = null; renderPathDetail(null); }
      syncList(); setHop(+this.value);
    });

    // ---- hover + focus ----
    function hover(e, d) {
      const near = adj.get(d.id);
      node.style("opacity", (n) => (n._vis && (n.id === d.id || near.has(n.id)) ? 1 : (n._vis ? 0.12 : 0)));
      link.style("stroke-opacity", (l) => (l._vis && (l.source === d.id || l.target === d.id) ? 0.75 : (l._vis ? 0.04 : 0)));
      label.style("opacity", (n) => (n.id === d.id || near.has(n.id) ? 1 : 0.06));
      const sub = d.no_papers
        ? "no indexed papers"
        : d.isolated
          ? (paths[d.id] ? `no direct link — click to trace ${paths[d.id].len} hops to ${byId.get(paths[d.id].target).label}`
            : "no co-authorship path within this set")
          : (d.is_list ? "on the list" : "connecting co-author")
            + (d.shared_papers ? ` · ${d.shared_papers} shared papers here` : "");
      // people with papers: list them (most-cited first) so hovering shows their notable work
      const prov = (d.sources && !d.no_papers) ? `<div class="t-src t-src-${d.sources}">${SRC[d.sources]}</div>` : "";
      tooltip.style("opacity", 1)
        .html(`<div class="t-name">${esc(d.label)}</div><div class="t-sub">${esc(sub)}</div>${prov}` + papersHtml(d));
      moveTip(e);
    }
    function moveTip(e) { tooltip.style("left", (e.clientX + 14) + "px").style("top", (e.clientY + 14) + "px"); }
    function unhover() { tooltip.style("opacity", 0); reapplyFocus(); }

    // ---- edge hover: spotlight the edge + its two people, and list the papers they share ----
    function edgeHover(e, d) {
      const a = d.source, b = d.target;
      node.style("opacity", (n) => (n._vis ? (n.id === a || n.id === b ? 1 : 0.12) : 0));
      link.style("stroke-opacity", (l) => (l === d ? 0.95 : (l._vis ? 0.05 : 0)))
        .style("stroke", (l) => (l === d ? "#5a5346" : null))
        .style("stroke-width", (l) => (l === d ? Math.max(2.6, lw(l.weight) * 1.7) : null));
      label.style("opacity", (n) => (n.id === a || n.id === b ? 1 : 0.08));
      const ps = d.papers || [], n = ps.length || d.n_papers || 1;
      const rows = ps.slice(0, PAPER_CAP).map((t) => `<li>${esc(t)}</li>`).join("");
      const more = ps.length > PAPER_CAP ? `<li class="t-more">+${ps.length - PAPER_CAP} more</li>` : "";
      const list = ps.length ? `<ul class="t-papers">${rows}${more}</ul>` : "";
      const prov = d.sources ? `<div class="t-src t-src-${d.sources}">${SRC[d.sources]}</div>` : "";
      tooltip.style("opacity", 1).html(
        `<div class="t-name">${esc(byId.get(a).label)} ↔ ${esc(byId.get(b).label)}</div>`
        + `<div class="t-sub">${n} shared paper${n === 1 ? "" : "s"}</div>${prov}` + list);
      moveTip(e);
    }
    function edgeOut() {
      tooltip.style("opacity", 0);
      link.style("stroke", null).style("stroke-width", null);   // drop the hover emphasis
      reapplyFocus();
    }

    function reapplyFocus() {
      if (activeRoute) highlightRoute();
      else if (selected) highlight(selected);
      else unhighlight();
    }
    // the pair finder paints its chain in warm ink; everything that isn't a pair route must wipe it
    function clearRouteInk() {
      link.style("stroke", null).style("stroke-width", null);
      node.select("circle.ava").style("stroke", null).style("stroke-width", null);
    }
    function unhighlight() {
      clearRouteInk();
      node.style("opacity", restOpacity);
      link.style("stroke-opacity", (l) => (l._vis ? (l.path_only ? 0.55 : 0.32) : 0));
      label.style("display", (d) => (d._vis && labelled(d) ? null : "none")).style("opacity", restOpacity);
    }
    function highlight(id) {
      clearRouteInk();
      const near = adj.get(id) || new Set();
      // cancel applyVisibility's pending fade-in, which would otherwise tween everything back to 1
      node.interrupt().style("opacity", (n) => (!n._vis ? 0 : (n.id === id || near.has(n.id) ? 1 : 0.12)));
      link.style("stroke-opacity", (l) => (l._vis && (l.source === id || l.target === id) ? 0.8 : (l._vis ? 0.05 : 0)));
      label.style("display", (d) => (!d._vis ? "none" : (d.id === id || near.has(d.id) || labelled(d) ? null : "none")))
        .style("opacity", (d) => (d.id === id || near.has(d.id) ? 1 : 0.5));
    }
    function highlightRoute() {                          // spotlight the open route's chain
      const on = new Set(activeRoute.ids), set = new Set(activeRoute.links);
      // cancel applyVisibility's pending fade-in, which would otherwise tween everything back to 1
      node.interrupt().style("opacity", (n) => (!n._vis ? 0 : (on.has(n.id) ? 1 : 0.12)));
      link.style("stroke-opacity", (l) => (set.has(l) ? 0.9 : (l._vis ? 0.05 : 0)));
      if (activeRoute.pair) {                            // pair path: warm ink + heavier endpoint rings
        const ends = new Set([activeRoute.ids[0], activeRoute.target]);
        link.style("stroke", (l) => (set.has(l) ? "#5a5346" : null))
          .style("stroke-width", (l) => (set.has(l) ? Math.max(2.6, lw(l.weight) * 1.5) : null));
        node.select("circle.ava")
          .style("stroke", (n) => (ends.has(n.id) ? "#5a5346" : null))
          .style("stroke-width", (n) => (ends.has(n.id) ? 3 : null));
      } else clearRouteInk();
      label.style("display", (d) => (!d._vis ? "none" : (on.has(d.id) || labelled(d) ? null : "none")))
        .style("opacity", (d) => (on.has(d.id) ? 1 : 0.4));
    }

    function selectPerson(id) {
      const n = byId.get(id);
      if (!n) return;
      if (pathPair.armed) { setPathEnd(id); return; }    // an armed slot captures the click
      resetPairSlots();                                  // normal selection exits path mode
      const wasRoute = !!activeRoute;
      if (selected === id) {                             // toggle off
        selected = null; activeRoute = null; syncList(); renderPathDetail(null);
        if (wasRoute) { applyVisibility(); restartSim(); }
        unhighlight();
        return;
      }
      selected = id;
      activeRoute = n.isolated ? routeFor(id) : null;
      syncList();
      if (n.isolated) {                                  // isolated person: open (or report) their route
        renderPathDetail(id);
        applyVisibility();
        restartSim();
        reapplyFocus();
        centerOnNode(n);
      } else {
        renderPathDetail(null);
        if (wasRoute) { applyVisibility(); restartSim(); }   // tear down any open route
        if (isFinite(n.revealHop) && n.revealHop > hop) setHop(n.revealHop, false);
        highlight(id);
        centerOnNode(n);
      }
    }

    function centerOnNode(n, dur = 550) {
      if (!n) return;
      const w = svg.node().clientWidth, h = svg.node().clientHeight, s = 1.5;
      svg.transition().duration(dur).call(zoom.transform,
        d3.zoomIdentity.translate(w / 2 - s * n.x, h / 2 - s * n.y).scale(s));
    }

    // ---- drag ----
    function dragStart(e, d) { if (!e.active) sim.alphaTarget(0.15).restart(); d.fx = d.x; d.fy = d.y; }
    function dragged(e, d) { d.fx = e.x; d.fy = e.y; }
    function dragEnd(e, d) { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }

    // ---- people list (sidebar) ----
    const listPeople = nodes.filter((d) => d.is_list)
      .sort((a, b) => d3.ascending(a.label, b.label));
    const clickable = (d) => isFinite(d.revealHop) || !!paths[d.id];   // in-network, or has a bridge route
    const ul = d3.select("#people");
    const li = ul.selectAll("li").data(listPeople).join("li")
      .attr("data-id", (d) => d.id)
      .classed("off", (d) => !clickable(d))
      .on("click", (e, d) => { if (clickable(d)) selectPerson(d.id); });
    li.append("span").attr("class", "dot")
      .style("background", (d) => (d.no_papers ? NOPAPER_FILL : color(d.community)))
      .style("color", (d) => (d.no_papers ? NOPAPER_TEXT : null))
      .style("border", (d) => (d.no_papers ? `1px dashed ${NOPAPER_STROKE}` : null))
      .text((d) => d.initials);
    li.append("span").attr("class", "name").text((d) => d.label);
    li.append("span").attr("class", "pct")
      .text((d) => (isFinite(d.revealHop) ? (d.revealHop === 1 ? "direct" : d.revealHop + " hops")
        : (paths[d.id] ? "↗ " + paths[d.id].len + " hops" : "—")));

    function syncList() { li.classed("sel", (d) => d.id === selected); }

    // ---- search filters the list ----
    d3.select("#search").on("input", function () {
      const q = this.value.trim().toLowerCase();
      li.style("display", (d) => (!q || d.label.toLowerCase().includes(q) ? null : "none"));
    });

    // ---- shortest-path finder (sidebar panel) ----
    // Two endpoint slots: click a slot to arm it, then click any listed person (graph or list)
    // to fill it. Both filled -> BFS over real co-authorship edges lights the minimum-hop chain.
    const slots = d3.selectAll("#pathctl .pp-slot");
    const slotHtml = (id) => {
      const n = byId.get(id);
      const bg = n.no_papers ? NOPAPER_FILL : color(n.community);
      const fg = n.no_papers ? NOPAPER_TEXT : "#fff";
      return `<span class="dot" style="background:${bg};color:${fg}">${esc(n.initials)}</span>`
        + `<span class="pp-name">${esc(n.label)}</span>`;
    };
    function renderPathPanel() {
      slots.each(function () {
        const end = this.dataset.end, id = pathPair[end];
        const slot = d3.select(this)
          .classed("armed", pathPair.armed === end).classed("filled", !!id);
        if (id) { slot.attr("title", "click to change").html(slotHtml(id)); return; }
        // empty slot: a type-ahead over the roster (typing and clicking a person both work)
        slot.attr("title", null).html('<input class="pp-input" type="text" placeholder="type or click…"'
          + ' autocomplete="off" spellcheck="false"><div class="pp-menu" hidden></div>');
        const inp = slot.select(".pp-input"), menu = slot.select(".pp-menu");
        const matches = (q) => {
          q = q.trim().toLowerCase();
          const other = pathPair[end === "from" ? "to" : "from"];
          return q ? listPeople.filter((d) => d.label.toLowerCase().includes(q) && d.id !== other).slice(0, 8) : [];
        };
        const showMenu = () => {
          const m = matches(inp.property("value"));
          menu.attr("hidden", m.length ? null : true).html("");
          // mousedown (not click): it fires before the input's blur would hide the menu
          m.forEach((d) => menu.append("div").attr("class", "pp-opt").html(slotHtml(d.id))
            .on("mousedown", (e) => { e.preventDefault(); pathPair.armed = end; setPathEnd(d.id); }));
        };
        inp.on("focus", () => {
          if (pathPair.armed !== end) { pathPair.armed = end; slots.classed("armed", function () { return this.dataset.end === end; }); }
        })
          .on("input", showMenu)
          .on("blur", () => menu.attr("hidden", true))
          .on("keydown", (e) => {
            if (e.key !== "Enter") return;
            const m = matches(inp.property("value"));
            if (m.length) { pathPair.armed = end; setPathEnd(m[0].id); }
          });
      });
      d3.select("#pathctl .pp-clear")
        .style("visibility", pathPair.from || pathPair.to ? "visible" : "hidden");
    }
    function focusSlot(end) {
      const inp = slots.filter(function () { return this.dataset.end === end; }).select(".pp-input").node();
      if (inp) inp.focus();
    }
    function setPathEnd(id) {
      const n = byId.get(id);
      if (!n || !n.is_list) return;                      // endpoints are core-roster people only
      const other = pathPair.armed === "from" ? "to" : "from";
      if (pathPair[other] === id) return;                // same person twice: stay armed
      pathPair[pathPair.armed] = id;
      pathPair.armed = pathPair[other] ? null : other;   // advance to the empty slot
      renderPathPanel();
      if (pathPair.from && pathPair.to) computePairPath();
      else if (pathPair.armed) focusSlot(pathPair.armed); // ready to type the second name
    }
    function computePairPath() {
      selected = null; syncList();
      activeRoute = shortestPath(pathPair.from, pathPair.to);
      renderPairDetail();
      applyVisibility(); restartSim(); reapplyFocus();
      if (activeRoute) fitToRoute(450);
    }
    function resetPairSlots() {                          // drop slot state without touching the graph
      if (!pathPair.from && !pathPair.to && !pathPair.armed) return;
      pathPair.from = pathPair.to = pathPair.armed = null;
      renderPathPanel();
    }
    function clearPathPair() {
      const hadRoute = activeRoute && activeRoute.pair;
      resetPairSlots();
      renderPathDetail(null);
      if (hadRoute) { activeRoute = null; applyVisibility(); restartSim(); fitVisible(450); }
      unhighlight();
    }
    function fitToRoute(dur = 650) {                     // frame just the lit chain
      if (!activeRoute) return;
      const on = new Set(activeRoute.ids);
      const fit = nodes.filter((d) => on.has(d.id));
      if (fit.length) frameNodes(fit, 140, 2, dur);
    }
    function renderPairDetail() {                        // hop badge + breadcrumb (or "no path")
      const box = d3.select("#path-detail").html("");
      if (!activeRoute) {
        box.append("div").attr("class", "pd-title")
          .text(`${byId.get(pathPair.from).label} → ${byId.get(pathPair.to).label}`);
        box.append("div").attr("class", "pd-none").text("No co-authorship path connects these two.");
        return;
      }
      box.append("div").attr("class", "pd-title")
        .text(activeRoute.len + (activeRoute.len === 1 ? " hop" : " hops"));
      renderChain(box, activeRoute.ids);
    }
    function renderChain(box, ids) {                     // A → x → y → B breadcrumb, ends bold
      const chain = box.append("div").attr("class", "pd-chain");
      ids.forEach((nid, i) => {
        if (i) chain.append("span").attr("class", "pd-arrow").text("→");
        const end = i === 0 || i === ids.length - 1;
        chain.append("span").attr("class", "pd-name" + (end ? " pd-end" : ""))
          .text(byId.get(nid).label);
      });
    }
    slots.on("click", function (e) {
      if (e.target.closest(".pp-input, .pp-menu")) return;    // the type-ahead handles itself
      const end = this.dataset.end;
      if (pathPair[end]) {                                    // re-open a filled slot
        pathPair[end] = null;
        if (activeRoute && activeRoute.pair) {                // its route is now stale — tear down
          activeRoute = null; renderPathDetail(null);
          applyVisibility(); restartSim(); unhighlight();
        }
      }
      pathPair.armed = end;
      renderPathPanel();
      focusSlot(end);
    });
    d3.select("#pathctl .pp-clear").on("click", clearPathPair);
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && pathPair.armed) { pathPair.armed = null; renderPathPanel(); }
    });
    renderPathPanel();

    // ---- route breadcrumb (sidebar) ----
    function renderPathDetail(id) {
      const box = d3.select("#path-detail").html("");
      if (!id) return;
      const n = byId.get(id), p = paths[id];
      if (!p) {                                          // isolated, but no bridge route exists
        box.append("div").attr("class", "pd-title").text(n.label);
        box.append("div").attr("class", "pd-none").text("No co-authorship path to anyone else here.");
        return;
      }
      box.append("div").attr("class", "pd-title")
        .text(`${p.len} hops to ${byId.get(p.target).label}`);
      renderChain(box, p.path);
    }

    // ---- legend (community labels from the build, then the two non-group categories) ----
    const legend = d3.select("#legend");
    function legendRow(swatchStyle, text) {
      const row = legend.append("div").attr("class", "row");
      const sw = row.append("span").attr("class", "swatch");
      Object.entries(swatchStyle).forEach(([k, v]) => sw.style(k, v));
      row.append("span").text(text);
    }
    (data.communities || []).forEach((c) => legendRow({ background: color(c.id) }, c.label + " group"));
    if (nodes.some((n) => n.is_list && !n.no_papers && n.degree === 0))
      legendRow({ background: OTHER }, "has papers, no shared co-author");
    if (nodes.some((n) => n.no_papers))
      legendRow({ background: NOPAPER_FILL, border: `1px dashed ${NOPAPER_STROKE}` }, "no indexed papers");
    // provenance key: solid edges are cross-referenced (in both indices), dashed seen in only one
    const prov = legend.append("div").attr("class", "prov");
    [["", "co-authorship in both indices"], ["dash", "one index only"]].forEach(([cls, txt]) => {
      const row = prov.append("div").attr("class", "row");
      row.append("span").attr("class", "ln " + cls);
      row.append("span").text(txt);
    });

    // ---- footnote ----
    const fn = [];
    if (data.unresolved && data.unresolved.length)
      fn.push(`${data.unresolved.length} have no indexed papers (shown as open circles): ${data.unresolved.join(", ")}.`);
    d3.select("#foot-note").text(" " + fn.join(" "));

    // ---- edit mode: people correct their own corner of the graph (open & instant) -------------
    // Every save POSTs one append-only correction event, then reloads — so the edit round-trips
    // through the real overlay (applyOverlay above) and renders identically to what everyone else
    // sees and what the nightly rebuild will bake into overrides.json. No divergent client state.
    let editMode = false;
    let editor = localStorage.getItem("coauthor_editor") || "";
    const panel = d3.select("#edit-panel");
    const toggle = d3.select("#edit-toggle");

    toggle.on("click", () => {
      editMode = !editMode;
      toggle.classed("on", editMode).text(editMode ? "✓ Editing — click a node or edge" : "✎ Suggest a correction");
      d3.select("#app").classed("editing", editMode);
      panel.attr("hidden", editMode ? null : true);
      if (editMode) showEditHome(); else panel.html("");
    });

    function showEditHome() {
      panel.html("");
      panel.append("div").attr("class", "ep-hint")
        .text("Click any person or any solid edge to correct it. Edits show immediately and are saved.");
      const lab = panel.append("div").attr("class", "ep-sec");
      lab.append("label").text("Your name (optional — credited in the edit log)");
      lab.append("input").attr("type", "text").attr("value", editor).attr("placeholder", "anonymous")
        .on("change", function () { editor = this.value.trim(); localStorage.setItem("coauthor_editor", editor); });
    }

    function field(parent, labelText, value, placeholder) {
      const wrap = parent.append("div");
      wrap.append("label").text(labelText);
      return wrap.append("input").attr("type", "text").attr("value", value || "").attr("placeholder", placeholder || "");
    }

    // POST one correction event, then reload to render it via the overlay. Returns on failure only.
    async function save(type, payload, btn) {
      const msg = panel.select(".ep-msg").empty() ? panel.append("div").attr("class", "ep-msg") : panel.select(".ep-msg");
      msg.attr("class", "ep-msg").text("saving…");
      if (btn) btn.attr("disabled", true);
      try {
        const r = await fetch(API_BASE + "/coauthorship/corrections", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ type, payload, editor: editor || null }),
        });
        if (!r.ok) throw new Error("HTTP " + r.status);
        msg.attr("class", "ep-msg ok").text("✓ saved — updating…");
        setTimeout(() => location.reload(), 500);
      } catch (err) {
        msg.attr("class", "ep-msg err").text("could not save (" + err.message + ") — try again");
        if (btn) btn.attr("disabled", null);
      }
    }

    function openNodeEditor(d) {
      panel.html("");
      panel.append("div").attr("class", "ep-title").text(d.label);

      const nameInput = field(panel, "Display name", d.label);
      const nameBtn = panel.append("button").attr("class", "ep-act").text("Rename person")
        .on("click", () => { const v = nameInput.property("value").trim(); if (v && v !== d.label) save("node_label", { id: d.id, label: v }, nameBtn); });

      // group / community
      const grpWrap = panel.append("div").attr("class", "ep-sec");
      grpWrap.append("label").text("Group");
      const sel = grpWrap.append("select");
      (data.communities || []).forEach((c) => sel.append("option").attr("value", c.id).text(c.label + " group").property("selected", c.id === d.community));
      grpWrap.append("button").attr("class", "ep-act").text("Move to group")
        .on("click", function () { save("node_community", { id: d.id, community: +sel.property("value") }, d3.select(this)); });

      // profile links
      const urlSec = panel.append("div").attr("class", "ep-sec");
      const s2 = field(urlSec, "Semantic Scholar URL", d.openalex);
      const oa = field(urlSec, "OpenAlex URL", d.oa_url);
      urlSec.append("button").attr("class", "ep-act").text("Save links")
        .on("click", function () { save("node_url", { id: d.id, openalex: s2.property("value").trim() || null, oa_url: oa.property("value").trim() || null }, d3.select(this)); });

      // photo filename
      const photoSec = panel.append("div").attr("class", "ep-sec");
      const ph = field(photoSec, "Photo filename (drop the image in assets/images/coauthors/)", "", "name.jpg");
      photoSec.append("button").attr("class", "ep-act").text("Set photo")
        .on("click", function () { const v = ph.property("value").trim(); if (v) save("node_photo", { id: d.id, filename: v }, d3.select(this)); });

      // add a co-authored paper with someone (creates/strengthens an edge)
      addPaperForm(panel.append("div").attr("class", "ep-sec"), d.id, null);

      // opt-out
      const rm = panel.append("div").attr("class", "ep-sec");
      rm.append("button").attr("class", "ep-mini ep-danger").text("Remove me from this graph")
        .on("click", function () { if (confirm(`Remove "${d.label}" and all their links from the graph?`)) save("remove_node", { id: d.id }, d3.select(this)); });

      panel.append("button").attr("class", "ep-mini").style("margin-top", "6px").text("← back").on("click", showEditHome);
    }

    function openEdgeEditor(d) {
      panel.html("");
      const a = byId.get(d.source), b = byId.get(d.target);
      panel.append("div").attr("class", "ep-title").text(`${a.label} ↔ ${b.label}`);
      panel.append("div").attr("class", "ep-hint").text("Shared papers — rename (everywhere) or delete each:");

      const list = panel.append("ul").attr("class", "ep-papers");
      (d.papers || []).forEach((title) => {
        const row = list.append("li");
        const inp = row.append("input").attr("class", "pt").attr("type", "text").attr("value", title);
        row.append("button").attr("class", "ep-mini").text("rename")
          .on("click", function () { const v = inp.property("value").trim(); if (v && v !== title) save("paper_rename", { old: title, new: v }, d3.select(this)); });
        row.append("button").attr("class", "ep-mini ep-danger").text("delete")
          .on("click", function () { save("remove_paper", { between: [d.source, d.target], title }, d3.select(this)); });
      });

      addPaperForm(panel.append("div").attr("class", "ep-sec"), d.source, d.target);

      const del = panel.append("div").attr("class", "ep-sec");
      del.append("button").attr("class", "ep-mini ep-danger").text("Delete this entire edge")
        .on("click", function () { if (confirm("Delete the co-authorship edge between these two people?")) save("remove_edge", { between: [d.source, d.target] }, d3.select(this)); });

      panel.append("button").attr("class", "ep-mini").style("margin-top", "6px").text("← back").on("click", showEditHome);
    }

    // shared "add a paper" form. If `other` is null, offer a person picker (node context); else the
    // pair is fixed (edge context). Creates the edge if the two people don't share one yet.
    function addPaperForm(sec, sourceId, other) {
      sec.append("label").text(other ? "Add another shared paper" : "Add a co-authored paper with…");
      let pick = null;
      if (!other) {
        pick = sec.append("select");
        pick.append("option").attr("value", "").text("— choose a person —");
        nodes.filter((n) => n.id !== sourceId).slice().sort((x, y) => d3.ascending(x.label, y.label))
          .forEach((n) => pick.append("option").attr("value", n.id).text(n.label));
      }
      const title = field(sec, "Paper title", "");
      const year = field(sec, "Year", "", "2025");
      sec.append("button").attr("class", "ep-act").text("Add paper")
        .on("click", function () {
          const t = title.property("value").trim();
          const tgt = other || (pick && pick.property("value"));
          if (!t || !tgt) return;
          const y = parseInt(year.property("value"), 10);
          save("add_paper", { between: [sourceId, tgt], title: t, year: Number.isFinite(y) ? y : null }, d3.select(this));
        });
    }

    // ---- sizing + init ----
    function resize() {
      const w = root.node().clientWidth, h = root.node().clientHeight;
      svg.attr("viewBox", [0, 0, w, h]).attr("width", w).attr("height", h);
    }
    resize();
    window.addEventListener("resize", () => { resize(); fitVisible(0); });
    const params = new URLSearchParams(location.search);
    const startHop = Math.max(1, Math.min(KMAX, +params.get("hop") || 1));
    setHop(startHop, false);
    // Pre-settle the initial layout synchronously so the FIRST paint is already framed. Otherwise the
    // good fit only lands on the sim's "end" event a few seconds in, and the page opens zoomed-out.
    sim.stop();
    for (let i = 0; i < 280; i++) sim.tick();
    tick();
    fitVisible(0);
    const focus = (params.get("focus") || "").toLowerCase();
    if (focus) {
      const m = nodes.find((n) => n.is_list && (n.id.includes(focus) || n.label.toLowerCase().includes(focus)));
      if (m && isFinite(m.revealHop)) selectPerson(m.id);
    }

    /* ---- finder: search anyone near this graph ----
     * This map already draws members AND their outside coauthors, so locals cover both;
     * selectPerson auto-bumps the hop slider to reveal a hidden coauthor. People beyond the
     * drawn graph (extended-graph index, OpenAlex) hand off to the careers map, whose guest
     * engine places them with temporary ties (?guest=<OpenAlex id>). */
    const finderQ = document.getElementById("finder-q");
    const finderRes = document.getElementById("finder-results");
    const GUEST_URL = (oa) => "/networks/affiliations/?guest=" + encodeURIComponent(oa);
    let gidx = null, finderTimer = null, finderSeq = 0;
    const nrm = (s) => String(s).toLowerCase().replace(/\s+/g, " ").trim();
    const finderRow = (html, cls) => {
      const div = document.createElement("div");
      div.className = cls || "fr";
      div.innerHTML = html;
      return div;
    };
    function runFinder(q) {
      const seq = ++finderSeq;
      finderRes.innerHTML = "";
      finderRes.hidden = false;
      const ql = q.toLowerCase();
      const locals = nodes.filter((n) => n.label.toLowerCase().includes(ql)).slice(0, 5);
      const taken = new Set(locals.map((n) => nrm(n.label)));
      for (const n of locals) {
        const row = finderRow(`<span class="fr-name">${esc(n.label)}</span><span class="fr-hint"></span>
          <span class="fr-tag">${n.is_list ? "on the map" : "coauthor — slide to reveal"}</span>`);
        row.addEventListener("mousedown", () => { closeFinder(); selectPerson(n.id); });
        finderRes.appendChild(row);
      }
      Promise.all([
        gidx ? Promise.resolve(gidx) : fetch("/assets/data/guest-index.json")
          .then((r) => (r.ok ? r.json() : { people: [] })).catch(() => ({ people: [] })),
        fetch(`https://api.openalex.org/autocomplete/authors?q=${encodeURIComponent(q)}&mailto=alexloftus2004%40gmail.com`,
              { signal: AbortSignal.timeout(6000) })
          .then((r) => r.json()).catch(() => ({ results: [] })),
      ]).then(([gi, oa]) => {
        if (seq !== finderSeq) return;
        gidx = gi;
        const idxHits = (gi.people || []).filter((it) =>
          it.label.toLowerCase().includes(ql) && !taken.has(nrm(it.label))).slice(0, 4);
        for (const it of idxHits) {
          taken.add(nrm(it.label));
          const row = finderRow(`<span class="fr-name">${esc(it.label)}</span>
            <span class="fr-hint">${it.n} paper${it.n === 1 ? "" : "s"} with the group</span>
            <span class="fr-tag">see them on the careers map →</span>`);
          row.addEventListener("mousedown", () => { location.href = GUEST_URL(it.oa); });
          finderRes.appendChild(row);
        }
        const score = (it) => (it.display_name.toLowerCase() === ql ? 1e9 : 0) + (it.works_count || 0);
        const items = (oa.results || []).slice(0, 8).sort((a, b) => score(b) - score(a))
          .slice(0, 4).filter((it) => !taken.has(nrm(it.display_name)));
        for (const it of items) {
          const row = finderRow(`<span class="fr-name">${esc(it.display_name)}</span>
            <span class="fr-hint">${esc(it.hint || "")}</span>
            <span class="fr-tag">${it.works_count} works · try on careers map →</span>`);
          row.addEventListener("mousedown", () => { location.href = GUEST_URL(it.id.split("/").pop()); });
          finderRes.appendChild(row);
        }
        if (!locals.length && !idxHits.length && !items.length) {
          finderRes.appendChild(finderRow("no public record found — try the careers map's add-a-person form", "fr-note"));
        }
      });
    }
    function closeFinder() {
      finderRes.hidden = true;
      finderRes.innerHTML = "";
      finderQ.value = "";
    }
    finderQ.addEventListener("input", () => {
      clearTimeout(finderTimer);
      const q = finderQ.value.trim();
      if (q.length < 3) { finderRes.hidden = true; return; }
      finderTimer = setTimeout(() => runFinder(q), 250);
    });
    finderQ.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeFinder(); });
    finderQ.addEventListener("blur", () => setTimeout(() => (finderRes.hidden = true), 200));
  });
})();
