/* Affiliation graph for /coauthorship/affiliations/
 * Bipartite view: people (monogram avatars, community colours shared with the map page) tied
 * to the orgs they've belonged to. "People only" view: the projection — ties between people
 * who share an org. Data pre-built by experiments/coauthorship/build_affiliations.py.
 *
 * Self-service: the page fetches the live event overlay (GET /affiliations/overlay) and merges
 * it before first render; the "✎ edit my row / join the map" panel POSTs events and applies
 * them optimistically (applyAffOverlay + recomputeDerived + render — no reload, so the person
 * watches their node re-wire). The nightly merge bakes events durably (merge_affiliations.py).
 * Keep applyAffOverlay's semantics in sync with affiliation_events.apply_aff_overlay. */
(function () {
  "use strict";

  // network-identity.js (loaded first on every /coauthorship/* page) owns the API base
  const API_BASE = (window.NetworkIdentity && window.NetworkIdentity.API_BASE) ||
    (location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://127.0.0.1:8000" : "https://llm-resume-restless-thunder-9259.fly.dev");

  // community colours: keep identical to coauthorship-network.js PALETTE
  const PALETTE = ["#4c6b8a", "#a6611a", "#5a7d5a", "#8a6d9b", "#b08968", "#9b6a6a"];
  const OTHER = "#b3a98f";                 // community -1 (no papers yet / just joined)
  const TYPE = {
    lab:        { color: "#8c510a", one: "lab", many: "labs" },
    program:    { color: "#2166ac", one: "program", many: "programs" },
    company:    { color: "#7b3294", one: "company", many: "companies" },
    community:  { color: "#35978f", one: "community", many: "communities" },
    university: { color: "#98917f", one: "university", many: "universities" },
  };
  const MODE_DESC = {
    bipartite: "People tied to the places they've been. Org size = how many people were there.",
    people: "Direct ties: two people who share an org. Line width = how much history they share.",
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const ni = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim(); // person ids (build norm)
  const orgKey = (s) => String(s).split(/\s+/).join(" ").toLowerCase();         // org fold key
  const slugJS = (s) => s.normalize("NFKD").replace(/[^\x00-\x7f]/g, "").toLowerCase()
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");   // mirrors build slug() (ascii-drop)
  // keep in sync with coauthorship-network.js initialsOf (the nightly bakes initials from there)
  const initialsOf = (s) => {
    const w = s.split(/\s+/).filter((t) => /^[a-z]/i.test(t));
    return ((w[0] || s).slice(0, 1) + (w.length > 1 ? w[w.length - 1].slice(0, 1) : "")).toUpperCase();
  };
  const colorOf = (p) => p.community >= 0 ? PALETTE[p.community % PALETTE.length] : OTHER;

  const graphEl = document.getElementById("graph");
  const tooltip = document.getElementById("tooltip");

  Promise.all([
    fetch("/assets/data/affiliations.json").then((r) => r.json()),
    fetch(API_BASE + "/affiliations/overlay", { signal: AbortSignal.timeout(4000) })
      .then((r) => (r.ok ? r.json() : null)).catch(() => null),
    fetch("/assets/data/affiliations-hops.json").then((r) => (r.ok ? r.json() : null))
      .catch(() => null),
    fetch("/assets/data/guest-index.json").then((r) => (r.ok ? r.json() : null))
      .catch(() => null),
  ]).then(([data, overlay, hops, gidx]) => {
    if (overlay) applyAffOverlay(data, overlay);
    init(data, hops || { people: [], links: [] }, (gidx && gidx.people) || []);
  });

  /* ---------- overlay merge (JS mirror of affiliation_events.apply_aff_overlay) ---------- */
  function applyAffOverlay(data, ov) {
    const people = new Map(data.people.map((p) => [p.id, p]));
    for (const pid of Object.keys(ov.join || {})) {
      if (people.has(pid)) continue;
      const j = ov.join[pid];
      const p = { id: pid, label: j.name, initials: initialsOf(j.name), community: -1,
                  city: j.city || "", kind: "person" };
      data.people.push(p);
      people.set(pid, p);
    }
    for (const pid of Object.keys(ov.entry_remove || {})) {
      const gone = new Set(ov.entry_remove[pid]);
      data.links = data.links.filter((l) => {
        const org = data.orgs.find((o) => o.id === l.org);
        return !(l.person === pid && org && gone.has(orgKey(org.label)));
      });
    }
    for (const pid of Object.keys(ov.entry_set || {})) {
      if (!people.has(pid)) continue;               // edits orphaned by a join revert
      for (const okey of Object.keys(ov.entry_set[pid])) {
        const spec = ov.entry_set[pid][okey];
        let org = data.orgs.find((o) => orgKey(o.label) === okey);
        if (!org) {                                 // new raw org node until the nightly CANON pass
          org = { id: slugJS(spec.org), label: spec.org, type: spec.type, n_members: 0, kind: "org" };
          data.orgs.push(org);
        }
        const link = data.links.find((l) => l.person === pid && l.org === org.id);
        const fields = { role: spec.role || "", years: spec.years || "",
                         source: spec.source || "self-reported" };
        if (link) Object.assign(link, fields);
        else data.links.push({ person: pid, org: org.id, ...fields });
      }
    }
    for (const pid of Object.keys(ov.city || {})) {
      if (people.has(pid)) people.get(pid).city = ov.city[pid];
    }
    recomputeDerived(data);
  }

  // re-derive org membership + the nesting-discounted projection (mirror of the build)
  function recomputeDerived(data) {
    const members = new Map();
    for (const l of data.links) {
      if (!members.has(l.org)) members.set(l.org, []);
      members.get(l.org).push(l.person);
    }
    data.orgs = data.orgs.filter((o) => (members.get(o.id) || []).length > 0);
    for (const o of data.orgs) o.n_members = members.get(o.id).length;

    const parent = data.meta.parent || {};
    const tw = data.meta.type_weights;
    const typeOf = new Map(data.orgs.map((o) => [o.id, o.type]));
    const pairShared = new Map();
    for (const o of data.orgs) {
      const m = members.get(o.id).slice().sort();
      for (let i = 0; i < m.length; i++) for (let j = i + 1; j < m.length; j++) {
        const k = m[i] + "|" + m[j];
        if (!pairShared.has(k)) pairShared.set(k, []);
        pairShared.get(k).push(o.id);
      }
    }
    data.projection = [...pairShared.keys()].sort().map((k) => {
      const [a, b] = k.split("|");
      const shared = pairShared.get(k);
      const implied = new Set(shared.map((o) => parent[o]).filter(Boolean));
      const kept = shared.filter((o) => !implied.has(o)).sort();
      return { a, b, weight: Math.round(kept.reduce((s, o) => s + tw[typeOf.get(o)], 0) * 10) / 10,
               shared: kept };
    });
  }

  function init(data, hops, gidx) {
    const state = {
      mode: "bipartite",
      types: new Set(Object.keys(TYPE)),
      singles: false,
      selected: null,                       // {kind: "person"|"org"|"hop", id}
      anchor: null,                         // person id whose reach anchors the hop reveal
      // one "Reach" slider, three meanings — each context keeps its own value:
      orgMin: 1,                            // people & orgs: only orgs with ≥ N members
      tieMin: 1,                            // people only: only ties of weight ≥ W
      hops: 1,                              // person selected: within K steps of them
    };
    data.people.forEach((p) => (p.kind = "person"));
    data.orgs.forEach((o) => (o.kind = "org"));
    // off-map hop layer (build_hops.py): people outside the map who verifiably shared a
    // room with a member — revealed by the reach slider when a person is selected
    hops.people.forEach((p) => (p.kind = "hop"));
    let hopById = new Map(hops.people.map((p) => [p.id, p]));
    let hopLinksByPerson = d3.group(hops.links, (l) => l.person);
    let shownHops = new Set();              // hop ids in the current render

    // someone joining the map stops being a hop person (their hollow node would shadow
    // the real one until the hourly bake; build_hops applies the same rule server-side)
    function dropHop(name) {
      const k = ni(name);
      hops.people = hops.people.filter((p) => ni(p.label) !== k);
      const ids = new Set(hops.people.map((p) => p.id));
      hops.links = hops.links.filter((l) => ids.has(l.person));
      hopById = new Map(hops.people.map((p) => [p.id, p]));
      hopLinksByPerson = d3.group(hops.links, (l) => l.person);
    }

    // ---------- mutable indexes (rebuilt after every optimistic apply) ----------
    let personById, orgById, linksByPerson, linksByOrg, projAdj, orgFont, projW;
    function reindex() {
      personById = new Map(data.people.map((p) => [p.id, p]));
      orgById = new Map(data.orgs.map((o) => [o.id, o]));
      linksByPerson = d3.group(data.links, (l) => l.person);
      linksByOrg = d3.group(data.links, (l) => l.org);
      projAdj = new Map(data.people.map((p) => [p.id, []]));
      for (const e of data.projection) {
        if (projAdj.has(e.a) && projAdj.has(e.b)) {
          projAdj.get(e.a).push(e.b);
          projAdj.get(e.b).push(e.a);
        }
      }
      orgFont = d3.scaleSqrt().domain([1, d3.max(data.orgs, (o) => o.n_members)]).range([10, 19]);
      projW = d3.scaleLinear().domain([1, d3.max(data.projection, (p) => p.weight) || 1])
        .range([0.8, 6]);
    }
    reindex();

    /* ---------- svg scaffold ---------- */
    const svg = d3.select(graphEl).append("svg");
    const root = svg.append("g");
    const linkG = root.append("g");
    const orgG = root.append("g");
    const peopleG = root.append("g");
    svg.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (ev) => root.attr("transform", ev.transform)))
       .on("dblclick.zoom", null);
    svg.on("click", () => select(null));

    const W = () => graphEl.clientWidth, H = () => graphEl.clientHeight;
    const sim = d3.forceSimulation()
      .force("charge", d3.forceManyBody().strength(-220))
      .force("x", d3.forceX(() => W() / 2).strength(0.05))
      .force("y", d3.forceY(() => H() / 2).strength(0.06))
      .on("tick", tick);

    /* ---------- render pass ---------- */
    let linkSel, orgSel, peopleSel;
    const orgVisible = (o) => state.types.has(o.type) && (state.singles || o.n_members >= 2)
      && o.n_members >= state.orgMin;

    // a hop person shows when a member who shared the room (and a co-event) with them is
    // strictly inside the reach (dist < hops) and the room itself passes the filters
    function revealedHopLinks(nb) {
      if (!nb || !nb.dist) return [];
      return hops.links.filter((l) => {
        const o = orgById.get(l.org);
        return o && orgVisible(o) &&
          l.via.some((v) => (nb.dist.get(v) ?? Infinity) < state.hops);
      });
    }

    function render() {
      const vOrgs = data.orgs.filter(orgVisible);
      const vOrgIds = new Set(vOrgs.map((o) => o.id));
      let nodes, links;

      if (state.mode === "bipartite") {
        links = data.links.filter((l) => vOrgIds.has(l.org))
          .map((l) => ({ ...l, source: personById.get(l.person), target: orgById.get(l.org) }));
        nodes = [...data.people, ...vOrgs];
      } else {
        links = data.projection
          .map((p) => ({ ...p, shown: p.shared.filter((o) => state.types.has(orgById.get(o).type)) }))
          .filter((p) => p.shown.length)
          .map((p) => ({ ...p, w: d3.sum(p.shown, (o) => data.meta.type_weights[orgById.get(o).type]),
                         source: personById.get(p.a), target: personById.get(p.b) }))
          .filter((p) => p.w >= state.tieMin);
        nodes = data.people;
      }

      // off-map reveal: hollow nodes for the people the reach can vouch for.
      // anchored to the last selected PERSON, so clicking a hop node keeps the reveal.
      const nb = neighborhood(state.anchor ? { kind: "person", id: state.anchor } : null);
      const hopLinks = revealedHopLinks(nb);
      shownHops = new Set(hopLinks.map((l) => l.person));
      if (hopLinks.length) {
        nodes = nodes.concat(hops.people.filter((p) => shownHops.has(p.id)));
        if (state.mode === "bipartite") {
          links = links.concat(hopLinks.map((l) =>
            ({ ...l, hop: true, source: hopById.get(l.person), target: orgById.get(l.org) })));
        } else {
          for (const l of hopLinks) {                 // tie them to the members who vouch
            for (const v of l.via) {
              if ((nb.dist.get(v) ?? Infinity) < state.hops) {
                links.push({ ...l, hop: true, viaM: v,
                             source: hopById.get(l.person), target: personById.get(v) });
              }
            }
          }
        }
      }

      sim.nodes(nodes);
      sim.force("link", d3.forceLink(links).id((d) => d.id)
        .distance((l) => l.hop ? 46
          : state.mode === "bipartite" ? 38 + 4 * Math.sqrt(l.target.n_members || 1) : 70)
        .strength((l) => l.hop ? 0.3 : state.mode === "bipartite" ? 0.4 : Math.min(1, l.w / 8)));
      sim.force("collide", d3.forceCollide().radius((d) =>
        d.kind === "org" ? orgFont(d.n_members) * (d.label.length * 0.27) : d.kind === "hop" ? 14 : 20));
      sim.alpha(0.6).restart();

      linkSel = linkG.selectAll("line").data(links, (l) => l.hop
          ? "h|" + l.person + "|" + (l.viaM || l.org)
          : l.a ? l.a + "|" + l.b : l.person + "|" + l.org)
        .join("line")
        .attr("stroke", (l) => l.hop ? "#cbc2ad" : state.mode === "bipartite" ? "#d8d2c0" : "#c4bba6")
        .attr("stroke-width", (l) => l.hop || state.mode === "bipartite" ? 1 : projW(l.w))
        .attr("stroke-dasharray", (l) => l.hop ? "3 3" : null)
        .attr("stroke-opacity", 0.75)
        .style("pointer-events", state.mode === "people" ? "stroke" : "none")
        .on("mousemove", state.mode === "people"
          ? (ev, l) => l.hop ? hopTip(ev, hopById.get(l.person)) : pairTip(ev, l) : null)
        .on("mouseleave", hideTip);

      orgSel = orgG.selectAll("text").data(state.mode === "bipartite" ? vOrgs : [], (o) => o.id)
        .join("text")
        .attr("class", "olabel")
        .attr("fill", (o) => TYPE[o.type].color)
        .style("font-size", (o) => orgFont(o.n_members) + "px")
        .text((o) => o.label)
        .on("mousemove", orgTip).on("mouseleave", hideTip)
        .on("click", (ev, o) => { ev.stopPropagation(); select({ kind: "org", id: o.id }); });

      peopleSel = peopleG.selectAll("g.person").data(nodes.filter((n) => n.kind !== "org"), (p) => p.id)
        .join((enter) => {
          const g = enter.append("g")
            .attr("class", (p) => p.kind === "hop" ? "person hop" : "person")
            .style("cursor", "pointer");
          g.append("circle").attr("r", (p) => p.kind === "hop" ? 9 : 13)
            .attr("fill", (p) => p.kind === "hop" ? "#faf8f3" : colorOf(p))
            .attr("stroke", (p) => p.kind === "hop" ? "#b3a98f"
              : guests.has(p.id) ? "#8a6d9b" : "#faf8f3")
            .attr("stroke-dasharray", (p) => guests.has(p.id) ? "4 3" : null)
            .attr("stroke-width", (p) => guests.has(p.id) ? 2 : 1.5);
          g.append("text").attr("class", "ava-text")
            .style("font-size", (p) => p.kind === "hop" ? "7.5px" : "9px")
            .style("fill", (p) => p.kind === "hop" ? "#8c867b" : null)
            .text((p) => p.initials);
          g.append("text").attr("class", "plabel").attr("y", (p) => p.kind === "hop" ? 19 : 23)
            .attr("text-anchor", "middle")
            .style("font-size", (p) => p.kind === "hop" ? "9.5px" : null)
            .style("fill", (p) => p.kind === "hop" ? "#8c867b" : null)
            .text((p) => p.label);
          g.on("mousemove", (ev, p) => p.kind === "hop" ? hopTip(ev, p) : personTip(ev, p))
            .on("mouseleave", hideTip)
            .on("click", (ev, p) => {
              ev.stopPropagation();
              select({ kind: p.kind === "hop" ? "hop" : "person", id: p.id });
            })
            .call(d3.drag()
              .on("start", (ev, d) => { sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; })
              .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
              .on("end", (ev, d) => { sim.alphaTarget(0); d.fx = d.fy = null; }));
          return g;
        });

      applyHighlight();
      updateReach();
    }

    /* ---------- adaptive reach slider: one control, three meanings ---------- */
    const reachEl = document.getElementById("reach");
    const reachCtx = () => state.anchor ? "hops"
      : (state.mode === "bipartite" ? "orgmin" : "tiemin");

    function updateReach() {
      const ctx = reachCtx();
      const tw = data.meta.type_weights;
      const ruler = `lab ${tw.lab} · program ${tw.program} · company ${tw.company} · ` +
        `university ${tw.university} · community ${tw.community}`;
      let cfg;
      if (ctx === "orgmin") {
        cfg = {
          name: "Reach — org size", min: 1, max: d3.max(data.orgs, (o) => o.n_members) || 1,
          value: state.orgMin, k: state.orgMin <= 1 ? "all orgs" : "≥ " + state.orgMin + " people",
          desc: state.orgMin <= 1
            ? "Every org on the map. Slide right to keep only the rooms more people share."
            : "Only the rooms " + state.orgMin + "+ people have passed through.",
        };
      } else if (ctx === "tiemin") {
        cfg = {
          name: "Reach — tie strength", min: 1,
          max: Math.ceil(d3.max(data.projection, (p) => p.weight) || 1),
          value: state.tieMin, k: state.tieMin <= 1 ? "all ties" : "weight ≥ " + state.tieMin,
          desc: (state.tieMin <= 1 ? "Every tie. " : "Only deeper shared history. ") + ruler + ".",
        };
      } else {
        cfg = {
          name: "Reach — steps from " +
            ((personById.get(state.anchor) || {}).label || "").split(" ")[0],
          min: 1, max: 4, value: state.hops,
          k: state.hops + (state.hops === 1 ? " step" : " steps"),
          desc: "Everyone within " + state.hops + (state.hops === 1 ? " step" : " steps") +
            " of them, and the rooms carrying each step." +
            (shownHops.size ? " Hollow nodes (" + shownHops.size + "): people off the map who " +
              "shared those rooms — click one to see how, or to add them." : ""),
        };
      }
      reachEl.min = cfg.min; reachEl.max = cfg.max; reachEl.step = 1; reachEl.value = cfg.value;
      document.getElementById("reach-name").textContent = cfg.name;
      document.getElementById("reach-k").textContent = cfg.k;
      document.getElementById("reach-lo").textContent = cfg.min;
      document.getElementById("reach-hi").textContent = cfg.max;
      document.getElementById("reach-desc").textContent = cfg.desc;
    }

    reachEl.addEventListener("input", () => {
      const v = +reachEl.value, ctx = reachCtx();
      if (ctx === "hops") { state.hops = v; updateReach(); refreshReach(); }
      else if (ctx === "orgmin") { state.orgMin = v; render(); }
      else { state.tieMin = v; render(); }
    });

    // re-render (sim reheat) only when the off-map reveal actually changes;
    // otherwise a cheap highlight pass keeps the layout still
    function refreshReach() {
      const anb = neighborhood(state.anchor ? { kind: "person", id: state.anchor } : null);
      const ids = new Set(revealedHopLinks(anb).map((l) => l.person));
      const same = ids.size === shownHops.size && [...ids].every((i) => shownHops.has(i));
      if (same) applyHighlight(); else render();
    }

    function tick() {
      linkSel
        .attr("x1", (l) => l.source.x).attr("y1", (l) => l.source.y)
        .attr("x2", (l) => l.target.x).attr("y2", (l) => l.target.y);
      orgSel.attr("x", (o) => o.x).attr("y", (o) => o.y);
      peopleSel.attr("transform", (p) => `translate(${p.x},${p.y})`);
    }

    /* ---------- selection + highlight ---------- */
    function neighborhood(sel) {
      if (!sel) return null;
      const people = new Set(), orgs = new Set(), dist = new Map();
      if (sel.kind === "person") {
        // BFS over the projection: people within `hops` steps of the selected person,
        // plus the rooms carrying each step (the orgs of everyone closer than `hops`)
        let frontier = [sel.id];
        people.add(sel.id);
        dist.set(sel.id, 0);
        for (let d = 0; d < state.hops; d++) {
          const next = [];
          for (const id of frontier) {
            (linksByPerson.get(id) || []).forEach((l) => orgs.add(l.org));
            for (const nbr of projAdj.get(id) || []) {
              if (!people.has(nbr)) { people.add(nbr); dist.set(nbr, d + 1); next.push(nbr); }
            }
          }
          frontier = next;
        }
      } else {
        orgs.add(sel.id);
        (linksByOrg.get(sel.id) || []).forEach((l) => people.add(l.person));
      }
      return { people, orgs, dist };
    }

    function applyHighlight() {
      // a selected hop node keeps the anchoring person's reach lit (it's what revealed it)
      const nb = neighborhood(state.selected && state.selected.kind === "hop"
        ? (state.anchor ? { kind: "person", id: state.anchor } : null) : state.selected);
      const dim = (on) => (on ? 1 : 0.13);
      peopleSel.attr("opacity", (p) =>
        p.kind === "hop" ? 1 : !nb ? 1 : dim(nb.people.has(p.id)));
      orgSel.attr("opacity", (o) => !nb ? 1 : dim(nb.orgs.has(o.id)));
      linkSel.attr("stroke-opacity", (l) => {
        if (l.hop) return 0.8;               // hop links only exist while revealed
        if (!nb) return 0.75;
        // induced subgraph of the reach: a link is lit when both its ends are
        const hit = l.org
          ? nb.people.has(l.person) && nb.orgs.has(l.org)
          : nb.people.has(l.a) && nb.people.has(l.b);
        return hit ? 0.95 : 0.05;
      });
      peopleLis.forEach((li) =>
        li.classList.toggle("sel", !!nb && state.selected.kind === "person" && li.dataset.id === state.selected.id));
    }

    function select(sel) {
      state.selected = sel;
      if (sel && sel.kind === "person") state.anchor = sel.id;
      else if (!sel || sel.kind === "org") state.anchor = null;   // hop keeps the anchor
      renderDetail();
      updateReach();                        // the slider may change meaning (person ⇄ view)
      refreshReach();
      if (sel && sel.kind === "person") location.hash = "p=" + encodeURIComponent(sel.id);
      else if (location.hash && (!sel || sel.kind !== "hop"))
        history.replaceState(null, "", location.pathname + location.search);
    }

    /* ---------- sidebar: detail panel ---------- */
    const detail = document.getElementById("detail");
    function renderDetail() {
      const sel = state.selected;
      if (!sel) { detail.innerHTML = ""; return; }
      if (sel.kind === "person") {
        const p = personById.get(sel.id);
        const rows = (linksByPerson.get(sel.id) || []).map((l) => {
          const o = orgById.get(l.org);
          const yrs = [l.role, l.years].filter(Boolean).join(" · ");
          return `<div class="d-row"><span class="swatch" style="background:${TYPE[o.type].color}"></span>
            <div><span class="d-org">${esc(o.label)}</span>
            ${yrs ? `<div class="d-meta">${esc(yrs)}</div>` : ""}
            <div class="d-src">${l.source.startsWith("http")
              ? `<a href="${esc(l.source)}" target="_blank" rel="noopener">source</a>`
              : esc(l.source)}</div></div></div>`;
        });
        detail.innerHTML = `<span class="d-clear" title="clear">✕</span>
          <div class="d-title">${esc(p.label)}</div>
          <div class="d-sub">${esc(p.city || "")}</div>${rows.join("")}
          ${guests.has(sel.id) ? guestDetailExtras(sel.id) : `<div style="margin-top:4px"><a class="d-seat"
            href="/networks/affiliations/analyses/?p=${encodeURIComponent(p.id)}#your-seat"
            style="color:#6b665d;font-size:11px">see ${esc(p.label.split(" ")[0])}’s seat in the analyses →</a></div>`}`;
        if (guests.has(sel.id)) {
          detail.querySelector("#d-keep-guest").addEventListener("click", () => keepGuest(sel.id));
          detail.querySelector("#d-drop-guest").addEventListener("click", () => removeGuest(sel.id));
        }
      } else if (sel.kind === "hop") {
        const p = hopById.get(sel.id);
        const profile = sel.id.startsWith("h:gh:")
          ? "https://github.com/" + sel.id.slice(5)
          : "https://openalex.org/" + sel.id.slice(2);
        const rows = (hopLinksByPerson.get(sel.id) || []).map((l) => {
          const o = orgById.get(l.org);
          if (!o) return "";
          const via = l.via.map((v) =>
            `<span class="d-member" data-person="${esc(v)}" style="text-decoration:underline">` +
            esc((personById.get(v) || { label: v }).label) + `</span>`).join(", ");
          return `<div class="d-row"><span class="swatch" style="background:${TYPE[o.type].color}"></span>
            <div><span class="d-org">${esc(o.label)}</span>
            <div class="d-meta">with ${via}${l.years ? ` · ${esc(l.years)}` : ""}
              · ${l.n} co-event${l.n === 1 ? "" : "s"}</div>
            ${l.top ? `<div class="d-src">${esc(l.top)}</div>` : ""}</div></div>`;
        });
        detail.innerHTML = `<span class="d-clear" title="clear">✕</span>
          <div class="d-title">${esc(p.label)}</div>
          <div class="d-sub">off the map — shared rooms with people on it</div>${rows.join("")}
          <div class="d-src" style="margin:2px 0 6px"><a href="${esc(profile)}" target="_blank"
            rel="noopener">${sel.id.startsWith("h:gh:") ? "GitHub profile" : "OpenAlex profile"}</a></div>
          <button class="ep-act" id="d-add-hop" type="button" style="font-family:inherit;font-size:12px;
            background:#4c6b8a;color:#fff;border:none;border-radius:4px;padding:5px 10px;cursor:pointer">
            add ${esc(p.label.split(" ")[0])} to the map →</button>`;
        detail.querySelectorAll(".d-member").forEach((el) =>
          el.addEventListener("click", () => select({ kind: "person", id: el.dataset.person })));
        detail.querySelector("#d-add-hop").addEventListener("click", () => startJoinFor(p));
      } else {
        const o = orgById.get(sel.id);
        const rows = (linksByOrg.get(sel.id) || []).map((l) => {
          const p = personById.get(l.person);
          const meta = [l.role, l.years].filter(Boolean).join(" · ");
          return `<div class="d-row d-member" data-person="${esc(l.person)}">
            <span class="swatch" style="background:${colorOf(p)}"></span>
            <div><span class="d-org">${esc(p.label)}</span>
            ${meta ? `<div class="d-meta">${esc(meta)}</div>` : ""}</div></div>`;
        });
        detail.innerHTML = `<span class="d-clear" title="clear">✕</span>
          <div class="d-title" style="color:${TYPE[o.type].color}">${esc(o.label)}</div>
          <div class="d-sub">${TYPE[o.type].one} · ${o.n_members} people</div>${rows.join("")}`;
        detail.querySelectorAll(".d-member").forEach((el) =>
          el.addEventListener("click", () => select({ kind: "person", id: el.dataset.person })));
      }
      detail.querySelector(".d-clear").addEventListener("click", () => select(null));
    }

    /* ---------- sidebar: mode, type filters, people list ---------- */
    const modeDesc = document.getElementById("mode-desc");
    document.querySelectorAll("#viewctl .mode").forEach((b) =>
      b.addEventListener("click", () => {
        state.mode = b.dataset.mode;
        document.querySelectorAll("#viewctl .mode").forEach((x) => x.classList.toggle("on", x === b));
        modeDesc.textContent = MODE_DESC[state.mode];
        render();
      }));
    modeDesc.textContent = MODE_DESC.bipartite;

    const typectl = document.getElementById("typectl");
    const peopleUl = document.getElementById("people");
    let peopleLis = [];
    function rebuildSidebar() {
      const counts = d3.rollup(data.orgs, (v) => v.length, (o) => o.type);
      typectl.innerHTML = Object.entries(TYPE).map(([t, cfg]) =>
        `<div class="row${state.types.has(t) ? "" : " off"}" data-type="${t}">
         <span class="swatch" style="background:${cfg.color}"></span>
         ${cfg.many}<span class="n">${counts.get(t) || 0}</span></div>`).join("") +
        `<label class="singles"><input type="checkbox" id="singles"${state.singles ? " checked" : ""}>
         show single-member orgs (${data.orgs.filter((o) => o.n_members < 2).length} hidden)</label>`;
      typectl.querySelectorAll(".row").forEach((row) =>
        row.addEventListener("click", () => {
          const t = row.dataset.type;
          state.types.has(t) && state.types.size > 1 ? state.types.delete(t) : state.types.add(t);
          row.classList.toggle("off", !state.types.has(t));
          render();
        }));
      typectl.querySelector("#singles").addEventListener("change", (ev) => {
        state.singles = ev.target.checked;
        render();
      });

      peopleUl.innerHTML = data.people.slice().sort((a, b) => a.label.localeCompare(b.label))
        .map((p) =>
          `<li data-id="${esc(p.id)}"><span class="dot" style="background:${colorOf(p)}">${esc(p.initials)}</span>
           ${esc(p.label)}<span class="n">${(linksByPerson.get(p.id) || []).length}</span></li>`).join("");
      peopleLis = [...peopleUl.querySelectorAll("li")];
      peopleLis.forEach((li) =>
        li.addEventListener("click", () => select({ kind: "person", id: li.dataset.id })));

      document.getElementById("legend").innerHTML =
        data.communities.map((c) =>
          `<div class="row"><span class="swatch" style="background:${PALETTE[c.id % PALETTE.length]}"></span>${esc(c.label)} group</div>`
        ).join("") +
        `<div class="row" style="margin-top:6px;color:#8c867b">org colour = org type (left panel)</div>`;
    }
    rebuildSidebar();
    document.getElementById("search").addEventListener("input", (ev) => {
      const q = ev.target.value.trim().toLowerCase();
      peopleLis.forEach((li) =>
        li.style.display = li.textContent.toLowerCase().includes(q) ? "" : "none");
    });

    /* ---------- tooltips ---------- */
    function showTip(ev, html) {
      tooltip.innerHTML = html;
      tooltip.style.opacity = 1;
      const pad = 14, w = tooltip.offsetWidth, h = tooltip.offsetHeight;
      tooltip.style.left = Math.min(ev.clientX + pad, innerWidth - w - 8) + "px";
      tooltip.style.top = Math.min(ev.clientY + pad, innerHeight - h - 8) + "px";
    }
    const hideTip = () => (tooltip.style.opacity = 0);
    function personTip(ev, p) {
      const ls = linksByPerson.get(p.id) || [];
      const items = ls.slice(0, 8).map((l) => {
        const o = orgById.get(l.org);
        return `<li><span style="color:${TYPE[o.type].color}">●</span> ${esc(o.label)}${l.years ? ` <span class="t-dim">${esc(l.years)}</span>` : ""}</li>`;
      });
      if (ls.length > 8) items.push(`<li class="t-dim">… ${ls.length - 8} more — click for all</li>`);
      showTip(ev, `<div class="t-name">${esc(p.label)}</div>
        ${p.city ? `<div class="t-sub">${esc(p.city)}</div>` : ""}<ul class="t-list">${items.join("")}</ul>`);
    }
    function orgTip(ev, o) {
      const ms = (linksByOrg.get(o.id) || []).map((l) => personById.get(l.person).label);
      showTip(ev, `<div class="t-name" style="color:${TYPE[o.type].color}">${esc(o.label)}</div>
        <div class="t-sub">${TYPE[o.type].one} · ${o.n_members} people</div>
        <ul class="t-list"><li>${ms.map(esc).join("</li><li>")}</li></ul>`);
    }
    function hopTip(ev, p) {
      const ls = hopLinksByPerson.get(p.id) || [];
      const items = ls.map((l) => {
        const o = orgById.get(l.org);
        if (!o) return "";
        const via = l.via.map((v) => (personById.get(v) || { label: v }).label).join(", ");
        return `<li><span style="color:${TYPE[o.type].color}">●</span> ${esc(o.label)} — with ${esc(via)}` +
          `${l.years ? ` <span class="t-dim">${esc(l.years)}</span>` : ""}</li>`;
      });
      const top = ls.find((l) => l.top);
      showTip(ev, `<div class="t-name">${esc(p.label)}</div>
        <div class="t-sub">off the map — shared rooms, backed by co-authored work</div>
        <ul class="t-list">${items.join("")}${top ? `<li class="t-dim">e.g. ${esc(top.top)}</li>` : ""}</ul>`);
    }
    function pairTip(ev, l) {
      showTip(ev, `<div class="t-name">${esc(personById.get(l.a).label)} &amp; ${esc(personById.get(l.b).label)}</div>
        <ul class="t-list"><li>${l.shown.map((o) => esc(orgById.get(o).label)).join("</li><li>")}</li></ul>`);
    }

    /* ================= edit my row / join the map ================= */
    const editToggle = document.getElementById("edit-toggle");
    const editPanel = document.getElementById("edit-panel");
    const identity = (window.NetworkIdentity && window.NetworkIdentity.get()) || null;
    let editPerson = null;                  // canonical id being edited
    let joinDraft = null;                   // {name, city, entries: []} in join mode

    function save(type, payload, onOk) {
      const editor = (identity && identity.label) || payload.person || payload.name || null;
      fetch(API_BASE + "/affiliations/corrections", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, payload, editor }),
      }).then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ("HTTP " + r.status));
        foldOneLocally(type, payload);
        reindex(); rebuildSidebar(); render();
        if (onOk) onOk();
      }).catch((e) => editMsg("couldn’t save — " + e.message + " (the API may not be live yet)", true));
    }

    // optimistic local apply: fold ONE event into the overlay shape and reuse the merge function
    function foldOneLocally(type, payload) {
      const now = new Date().toISOString();
      const ov = { entry_set: {}, entry_remove: {}, city: {}, join: {} };
      const pid = ni(payload.person || payload.name || "");
      if (type === "aff_entry_set") {
        ov.entry_set[pid] = { [orgKey(payload.org)]: { org: payload.org, type: payload.type,
          role: payload.role || "", years: payload.years || "", current: !!payload.current,
          source: payload.source || "", ts: now } };
      } else if (type === "aff_entry_remove") {
        ov.entry_remove[pid] = [orgKey(payload.org)];
      } else if (type === "aff_city") {
        ov.city[pid] = payload.city;
      } else if (type === "aff_join") {
        dropHop(payload.name);
        ov.join[pid] = { name: payload.name, city: payload.city || "", ts: now };
        ov.entry_set[pid] = {};
        for (const e of payload.entries || []) {
          ov.entry_set[pid][orgKey(e.org)] = { org: e.org, type: e.type, role: e.role || "",
            years: e.years || "", current: !!e.current, source: e.source || "", ts: now };
        }
        if (payload.city) ov.city[pid] = payload.city;
      }
      applyAffOverlay(data, ov);   // mints new nodes with kind already set
    }

    function editMsg(text, isErr) {
      const m = editPanel.querySelector(".ep-msg");
      if (m) { m.textContent = text; m.className = "ep-msg " + (isErr ? "err" : "ok"); }
    }

    const orgDatalist = () =>
      `<datalist id="aff-orgs">${data.orgs.map((o) => `<option value="${esc(o.label)}">`).join("")}</datalist>`;
    const peopleDatalist = () =>
      `<datalist id="aff-people">${data.people.map((p) => `<option value="${esc(p.label)}">`).join("")}</datalist>`;

    const entryFormHTML = (idPrefix) => `
      <div class="ep-sec">
        <div class="ep-title">add an entry</div>
        <label>org (pick an existing spelling if it's there)</label>
        <input id="${idPrefix}-org" list="aff-orgs" autocomplete="off">
        <label>type</label>
        <select id="${idPrefix}-type">${Object.entries(TYPE).map(([t, c]) =>
          `<option value="${t}">${c.one}</option>`).join("")}</select>
        <label>role</label><input id="${idPrefix}-role" placeholder="PhD student, engineer…">
        <label>years</label><input id="${idPrefix}-years" placeholder="2021–2024 or 2021–">
        <label class="ep-row" style="margin-top:6px"><input type="checkbox" id="${idPrefix}-current"
          style="width:auto"> currently there</label>
        <label>source URL (optional — a page that shows it)</label>
        <input id="${idPrefix}-source" placeholder="https://…">
      </div>`;

    function readEntryForm(idPrefix) {
      const org = document.getElementById(idPrefix + "-org").value.trim();
      if (!org) { editMsg("org is required", true); return null; }
      const matched = data.orgs.find((o) => orgKey(o.label) === orgKey(org));
      return { org: matched ? matched.label : org,
               type: matched ? matched.type : document.getElementById(idPrefix + "-type").value,
               role: document.getElementById(idPrefix + "-role").value.trim(),
               years: document.getElementById(idPrefix + "-years").value.trim(),
               current: document.getElementById(idPrefix + "-current").checked,
               source: document.getElementById(idPrefix + "-source").value.trim() };
    }

    function showPicker(prefillJoin) {
      editPerson = null; joinDraft = prefillJoin ? { name: "", city: "", entries: [] } : null;
      if (joinDraft) { renderJoinForm(); return; }
      editPanel.innerHTML = `${peopleDatalist()}
        <div class="ep-hint">Edits are open and show up immediately; the hourly rebuild makes
          them permanent. Your row is yours — say so and it changes.</div>
        <label>whose row?</label>
        <input id="ep-who" list="aff-people" autocomplete="off"
          value="${identity ? esc(identity.label) : ""}">
        <button class="ep-act" id="ep-who-go" type="button">edit this row</button>
        <div>Someone missing — you, or anyone else? <a href="#" id="ep-join" style="color:#6b665d">add a person →</a></div>
        <div class="ep-msg"></div>`;
      editPanel.querySelector("#ep-who-go").addEventListener("click", () => {
        const label = editPanel.querySelector("#ep-who").value.trim();
        const p = data.people.find((x) => ni(x.label) === ni(label));
        if (!p) { editMsg("pick a name from the list", true); return; }
        editPerson = p.id;
        select({ kind: "person", id: p.id });
        renderRowEditor();
      });
      editPanel.querySelector("#ep-join").addEventListener("click", (ev) => {
        ev.preventDefault();
        joinDraft = { name: "", city: "", entries: [] };
        renderJoinForm();
      });
    }

    function renderRowEditor() {
      const p = personById.get(editPerson);
      const rows = (linksByPerson.get(editPerson) || []).map((l) => {
        const o = orgById.get(l.org);
        return `<li><span class="swatch" style="background:${TYPE[o.type].color};width:9px;height:9px;
          border-radius:50%;display:inline-block;margin-right:5px"></span>
          <span class="pt">${esc(o.label)}<div style="color:#8c867b">${esc([l.role, l.years].filter(Boolean).join(" · "))}</div></span>
          <button class="ep-mini" data-edit="${esc(o.label)}" data-etype="${o.type}"
            data-role="${esc(l.role)}" data-years="${esc(l.years)}">✎</button>
          <button class="ep-mini ep-danger" data-remove="${esc(o.label)}">✕</button></li>`;
      });
      editPanel.innerHTML = `${orgDatalist()}
        <div class="ep-title">${esc(p.label)} <button class="ep-mini" id="ep-back"
          style="float:right">↩ someone else</button></div>
        <div class="ep-row"><input id="ep-city" placeholder="city" value="${esc(p.city || "")}">
          <button class="ep-mini" id="ep-city-save">save</button></div>
        <ul class="ep-papers">${rows.join("")}</ul>
        ${entryFormHTML("ep")}
        <button class="ep-act" id="ep-add" type="button">save entry</button>
        <div class="ep-msg"></div>`;
      editPanel.querySelector("#ep-back").addEventListener("click", () => showPicker(false));
      editPanel.querySelector("#ep-city-save").addEventListener("click", () =>
        save("aff_city", { person: editPerson, city: editPanel.querySelector("#ep-city").value.trim() },
             () => { editMsg("city saved", false); }));
      editPanel.querySelectorAll("[data-remove]").forEach((b) =>
        b.addEventListener("click", () => {
          if (!confirm(`Remove “${b.dataset.remove}” from ${p.label}’s row?`)) return;
          save("aff_entry_remove", { person: editPerson, org: b.dataset.remove },
               () => { renderRowEditor(); select({ kind: "person", id: editPerson }); });
        }));
      editPanel.querySelectorAll("[data-edit]").forEach((b) =>
        b.addEventListener("click", () => {
          editPanel.querySelector("#ep-org").value = b.dataset.edit;
          editPanel.querySelector("#ep-type").value = b.dataset.etype;
          editPanel.querySelector("#ep-role").value = b.dataset.role;
          editPanel.querySelector("#ep-years").value = b.dataset.years;
        }));
      editPanel.querySelector("#ep-add").addEventListener("click", () => {
        const e = readEntryForm("ep");
        if (!e) return;
        save("aff_entry_set", { person: editPerson, ...e },
             () => { renderRowEditor(); select({ kind: "person", id: editPerson }); });
      });
    }

    function renderJoinForm() {
      const drafted = joinDraft.entries.map((e, i) =>
        `<li><span class="pt">${esc(e.org)} <span style="color:#8c867b">(${esc(e.type)})</span></span>
         <button class="ep-mini ep-danger" data-undraft="${i}">✕</button></li>`);
      editPanel.innerHTML = `${orgDatalist()}
        <div class="ep-title">add a person to the maps <button class="ep-mini" id="ep-back"
          style="float:right">↩ back</button></div>
        <div class="ep-hint">They appear on this map immediately; the papers map, the analyses,
          and their personal page (/networks/their-name) fill in within the hour.</div>
        <div class="ep-hint">Adding someone else is fine — they can edit their row anytime.</div>
        <label>your name</label><input id="ep-jname" value="${esc(joinDraft.name)}">
        <label>city (optional)</label><input id="ep-jcity" value="${esc(joinDraft.city)}">
        <label>homepage / Scholar (optional — helps Alex verify your papers later)</label>
        <input id="ep-jhome" value="${esc(joinDraft.homepage || "")}">
        <div class="ep-title" style="margin-top:6px">chapters to add (${joinDraft.entries.length})</div>
        <ul class="ep-papers">${drafted.join("")}</ul>
        ${entryFormHTML("epj")}
        <button class="ep-mini" id="epj-add" type="button">+ add to the list</button>
        <button class="ep-act" id="ep-join-save" type="button">join the map</button>
        <div class="ep-msg"></div>`;
      editPanel.querySelector("#ep-back").addEventListener("click", () => showPicker(false));
      editPanel.querySelectorAll("[data-undraft]").forEach((b) =>
        b.addEventListener("click", () => {
          joinDraft.entries.splice(+b.dataset.undraft, 1);
          keepDraftFields(); renderJoinForm();
        }));
      editPanel.querySelector("#epj-add").addEventListener("click", () => {
        const e = readEntryForm("epj");
        if (!e) return;
        if (joinDraft.entries.length >= 10) { editMsg("10 chapters max in one join", true); return; }
        joinDraft.entries.push(e);
        keepDraftFields(); renderJoinForm();
      });
      function keepDraftFields() {
        joinDraft.name = editPanel.querySelector("#ep-jname").value.trim();
        joinDraft.city = editPanel.querySelector("#ep-jcity").value.trim();
        joinDraft.homepage = editPanel.querySelector("#ep-jhome").value.trim();
      }
      editPanel.querySelector("#ep-join-save").addEventListener("click", () => {
        keepDraftFields();
        if (!joinDraft.name) { editMsg("name is required", true); return; }
        save("aff_join", { name: joinDraft.name, city: joinDraft.city,
               homepage: joinDraft.homepage || null, entries: joinDraft.entries },
             () => {
               const pid = ni(joinDraft.name);
               select({ kind: "person", id: pid });
               editPerson = pid; joinDraft = null;
               renderRowEditor();
               editMsg("they're on the map — both maps, the analyses, and their page fill in within the hour", false);
             });
      });
    }

    // "add them to the map" from a hop person's detail: prefill the join form with the
    // name and the rooms the co-events already vouch for (editable before saving)
    function startJoinFor(p) {
      const entries = (hopLinksByPerson.get(p.id) || []).map((l) => {
        const o = orgById.get(l.org);
        const m = (l.top || "").match(/github\.com\/[\w.-]+(\/[\w.-]+)?/);
        return o ? { org: o.label, type: o.type, role: "", years: l.years || "",
                     current: false, source: m ? "https://" + m[0] : "" } : null;
      }).filter(Boolean).slice(0, 10);
      joinDraft = { name: p.label, city: "", entries };
      editPanel.hidden = false;
      editToggle.classList.add("on");
      renderJoinForm();
    }

    editToggle.addEventListener("click", () => {
      const on = editPanel.hidden;
      editPanel.hidden = !on;
      editToggle.classList.toggle("on", on);
      if (on) showPicker(false);
    });

    /* ================= guest finder: try anyone on this map ================= */
    // Type a name -> people already here come first; otherwise OpenAlex (CORS-open, no key)
    // finds the author, and their affiliation record is placed as a TEMPORARY guest node,
    // wired through the same org nodes as everyone else. Unverified until added for real.
    const OA = "https://api.openalex.org";
    const MAILTO = "mailto=alexloftus2004%40gmail.com";
    const OATYPE = { education: "university", company: "company", facility: "lab",
                     government: "lab", healthcare: "university", nonprofit: "community" };
    const COUNTRYISH = /\s*\(([A-Z][a-z]+ ?){1,3}\)$/;   // "Google (United States)" -> "Google"
    const inst2org = (hops.meta && hops.meta.inst2org) || {};
    const guests = new Map();               // guest pid -> bare OpenAlex author id

    function guestEntries(author) {
      return (author.affiliations || []).map((a) => {
        const inst = a.institution || {};
        const nm = inst.display_name || "";
        if (!nm) return null;
        const oid = inst2org[nm.toLowerCase()] || inst2org[nm.replace(COUNTRYISH, "").toLowerCase()];
        const known = oid && orgById.get(oid);
        const ys = (a.years || []).slice().sort();
        return {
          org: known ? known.label : nm.replace(COUNTRYISH, ""),
          type: known ? known.type : (OATYPE[inst.type] || "community"),
          years: ys.length ? (ys[0] === ys[ys.length - 1] ? String(ys[0]) : ys[0] + "–" + ys[ys.length - 1]) : "",
        };
      }).filter(Boolean).slice(0, 12);
    }

    function addGuest(author, extraEntries) {
      const oaid = author.id.split("/").pop();
      const pid = "guest:" + oaid;
      if (!guests.has(pid)) {
        const now = new Date().toISOString();
        const ov = { join: { [pid]: { name: author.display_name, city: "", ts: now } },
                     entry_set: { [pid]: {} } };
        // live affiliations first; index-cached entries fill rooms the live record misses
        for (const e of guestEntries(author).concat(extraEntries || [])) {
          if (ov.entry_set[pid][orgKey(e.org)]) continue;
          ov.entry_set[pid][orgKey(e.org)] = { org: e.org, type: e.type, role: "",
            years: e.years, current: false, source: "OpenAlex (unverified)", ts: now };
        }
        applyAffOverlay(data, ov);
        guests.set(pid, oaid);
        reindex(); rebuildSidebar(); render();
      }
      select({ kind: "person", id: pid });
    }

    // place by OpenAlex id: live record + any index-cached entries, cache-only if offline
    function placeByOa(oaid, label, entries) {
      fetch(`${OA}/authors/${oaid}?${MAILTO}`, { signal: AbortSignal.timeout(8000) })
        .then((r) => r.json())
        .then((author) => addGuest(author, entries))
        .catch(() => addGuest({ id: "x/" + oaid, display_name: label, affiliations: [] }, entries));
    }

    function removeGuest(pid) {
      data.people = data.people.filter((p) => p.id !== pid);
      data.links = data.links.filter((l) => l.person !== pid);
      guests.delete(pid);
      recomputeDerived(data);
      reindex(); rebuildSidebar();
      select(null);                          // also clears anchor; render below rebuilds nodes
      render();
    }

    // a guest's detail panel gets the unverified badge + keep/dismiss actions
    function guestDetailExtras(pid) {
      const p = personById.get(pid);
      const oaid = guests.get(pid);
      return `<div style="margin:6px 0 4px"><span class="guest-badge">guest preview · OpenAlex, unverified</span></div>
        <div class="d-src" style="margin-bottom:6px"><a href="https://openalex.org/${esc(oaid)}"
          target="_blank" rel="noopener">OpenAlex profile</a></div>
        <button class="ep-act" id="d-keep-guest" type="button" style="font-family:inherit;font-size:12px;
          background:#4c6b8a;color:#fff;border:none;border-radius:4px;padding:5px 10px;cursor:pointer">
          add ${esc(p.label.split(" ")[0])} to the map →</button>
        <button class="ep-mini" id="d-drop-guest" type="button" style="font-family:inherit;font-size:11px;
          background:none;border:1px solid #ddd6c8;border-radius:4px;padding:4px 8px;cursor:pointer;
          color:#6b665d;margin-left:6px">dismiss</button>`;
    }

    function keepGuest(pid) {
      const p = personById.get(pid);
      const oaid = guests.get(pid);
      const entries = (linksByPerson.get(pid) || []).map((l) => {
        const o = orgById.get(l.org);
        return o ? { org: o.label, type: o.type, role: "", years: l.years || "",
                     current: false, source: "https://openalex.org/" + oaid } : null;
      }).filter(Boolean).slice(0, 10);
      const name = p.label;
      removeGuest(pid);                      // the join flow mints the real node
      joinDraft = { name, city: "", entries };
      editPanel.hidden = false;
      editToggle.classList.add("on");
      renderJoinForm();
    }

    const finderQ = document.getElementById("finder-q");
    const finderRes = document.getElementById("finder-results");
    let finderTimer = null, finderSeq = 0;

    function finderRow(html, cls) {
      const div = document.createElement("div");
      div.className = cls || "fr";
      div.innerHTML = html;
      return div;
    }

    function runFinder(q) {
      const seq = ++finderSeq;
      finderRes.innerHTML = "";
      finderRes.hidden = false;
      const ql = q.toLowerCase();
      const locals = data.people.filter((p) => p.label.toLowerCase().includes(ql)).slice(0, 4);
      for (const p of locals) {
        const row = finderRow(`<span class="fr-name">${esc(p.label)}</span>
          <span class="fr-hint">${esc(p.city || "")}</span>
          <span class="fr-tag">${guests.has(p.id) ? "guest" : "on the map"}</span>`);
        row.addEventListener("mousedown", () => { closeFinder(); select({ kind: "person", id: p.id }); });
        finderRes.appendChild(row);
      }
      // the extended graph: outside coauthors ranked by papers shared with the group
      // (guest-index.json) — placeable instantly, most likely to be searched for
      const taken = new Set(locals.map((p) => ni(p.label)));
      const idxHits = gidx.filter((it) => it.label.toLowerCase().includes(ql)
        && !taken.has(ni(it.label))).slice(0, 4);
      for (const it of idxHits) {
        taken.add(ni(it.label));
        const row = finderRow(`<span class="fr-name">${esc(it.label)}</span>
          <span class="fr-hint">${esc((it.via || []).map((v) =>
            (personById.get(v) || { label: v }).label.split(" ")[0]).join(", "))}</span>
          <span class="fr-tag">${it.n} paper${it.n === 1 ? "" : "s"} w/ the group</span>`);
        row.addEventListener("mousedown", () => { closeFinder(); placeByOa(it.oa, it.label, it.entries); });
        finderRes.appendChild(row);
      }
      fetch(`${OA}/autocomplete/authors?q=${encodeURIComponent(q)}&${MAILTO}`,
            { signal: AbortSignal.timeout(6000) })
        .then((r) => r.json())
        .then((res) => {
          if (seq !== finderSeq) return;     // a newer query superseded this one
          // exact name match outranks volume (David Bau beats David Baum), then works count
          const score = (it) => (it.display_name.toLowerCase() === ql ? 1e9 : 0) + (it.works_count || 0);
          const items = (res.results || []).slice(0, 8)
            .sort((a, b) => score(b) - score(a)).slice(0, 6)
            .filter((it) => !taken.has(ni(it.display_name)));
          for (const it of items) {
            const row = finderRow(`<span class="fr-name">${esc(it.display_name)}</span>
              <span class="fr-hint">${esc(it.hint || "")}</span>
              <span class="fr-tag">${it.works_count} works · OpenAlex</span>`);
            row.addEventListener("mousedown", () => {
              closeFinder();
              placeByOa(it.id.split("/").pop(), it.display_name);
            });
            finderRes.appendChild(row);
          }
          if (!items.length && !locals.length && !idxHits.length) {
            const row = finderRow(`no public record found — <u>add them by hand →</u>`, "fr-note");
            row.style.cursor = "pointer";
            row.addEventListener("mousedown", () => {
              closeFinder();
              editPanel.hidden = false;
              editToggle.classList.add("on");
              joinDraft = { name: q, city: "", entries: [] };
              renderJoinForm();
            });
            finderRes.appendChild(row);
          }
        })
        .catch(() => {});
    }

    function closeFinder() {
      finderRes.hidden = true;
      finderRes.innerHTML = "";
      finderQ.value = "";
    }

    // finder + add-person are optional chrome; if the page markup is missing them
    // (e.g. a stale HTML build), skip wiring rather than throwing and aborting init()
    // before render() — the graph must always draw.
    if (finderQ && finderRes) {
      finderQ.addEventListener("input", () => {
        clearTimeout(finderTimer);
        const q = finderQ.value.trim();
        if (q.length < 3) { finderRes.hidden = true; return; }
        finderTimer = setTimeout(() => runFinder(q), 250);
      });
      finderQ.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeFinder(); });
      finderQ.addEventListener("blur", () => setTimeout(() => (finderRes.hidden = true), 200));
    }

    // "+" floating top-right of the graph: straight into the add-a-person form
    const addPersonBtn = document.getElementById("add-person");
    if (addPersonBtn) addPersonBtn.addEventListener("click", () => {
      editPanel.hidden = false;
      editToggle.classList.add("on");
      showPicker(true);
      editPanel.scrollIntoView({ block: "nearest" });
    });

    /* ---------- deep links: #p= select, ?edit=1 (&join=1) auto-open ---------- */
    function selectFromHash() {
      const m = location.hash.match(/^#p=(.+)$/);
      if (m) {
        const id = decodeURIComponent(m[1]);
        if (personById.has(id)) { select({ kind: "person", id }); return true; }
      }
      return false;
    }
    render();
    selectFromHash();
    window.addEventListener("hashchange", selectFromHash);
    const params = new URLSearchParams(location.search);
    if (params.get("guest")) {              // handoff from the papers map's finder
      const it = gidx.find((x) => x.oa === params.get("guest"));
      placeByOa(params.get("guest"), (it && it.label) || "", it && it.entries);
    }
    if (params.get("edit") === "1") {
      editPanel.hidden = false;
      editToggle.classList.add("on");
      if (params.get("join") === "1") showPicker(true);
      else {
        showPicker(false);
        const pre = params.get("p") || (location.hash.match(/^#p=(.+)$/) || [])[1];
        if (pre) {
          const p = personById.get(decodeURIComponent(pre));
          if (p) { editPanel.querySelector("#ep-who").value = p.label; }
        }
      }
    }
  }
})();
