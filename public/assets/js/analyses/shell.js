/* shell.js — registry, loader, toolbar, and shared utilities for the analyses pages.
   Orchestrator-owned: panel agents read this but never edit it. Panels self-register via
   window.AnalysesRegistry.register(slug, def) from their own IIFE files (any load order).

   Each analyses page declares its panel set + data sources in a plain (non-defer) script
   BEFORE this file loads:
     window.ANALYSES_CONFIG = {
       methods: [{slug, sec, label, sub}, …],          // required — panel order/grouping
       sharedPath: "/assets/data/analyses/shared.json", // node lookup {nodes, communities, …}
       panelDataDir: "/assets/data/analyses/",          // <slug>.json per panel
       minimap: { layoutPath: "/assets/data/coauthorship.json",  // node x/y source
                  restrictToShared: false,   // draw only nodes present in shared.json
                  links: "layout" }          // "layout" = graph links | "shared" = shared.links
     };
   Defaults reproduce the original /coauthorship/analyses/ behavior. */
(function () {
  "use strict";

  var CFG = window.ANALYSES_CONFIG || {};
  var METHODS = CFG.methods || [];
  var SHARED_PATH = CFG.sharedPath || "/assets/data/analyses/shared.json";
  var DATA_DIR = CFG.panelDataDir || "/assets/data/analyses/";
  var MM = CFG.minimap || {};

  // ---- registry stub (panels may have registered before this file ran) ----
  var prior = (window.AnalysesRegistry && window.AnalysesRegistry._q) || [];
  var defs = {};
  window.AnalysesRegistry = { _q: [], register: function (s, d) { defs[s] = d; } };
  prior.forEach(function (p) { defs[p[0]] = p[1]; });

  // ---- shared utilities (built once shared.json arrives) ----
  var COLORS = {
    community: ["#4c6b8a", "#a6611a", "#5a7d5a"],
    other: "#b3a98f", bg: "#faf8f3", ink: "#2b2b2b", muted: "#8c867b", hair: "#e3ddcf",
    src: { both: "#5a7d5a", s2: "#a6611a", oa: "#4c6b8a" },
  };
  var fmt = {
    num: function (x) { return (+x).toLocaleString("en-US", { maximumFractionDigits: 1 }); },
    pct: function (x) { return Math.round(100 * x) + "%"; },
    sig: function (x, d) { return (+x).toPrecision(d || 2); },
  };
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  var tipEl = null;
  var tooltip = {
    show: function (html, evt) {
      if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "m-tip"; document.body.appendChild(tipEl); }
      if (html !== tipEl._last) { tipEl.innerHTML = html; tipEl._last = html; }  // mousemove fires at pointer rate
      var x = Math.min(evt.clientX + 14, window.innerWidth - 320), y = evt.clientY + 12;
      tipEl.style.left = x + "px"; tipEl.style.top = y + "px"; tipEl.style.opacity = 1;
    },
    hide: function () { if (tipEl) tipEl.style.opacity = 0; },
  };

  var shared = {
    colors: COLORS, fmt: fmt, esc: esc, tooltip: tooltip,
    nodes: new Map(), communities: [],
    labelOf: function (id) { var n = shared.nodes.get(id); return n ? n.label : id; },
    communityOf: function (id) { var n = shared.nodes.get(id); return n ? n.community : -1; },
    isList: function (id) { var n = shared.nodes.get(id); return !!(n && n.is_list); },
    // a roster member's seat page; null for connectors/guests (who have no seat). The
    // /networks/<slug>/ stub (built by build_affiliations.py) forwards to ?p=…#your-seat.
    seatHref: function (id) {
      return shared.isList(id) ? "/networks/" + id.replace(/ /g, "-") + "/" : null;
    },
    seatLink: function (id, label) {                 // -> <a> to the seat, or plain text
      var txt = esc(label != null ? label : shared.labelOf(id));
      var href = shared.seatHref(id);
      return href ? '<a href="' + href + '" class="seat-link">' + txt + "</a>" : txt;
    },
    colorOf: function (id) {
      var c = shared.communityOf(id);
      return c >= 0 ? COLORS.community[c % COLORS.community.length] : COLORS.other;
    },
    minimap: minimap,
  };

  // ---- shared.json: node lookup, loaded once at module scope (defer order guarantees d3) ----
  var sharedLinks = [];                      // [[a, b, weight], …] when the shared file carries them
  var sharedReady = d3.json(SHARED_PATH).then(function (s) {
    Object.keys(s.nodes).forEach(function (id) { shared.nodes.set(id, s.nodes[id]); });
    shared.communities = s.communities;
    shared.years = s.years;
    sharedLinks = s.links || [];
  });

  // ---- mini-map: the page's repeated visual anchor (same layout, recolored per panel) ----
  var graphPromise = null;
  function loadGraph() {
    if (!graphPromise) graphPromise = d3.json(MM.layoutPath || "/assets/data/coauthorship.json");
    return graphPromise;
  }
  function minimap(el, colorFn, opts) {
    opts = opts || {};
    var W = opts.width || 200, H = opts.height || 160;
    var svg = d3.select(el).append("svg").attr("width", W).attr("height", H);
    var api = { svg: svg, highlight: function () {} };
    Promise.all([loadGraph(), sharedReady]).then(function (rs) {
      var g = rs[0];
      var drawn = MM.restrictToShared
        ? g.nodes.filter(function (n) { return shared.nodes.has(n.id); })
        : g.nodes;
      var xs = d3.extent(drawn, function (n) { return n.x; });
      var ys = d3.extent(drawn, function (n) { return n.y; });
      var sx = d3.scaleLinear().domain(xs).range([6, W - 6]);
      var sy = d3.scaleLinear().domain(ys).range([6, H - 6]);
      var pos = {};
      drawn.forEach(function (n) { pos[n.id] = [sx(n.x), sy(n.y)]; });
      var edges = MM.links === "shared"
        ? sharedLinks.map(function (l) { return { source: l[0], target: l[1] }; })
        : g.links;
      svg.append("g").selectAll("line")
        .data(edges.filter(function (l) { return pos[l.source] && pos[l.target]; })).join("line")
        .attr("x1", function (l) { return pos[l.source][0]; })
        .attr("y1", function (l) { return pos[l.source][1]; })
        .attr("x2", function (l) { return pos[l.target][0]; })
        .attr("y2", function (l) { return pos[l.target][1]; })
        .attr("stroke", COLORS.hair).attr("stroke-width", 0.5)
        .attr("stroke-opacity", opts.edgeOpacity == null ? 0.8 : opts.edgeOpacity);
      var nodes = svg.append("g").selectAll("circle")
        .data(drawn.filter(function (n) { return !n.path_only; })).join("circle")
        .attr("cx", function (n) { return pos[n.id][0]; })
        .attr("cy", function (n) { return pos[n.id][1]; })
        .attr("r", function (n) { return opts.radiusFn ? opts.radiusFn(n.id) : (n.is_list ? 3.2 : 2.2); })
        .attr("fill", function (n) { return colorFn(n.id) || COLORS.other; })
        .attr("fill-opacity", function (n) { return opts.opacityFn ? opts.opacityFn(n.id) : 0.9; });
      api.highlight = function (id) {
        nodes.interrupt();
        if (id == null) { nodes.attr("stroke", null).attr("fill-opacity", function (n) { return opts.opacityFn ? opts.opacityFn(n.id) : 0.9; }); }
        else {
          nodes.attr("stroke", function (n) { return n.id === id ? COLORS.ink : null; })
            .attr("stroke-width", 1.4)
            .attr("fill-opacity", function (n) { return n.id === id ? 1 : 0.35; });
        }
      };
      if (opts.onReady) opts.onReady(api);
    });
    return api;
  }

  // ---- page assembly ----
  document.addEventListener("DOMContentLoaded", function () {
    var nav = document.getElementById("nav");
    var main = document.getElementById("main");
    var panels = {}, loaded = {}, current = null;

    if (!METHODS.length) {   // stale-cache window: page HTML predates the config split
      console.warn("[analyses] ANALYSES_CONFIG.methods missing or empty");
      main.innerHTML = '<div class="m-panel active"><div class="m-err"><div class="m-err-t">' +
        "Loading mismatch.</div>This page and its script are briefly out of sync — " +
        "refresh in a minute and the analyses will be back.</div></div>";
      return;
    }

    var lastSec = null;
    METHODS.forEach(function (m, i) {
      if (m.sec !== lastSec) {
        var sec = document.createElement("div");
        sec.className = "nav-sec"; sec.textContent = m.sec;
        nav.appendChild(sec); lastSec = m.sec;
      }
      var item = document.createElement("div");
      item.className = "nav-item"; item.dataset.slug = m.slug;
      item.innerHTML = '<span class="nav-label"><span class="nav-idx">' + (i + 1) + "</span>" +
        esc(m.label) + '</span><div class="nav-sub">' + esc(m.sub) + "</div>";
      item.addEventListener("click", function () { activate(m.slug, true); });
      item.addEventListener("mouseenter", function () { prefetch(m.slug); });
      nav.appendChild(item);

      var p = document.createElement("section");
      p.className = "m-panel"; p.id = "panel-" + m.slug;
      p.innerHTML = '<div class="m-kicker">' + esc(m.sec) + " · " + (i + 1) + " of " + METHODS.length +
        '</div><h2 class="m-title">' + esc(m.label) + '</h2>' +
        '<div class="m-intro m-prose"></div><div class="m-viz"></div>' +
        '<div class="m-how" style="display:none"><div class="m-how-k">How it works</div><div class="m-prose"></div></div>' +
        '<details class="m-note" style="display:none"><summary>For the curious</summary><div></div></details>';
      main.appendChild(p);
      panels[m.slug] = p;
    });

    var fetches = {};
    function prefetch(slug) {
      if (!fetches[slug]) fetches[slug] = d3.json(DATA_DIR + slug + ".json");
      return fetches[slug];
    }

    function showError(slug, why) {
      var m = METHODS.find(function (x) { return x.slug === slug; });
      panels[slug].querySelector(".m-viz").innerHTML =
        '<div class="m-err"><div class="m-err-t">This panel didn’t load.</div>' +
        "The numbers behind “" + esc(m.label) + "” couldn’t be fetched — the network itself is fine, " +
        "just this analysis of it. Pick another panel on the left.<br><span style=\"font-size:11.5px\">(" +
        esc(why) + ")</span></div>";
    }

    function renderInto(slug, data) {
      var def = defs[slug], p = panels[slug];
      var viz = p.querySelector(".m-viz");
      viz.innerHTML = "";
      if (data.headline) p.querySelector(".m-title").innerHTML = data.headline; // build-time JSON, trusted
      if (def.prose) {
        if (def.prose.intro) p.querySelector(".m-intro").innerHTML = def.prose.intro;
        if (def.prose.how) {
          var how = p.querySelector(".m-how");
          how.style.display = ""; how.querySelector(".m-prose").innerHTML = def.prose.how;
        }
        if (def.prose.method) {
          var note = p.querySelector(".m-note");
          note.style.display = ""; note.querySelector("div").innerHTML = def.prose.method;
        }
      }
      try { def.render(viz, data, shared); }
      catch (e) { console.warn("[analyses] render failed:", slug, e); showError(slug, "render error: " + e.message); }
    }

    function activate(slug, push) {
      if (!panels[slug]) slug = METHODS[0].slug;
      if (current === slug) return;
      if (current) { panels[current].classList.remove("active"); }
      current = slug;
      panels[slug].classList.add("active");
      document.getElementById("main").scrollTop = 0;
      nav.querySelectorAll(".nav-item").forEach(function (it) {
        it.classList.toggle("sel", it.dataset.slug === slug);
      });
      if (push) history.pushState(null, "", "#" + slug);
      if (!defs[slug]) { showError(slug, "module not registered"); return; }
      if (loaded[slug]) return;
      panels[slug].querySelector(".m-viz").innerHTML = '<div class="m-loading">computing the picture…</div>';
      Promise.all([prefetch(slug), sharedReady]).then(function (rs) {
        loaded[slug] = true; renderInto(slug, rs[0]);
      }).catch(function (e) { console.warn("[analyses] load failed:", slug, e); showError(slug, String(e)); });
    }

    // resize: re-render current panel from scratch (debounced)
    var rt = null;
    window.addEventListener("resize", function () {
      clearTimeout(rt);
      rt = setTimeout(function () {
        if (current && loaded[current] && defs[current]) {
          prefetch(current).then(function (d) { renderInto(current, d); });
        }
      }, 220);
    });

    // keyboard: step panels
    document.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      var idx = METHODS.findIndex(function (m) { return m.slug === current; });
      var next = METHODS[idx + (e.key === "ArrowDown" ? 1 : -1)];
      if (next) { e.preventDefault(); activate(next.slug, true); }
    });
    window.addEventListener("popstate", function () { activate(location.hash.slice(1) || METHODS[0].slug, false); });

    activate(location.hash.slice(1) || METHODS[0].slug, false);
  });
})();
