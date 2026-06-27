/* The Perfume Atlas — /perfumes/
 * 24,050 fragrances laid out by smell (UMAP on IDF + pyramid-weighted note/accord vectors):
 * closer together  ==  smells more alike. Colour = scent family. Size = how loved (review count).
 * Nodes are the actual bottles: far out they're soft coloured points, zoom in and they bloom into
 * photographs. Hover for a card; click to trace a perfume's closest scent-twins (the graph edges,
 * weighted by similarity). Canvas + d3-zoom; the bottle images are hot-linked from Fragrantica's CDN.
 */
(function () {
  "use strict";

  const ATLAS_URL = "/assets/data/perfumes-atlas.json";
  const NBR_URL = "/assets/data/perfumes-neighbors.json";

  const MIN_R = 0.7;          // px — faintest dot
  const MAX_R = 56;           // px — biggest bloomed bottle
  const IMG_AT = 11;          // px screen radius at/above which a node shows its photograph
  const IMG_CAP = 420;        // max bottle photos drawn per frame (most-loved win the slots)
  const INFLIGHT_MAX = 12;    // concurrent image downloads
  const IMG_CACHE_MAX = 3000; // bound the decoded-image cache (eviction keeps memory in check)
  const ROSE = "#b8757d";

  const canvas = document.getElementById("atlas");
  const ctx = canvas.getContext("2d");
  const tip = document.getElementById("tip");

  let A = null, NBR = null;            // data
  let N = 0, dataR = null;             // per-node data-space radius (popularity)
  let colorByNode = null;              // per-node family colour, precomputed (hot-path lookup)
  let famById = new Map();
  let quad = null;                     // d3 quadtree for hover hit-testing
  let transform = d3.zoomIdentity;
  let W = 0, H = 0, dpr = 1;
  let hover = -1, selected = -1, hoverFam = -1, pinnedFam = -1;
  const imgCache = new Map();          // pid -> Image | "err"
  let inFlight = 0;
  let needsDraw = false;

  // URL templates are single-sourced from the build (A.meta) so they can't drift across files
  const imgURL = (pid) => A.meta.img.replace("{pid}", pid);
  const fragURL = (pid) => A.meta.frag.replace("{pid}", pid);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const star = (r) => (r ? r.toFixed(2) : "—");
  const reviewsFmt = (n) => (n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : "" + n);

  // ---------- sizing ----------
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    requestDraw();
  }

  // ---------- image loading (lazy, capped, bounded) ----------
  function getImage(pid) {
    const c = imgCache.get(pid);
    // only return a *drawable* image; an in-flight one (naturalWidth 0) draws nothing, so fall
    // back to the placeholder dot until it has decoded
    if (c !== undefined) return c === "err" ? null : (c.complete && c.naturalWidth ? c : null);
    if (inFlight >= INFLIGHT_MAX) return null;        // try again a later frame
    inFlight++;
    const img = new Image();                           // no crossOrigin: CDN sends no ACAO; we only draw
    img.onload = () => { inFlight--; imgCache.set(pid, img); evictImages(); requestDraw(); };
    img.onerror = () => { inFlight--; imgCache.set(pid, "err"); };
    img.src = imgURL(pid);
    imgCache.set(pid, img);                            // mark in-flight so we don't re-request
    return null;                                       // not ready this frame
  }
  function evictImages() {
    if (imgCache.size <= IMG_CACHE_MAX) return;
    for (const k of imgCache.keys()) {                 // Map iterates oldest-first → FIFO eviction
      if (imgCache.size <= IMG_CACHE_MAX) break;
      const v = imgCache.get(k);
      if (v !== "err" && v.complete) imgCache.delete(k);   // never drop an in-flight load
    }
  }

  // ---------- draw ----------
  function requestDraw() { if (!needsDraw) { needsDraw = true; requestAnimationFrame(draw); } }

  function screenR(i, k) {
    return Math.max(MIN_R, Math.min(MAX_R, k * dataR[i]));
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);   // native (Chrome 99+/Safari 16.4+/FF 112+)
  }

  function drawBottle(sx, sy, r, img, color, emph) {
    // portrait tile centred on (sx,sy); image drawn "contain" on a cream card with a hairline frame
    const h = r * 2, w = h * 0.8, x = sx - w / 2, y = sy - h / 2, rad = Math.max(2, r * 0.16);
    ctx.save();
    if (emph) { ctx.shadowColor = "rgba(80,60,50,.28)"; ctx.shadowBlur = 14; ctx.shadowOffsetY = 3; }
    roundRect(x, y, w, h, rad); ctx.fillStyle = "#fffdf8"; ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.save(); roundRect(x + 1, y + 1, w - 2, h - 2, rad - 1); ctx.clip();
    const pad = r * 0.12, aw = w - 2 * pad, ah = h - 2 * pad;
    const ir = img.naturalWidth / img.naturalHeight || 0.75;
    let dw = aw, dh = aw / ir; if (dh > ah) { dh = ah; dw = ah * ir; }
    ctx.drawImage(img, sx - dw / 2, sy - dh / 2, dw, dh);
    ctx.restore();
    roundRect(x, y, w, h, rad);
    ctx.lineWidth = emph ? 2 : 1; ctx.strokeStyle = emph ? ROSE : color; ctx.stroke();
    ctx.restore();
  }

  function draw() {
    needsDraw = false;
    if (!A) return;
    const k = transform.k;
    ctx.clearRect(0, 0, W, H);

    // visible window in data space (+ margin)
    const m = MAX_R;
    const x0 = (-m - transform.x) / k, x1 = (W + m - transform.x) / k;
    const y0 = (-m - transform.y) / k, y1 = (H + m - transform.y) / k;
    const X = A.x, Y = A.y, F = A.fam, RV = A.reviews, COL = colorByNode;

    // selected node's twin edges, drawn beneath everything
    if (selected >= 0 && NBR) {
      const sx = transform.applyX(X[selected]), sy = transform.applyY(Y[selected]);
      const nb = NBR.nbr[selected], w = NBR.w[selected];
      for (let t = 0; t < nb.length; t++) {
        const j = nb[t];
        const jx = transform.applyX(X[j]), jy = transform.applyY(Y[j]);
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(jx, jy);
        ctx.lineWidth = 0.6 + w[t] * 3.4;
        ctx.strokeStyle = `rgba(184,117,125,${0.25 + w[t] * 0.55})`;
        ctx.stroke();
      }
    }

    // collect nodes that qualify for a photo; cap by popularity so frames stay cheap
    const photo = [];
    // pass 1: dots (and gather photo candidates)
    for (let i = 0; i < N; i++) {
      const x = X[i], y = Y[i];
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      const dim = (hoverFam >= 0 && F[i] !== hoverFam);
      const r = screenR(i, k);
      const sx = transform.applyX(x), sy = transform.applyY(y);
      if (r >= IMG_AT) { photo.push(i); }
      ctx.globalAlpha = dim ? 0.07 : (r < 2 ? 0.55 : 0.82);
      ctx.beginPath(); ctx.arc(sx, sy, Math.min(r, IMG_AT - 1), 0, 6.2832);
      ctx.fillStyle = COL[i]; ctx.fill();
    }
    ctx.globalAlpha = 1;

    // pass 2: photos — most-reviewed first, capped
    photo.sort((a, b) => RV[b] - RV[a]);
    let drawn = 0;
    for (let p = 0; p < photo.length && drawn < IMG_CAP; p++) {
      const i = photo[p];
      if (hoverFam >= 0 && F[i] !== hoverFam) continue;
      const img = getImage(A.pid[i]);
      const r = screenR(i, k);
      const sx = transform.applyX(X[i]), sy = transform.applyY(Y[i]);
      if (img) { drawBottle(sx, sy, r, img, COL[i], false); drawn++; }
      else {     // not loaded yet: a slightly richer dot so the spot isn't empty
        ctx.beginPath(); ctx.arc(sx, sy, r, 0, 6.2832);
        ctx.fillStyle = COL[i]; ctx.globalAlpha = 0.9; ctx.fill();
        ctx.globalAlpha = 1; ctx.lineWidth = 1; ctx.strokeStyle = "#fffdf8"; ctx.stroke();
      }
    }

    // emphasise selected + its twins (always with photo if possible)
    if (selected >= 0 && NBR) {
      for (const j of NBR.nbr[selected]) emphasise(j, k, false);
      emphasise(selected, k, true);
    }
    if (hover >= 0 && hover !== selected) emphasise(hover, k, false);
  }

  function emphasise(i, k, isSel) {
    const sx = transform.applyX(A.x[i]), sy = transform.applyY(A.y[i]);
    const r = Math.max(screenR(i, k), isSel ? 22 : 16);
    const img = getImage(A.pid[i]);
    if (img) drawBottle(sx, sy, r, img, colorByNode[i], true);
    else {
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, 6.2832);
      ctx.fillStyle = colorByNode[i]; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = ROSE; ctx.stroke();
    }
  }

  // ---------- hover hit-testing ----------
  function nodeAt(mx, my) {
    const dx = (mx - transform.x) / transform.k, dy = (my - transform.y) / transform.k;
    let best = -1, bestD = Infinity;
    const rad = 14 / transform.k;                       // min hit window in data space
    // prune by the LARGEST possible node radius, else a big bloomed bottle whose centre is
    // 14–56 px away gets pruned out and becomes unhoverable over its outer ring
    const prune = MAX_R / transform.k;
    quad.visit((node, qx0, qy0, qx1, qy1) => {
      if (!node.length) {
        do {
          const i = node.data;
          const d = (A.x[i] - dx) ** 2 + (A.y[i] - dy) ** 2;
          const hitR = Math.max(screenR(i, transform.k) / transform.k, rad * 0.5);
          if (d < hitR * hitR && d < bestD) { bestD = d; best = i; }
        } while ((node = node.next));
      }
      return qx0 > dx + prune || qx1 < dx - prune || qy0 > dy + prune || qy1 < dy - prune;
    });
    return best;
  }

  // ---------- the card ----------
  function tipHTML(i, withTwins) {
    const fam = famById.get(A.fam[i]);
    const notes = (A.notes[i] || "").split(",").map((s) => s.trim()).filter(Boolean);
    const yr = A.year[i] ? ` · ${A.year[i]}` : "";
    const g = A.meta.gender_labels[A.gender[i]] || "";
    let h = `<div class="hero">
        <div class="shot" style="background-image:url('${imgURL(A.pid[i])}')"></div>
        <div class="meta">
          <div class="nm">${esc(A.name[i])}</div>
          <div class="br">${esc(A.brand[i])}${yr}</div>
          <div class="stat"><span class="star">★</span> ${star(A.rating[i])}
            <span style="color:#cbb9a0">·</span> ${reviewsFmt(A.reviews[i])} reviews
            ${g ? `<span style="color:#cbb9a0">·</span> ${g}` : ""}</div>
          <div class="famtag"><span class="sw" style="background:${fam ? fam.color : "#bbb"}"></span>${esc(fam ? fam.name : "")}</div>
        </div>
      </div>`;
    if (notes.length) {
      h += `<div class="notes"><div class="lab">Notes</div>` +
        notes.slice(0, 8).map((n) => `<span class="n">${esc(n)}</span>`).join("") + `</div>`;
    }
    if (withTwins && NBR) {
      const nb = NBR.nbr[i], w = NBR.w[i];
      const rows = nb.map((j, t) => `
        <div class="twin" data-i="${j}">
          <div class="ts" style="background-image:url('${imgURL(A.pid[j])}')"></div>
          <div class="tw">
            <div class="tn">${esc(A.name[j])}</div>
            <div class="tb">${esc(A.brand[j])}</div>
            <div class="bar"><i style="width:${Math.round(w[t] * 100)}%"></i></div>
          </div>
        </div>`).join("");
      h += `<div class="twins"><div class="lab">Closest scent-twins</div>${rows}</div>`;
      h += `<a class="open" href="${fragURL(A.pid[i])}" target="_blank" rel="noopener">open on Fragrantica →</a>`;
    }
    return h;
  }

  // position the tip near an anchor, flipping/clamping to stay on screen
  function placeTip(ax, ay) {
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = ax + 18, y = ay - 20;
    if (x + tw > W - 10) x = ax - tw - 18;
    if (y + th > H - 10) y = H - th - 10;
    if (y < 10) y = 10;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function showTip(i, mx, my, pinned) {
    tip.innerHTML = tipHTML(i, pinned);
    tip.classList.toggle("pinned", !!pinned);
    tip.classList.add("show");
    placeTip(mx, my);
  }
  function hideTip() { tip.classList.remove("show", "pinned"); }

  // ---------- selection ----------
  function loadNeighbors() {
    if (NBR) return Promise.resolve();
    return fetch(NBR_URL).then((r) => r.json()).then((d) => { NBR = d; });
  }
  function select(i, center) {
    if (!(i >= 0)) return;
    selected = i;
    const want = i;                                     // guard the async card against a later (de)select
    loadNeighbors().then(() => {
      if (selected !== want) return;
      const sx = transform.applyX(A.x[i]), sy = transform.applyY(A.y[i]);
      showTip(i, center ? W / 2 + 30 : sx, center ? H / 2 - 40 : sy, true);
      requestDraw();
    });
    requestDraw();
  }
  function deselect() { selected = -1; hideTip(); requestDraw(); }

  // ---------- camera ----------
  let zoom;
  function fitAll(animate) {
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (let i = 0; i < N; i++) {
      if (A.x[i] < minx) minx = A.x[i]; if (A.x[i] > maxx) maxx = A.x[i];
      if (A.y[i] < miny) miny = A.y[i]; if (A.y[i] > maxy) maxy = A.y[i];
    }
    flyToBox(minx, miny, maxx, maxy, 0.92, animate);
  }
  function flyToBox(minx, miny, maxx, maxy, fill, animate) {
    const bw = Math.max(maxx - minx, 1), bh = Math.max(maxy - miny, 1);
    const k = Math.min(W / bw, H / bh) * fill;
    const cx = (minx + maxx) / 2, cy = (miny + maxy) / 2;
    const t = d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-cx, -cy);
    const sel = d3.select(canvas);
    (animate ? sel.transition().duration(750) : sel).call(zoom.transform, t);
  }
  function flyToNode(i) {
    if (!(i >= 0)) return;
    const k = Math.min(zoom.scaleExtent()[1], transform.k < 18 ? 26 : transform.k);
    const t = d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-A.x[i], -A.y[i]);
    d3.select(canvas).transition().duration(750).call(zoom.transform, t)
      .on("end", () => select(i, false));
  }

  // ---------- search ----------
  function setupSearch() {
    const inp = document.getElementById("search");
    const box = document.getElementById("results");
    let active = -1, hits = [];
    const render = () => {
      box.innerHTML = hits.map((i, n) => {
        const fam = famById.get(A.fam[i]);
        return `<div class="r${n === active ? " active" : ""}" role="option" id="opt${n}"
          aria-selected="${n === active}" data-i="${i}">
          <span class="dot" style="background:${fam ? fam.color : "#bbb"}"></span>
          <span><span class="nm">${esc(A.name[i])}</span> <span class="br">${esc(A.brand[i])}</span></span>
        </div>`;
      }).join("");
      inp.setAttribute("aria-expanded", hits.length ? "true" : "false");
      if (active >= 0) inp.setAttribute("aria-activedescendant", "opt" + active);
      else inp.removeAttribute("aria-activedescendant");
    };
    const search = (q) => {
      q = q.trim().toLowerCase();
      hits = []; active = -1;
      if (q.length < 2) { box.innerHTML = ""; return; }
      const scored = [];
      for (let i = 0; i < N; i++) {
        const nm = A.name[i].toLowerCase();
        let s = -1;
        if (nm === q) s = 1e9; else if (nm.startsWith(q)) s = 1e6 + A.reviews[i];
        else if (nm.includes(q)) s = 1e3 + A.reviews[i];
        else if ((A.brand[i] + " " + nm).includes(q)) s = A.reviews[i];
        if (s > 0) scored.push([s, i]);
      }
      scored.sort((a, b) => b[0] - a[0]);
      hits = scored.slice(0, 14).map((p) => p[1]);
      render();
    };
    inp.addEventListener("input", () => search(inp.value));
    inp.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { active = Math.min(active + 1, hits.length - 1); render(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); render(); e.preventDefault(); }
      else if (e.key === "Enter" && hits.length) { pick(hits[active < 0 ? 0 : active]); }
      else if (e.key === "Escape") { box.innerHTML = ""; inp.blur(); }
    });
    box.addEventListener("click", (e) => {
      const r = e.target.closest(".r"); if (r) pick(+r.dataset.i);
    });
    const pick = (i) => { box.innerHTML = ""; inp.value = A.name[i]; flyToNode(i); };
  }

  // ---------- legend ----------
  // hover previews a family (dims the rest); click *pins* that filter and flies to it (so the
  // affordance works on touch + keyboard, where there is no hover). Click again to release.
  function setupLegend() {
    const el = document.getElementById("legend");
    el.innerHTML = A.families.map((f) => `
      <button class="fam" type="button" data-id="${f.id}" aria-pressed="false"
              title="show only ${esc(f.name)}">
        <span class="sw" style="background:${f.color}"></span>
        <span class="nm">${esc(f.name)}</span>
        <span class="ct">${f.size.toLocaleString()}</span>
      </button>`).join("");
    el.querySelectorAll(".fam").forEach((node) => {
      const id = +node.dataset.id;
      node.addEventListener("mouseenter", () => { hoverFam = id; markDim(el, id); requestDraw(); });
      node.addEventListener("mouseleave", () => { hoverFam = pinnedFam; markDim(el, pinnedFam); requestDraw(); });
      node.addEventListener("click", () => {
        if (pinnedFam === id) { pinnedFam = hoverFam = -1; }
        else { pinnedFam = hoverFam = id; flyToFamily(id); }
        markDim(el, pinnedFam);
        el.querySelectorAll(".fam").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.id === pinnedFam)));
        requestDraw();
      });
    });
  }
  function setupHudToggle() {
    const hud = document.getElementById("hud"), btn = document.getElementById("hud-toggle");
    if (!btn) return;
    if (window.innerWidth <= 640) hud.classList.add("collapsed");   // map first on phones
    btn.addEventListener("click", () => {
      const open = hud.classList.toggle("collapsed") === false;
      btn.setAttribute("aria-expanded", String(open));
      btn.innerHTML = open ? "hide ▴" : "scent families &amp; more ▾";
    });
  }

  function markDim(el, id) {
    el.querySelectorAll(".fam").forEach((n) => n.classList.toggle("dim", id >= 0 && +n.dataset.id !== id));
  }
  function flyToFamily(id) {
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity, cnt = 0;
    for (let i = 0; i < N; i++) if (A.fam[i] === id) {
      cnt++; if (A.x[i] < minx) minx = A.x[i]; if (A.x[i] > maxx) maxx = A.x[i];
      if (A.y[i] < miny) miny = A.y[i]; if (A.y[i] > maxy) maxy = A.y[i];
    }
    if (cnt) flyToBox(minx, miny, maxx, maxy, 0.7, true);
  }

  // ---------- interactions ----------
  function setupCanvas() {
    zoom = d3.zoom().scaleExtent([0.4, 90])
      .on("start", () => canvas.classList.add("dragging"))
      .on("zoom", (e) => { transform = e.transform; if (tip.classList.contains("pinned")) repositionPinned(); requestDraw(); })
      .on("end", () => canvas.classList.remove("dragging"));
    d3.select(canvas).call(zoom).on("dblclick.zoom", null);

    canvas.addEventListener("mousemove", (e) => {
      if (canvas.classList.contains("dragging")) return;
      const i = nodeAt(e.clientX, e.clientY);
      canvas.classList.toggle("pointing", i >= 0);
      if (i === hover) { if (i >= 0 && !tip.classList.contains("pinned")) showTip(i, e.clientX, e.clientY, false); return; }
      hover = i;
      if (i >= 0) { if (!tip.classList.contains("pinned")) showTip(i, e.clientX, e.clientY, false); }
      else if (!tip.classList.contains("pinned")) hideTip();
      requestDraw();
    });
    canvas.addEventListener("mouseleave", () => { hover = -1; if (!tip.classList.contains("pinned")) hideTip(); requestDraw(); });
    canvas.addEventListener("click", (e) => {
      const i = nodeAt(e.clientX, e.clientY);
      if (i >= 0) select(i, false); else deselect();
    });
    // click a twin inside the pinned card -> jump to it
    tip.addEventListener("click", (e) => {
      const t = e.target.closest(".twin"); if (t) select(+t.dataset.i, false);
    });
    document.getElementById("reset").addEventListener("click", () => { deselect(); fitAll(true); });
  }
  function repositionPinned() {
    if (selected < 0) return;
    placeTip(transform.applyX(A.x[selected]), transform.applyY(A.y[selected]));
  }

  // ---------- boot ----------
  fetch(ATLAS_URL).then((r) => r.json()).then((data) => {
    A = data; N = A.x.length;
    A.families.forEach((f) => famById.set(f.id, f));
    // popularity -> data-space radius. The exponent sharpens the tail so only the few dozen
    // most-loved fragrances bloom into bottles at full zoom-out (the rest stay soft dots until
    // you zoom in), giving the map recognisable landmarks the moment it opens.
    let revMax = 0; for (let i = 0; i < N; i++) if (A.reviews[i] > revMax) revMax = A.reviews[i];
    const sMax = Math.sqrt(revMax + 1);
    dataR = new Float32Array(N);
    colorByNode = new Array(N);
    for (let i = 0; i < N; i++) {
      dataR[i] = 0.18 + 3.0 * Math.pow(Math.sqrt(A.reviews[i] + 1) / sMax, 1.4);
      const fam = famById.get(A.fam[i]);
      colorByNode[i] = fam ? fam.color : "#bbb";
    }
    quad = d3.quadtree().x((i) => A.x[i]).y((i) => A.y[i]).addAll(d3.range(N));

    document.getElementById("count").textContent = N.toLocaleString();
    setupCanvas(); setupSearch(); setupLegend(); setupHudToggle();
    window.addEventListener("resize", resize);
    resize(); fitAll(false);

    // fade the hint once the user starts exploring; device-appropriate copy for touch
    const hint = document.getElementById("hint");
    if (window.matchMedia && window.matchMedia("(hover: none)").matches)
      hint.textContent = "pinch to zoom · drag to pan · tap a bottle for its scent-twins";
    const fade = () => { hint.style.opacity = "0"; };
    canvas.addEventListener("wheel", fade, { once: true });
    canvas.addEventListener("mousedown", fade, { once: true });
    canvas.addEventListener("touchstart", fade, { once: true, passive: true });

    // expose a couple of "fun facts" landmarks for the intro links
    wireFacts();

    // small control surface (search-by-name, fly, select) — handy for the analyses page + testing
    window.PerfumeAtlas = {
      data: A,
      findByName: (q) => {
        const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
        q = norm(q);
        let best = -1, bestRev = -1;
        for (let i = 0; i < N; i++) if (norm(A.name[i]) === q && A.reviews[i] > bestRev) { best = i; bestRev = A.reviews[i]; }
        return best;
      },
      flyToNode, flyToFamily, fitAll, select, ready: true,
    };
    window.dispatchEvent(new Event("atlas:ready"));
  });

  function wireFacts() {
    const f = A.meta.facts;
    const byId = (id) => document.getElementById(id);
    const node = (id, idx) => {
      const el = byId(id); if (!el) return;
      el.textContent = A.name[idx];
      el.addEventListener("click", (e) => { e.preventDefault(); flyToNode(idx); });
    };
    node("fact-twin-a", f.twins.a); node("fact-twin-b", f.twins.b); node("fact-loner", f.loner.i);
    const bf = byId("fact-bigfam");
    if (bf) {
      const fam = famById.get(f.biggest_family);
      bf.textContent = fam ? fam.name : "the biggest family";
      bf.addEventListener("click", (e) => { e.preventDefault(); flyToFamily(f.biggest_family); });
    }
    const pct = byId("fact-twin-pct"); if (pct) pct.textContent = Math.round(f.twins.sim * 100) + "%";
  }
})();
