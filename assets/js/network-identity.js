/* network-identity.js — site-entry identity module for the /networks/* pages.
   First visit: "Who are you?" picker → personalized defaults + a quick self-data check.
   Standalone IIFE, no deps. Must be loaded BEFORE the page scripts (eval-time deep-linking).
   Storage: localStorage "network_identity" {id,label} / "network_identity_dismissed" "1". */
(function () {
  "use strict";

  var API_BASE = location.hostname === "localhost" || location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://llm-resume-restless-thunder-9259.fly.dev";
  var ID_KEY = "network_identity";
  var DISMISS_KEY = "network_identity_dismissed";

  var path = location.pathname.endsWith("/") ? location.pathname : location.pathname + "/";
  var ON_AFF_ANALYSES = path === "/networks/affiliations/analyses/";
  var ON_AFF_MAP = path === "/networks/affiliations/";

  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }
  function getIdentity() {
    try { var v = JSON.parse(lsGet(ID_KEY)); return v && v.id ? v : null; } catch (e) { return null; }
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function sig() { return AbortSignal.timeout ? AbortSignal.timeout(4000) : undefined; }

  /* ---- personalized default for the current page (eval-time on load; live after a pick) ---- */
  function applyDefault(id, live) {
    if (location.hash || /[?&](p|edit)=/.test(location.search)) return; // never clobber deep links
    var p = encodeURIComponent(id);
    if (ON_AFF_ANALYSES) {
      history.replaceState(null, "", "?p=" + p + (live ? "" : "#your-seat"));
      if (live) location.hash = "your-seat"; // fires popstate → the analyses shell activates it
    } else if (ON_AFF_MAP) {
      location.hash = "p=" + p; // the map reads #p= at init (eval time is before its init)
    }
  }
  var boot = getIdentity();
  if (boot) applyDefault(boot.id, false);

  /* ---- styles (namespaced; matches the pages' Tufte palette) ---- */
  var CSS = [
    ".nid-scrim{position:fixed;inset:0;background:rgba(43,43,43,.25);z-index:2000;display:flex;align-items:center;justify-content:center;}",
    ".nid-card{width:340px;max-width:calc(100vw - 32px);box-sizing:border-box;background:#faf8f3;border:1px solid #e3ddcf;border-radius:6px;padding:18px;color:#2b2b2b;font-size:13px;line-height:1.4;}",
    ".nid-title{font-size:16px;font-weight:600;margin:0 0 4px;}",
    ".nid-sub{color:#8c867b;font-size:12.5px;margin:0 0 10px;}",
    ".nid-input{font-family:inherit;font-size:13px;color:#2b2b2b;width:100%;box-sizing:border-box;background:#fff;border:1px solid #ddd6c8;border-radius:4px;padding:6px 9px;outline:none;}",
    ".nid-input:focus{border-color:#9a958c;}",
    ".nid-list{max-height:240px;overflow-y:auto;margin:8px 0 12px;padding:0;list-style:none;}",
    ".nid-list li{padding:4px 6px;border-radius:4px;cursor:pointer;}",
    ".nid-list li:hover,.nid-list li.on{background:#eae5d8;}",
    ".nid-list .nid-none,.nid-list .nid-none:hover{color:#8c867b;cursor:default;background:none;}",
    ".nid-links{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;font-size:12px;}",
    ".nid-links a{color:#8c867b;text-decoration:none;cursor:pointer;}",
    ".nid-links a:hover{color:#2b2b2b;}",
    ".nid-rows{max-height:180px;overflow-y:auto;margin:10px 0 0;font-size:12px;}",
    ".nid-rows.nid-muted,.nid-row .nid-meta{color:#6b665d;}",
    ".nid-row{margin:0 0 6px;line-height:1.35;}",
    ".nid-row .nid-org{font-weight:600;}",
    ".nid-pills{display:flex;gap:6px;margin-top:14px;}",
    ".nid-pill{flex:1 1 0;font-family:inherit;font-size:12px;padding:4px 0;text-align:center;border-radius:999px;border:1px solid #ddd6c8;background:#fff;color:#6b665d;cursor:pointer;}",
    ".nid-pill:hover{background:#eae5d8;}",
    ".nid-pill.nid-yes{background:#e2ebf2;border-color:#b9cbdc;color:#2b2b2b;font-weight:600;}",
    ".nid-chip{color:#8c867b;margin:0 0 6px;}",
    ".nid-chip a{color:#6b665d;text-decoration:underline;cursor:pointer;}",
  ].join("\n");
  function injectStyle() {
    if (document.getElementById("nid-style") || !document.head) return;
    var st = document.createElement("style");
    st.id = "nid-style"; st.textContent = CSS;
    document.head.appendChild(st);
  }

  /* ---- sidebar chip ---- */
  function renderChip(ident) {
    var foot = document.querySelector("#sidebar #foot");
    if (!foot) return;
    injectStyle();
    var old = foot.querySelector(".nid-chip"); if (old) old.remove();
    var div = document.createElement("div");
    div.className = "nid-chip";
    div.innerHTML = "you are " + esc(ident.label) + ' · <a class="nid-switch">switch</a>' +
      ' · <a class="nid-add" href="/networks/affiliations/?edit=1&join=1">add someone</a>';
    div.querySelector(".nid-switch").addEventListener("click", function (e) {
      e.preventDefault();
      openPicker(ident.label);
    });
    foot.prepend(div);
  }

  /* ---- roster (48 names) ---- */
  var names = null;
  function loadNames() {
    if (names) return Promise.resolve(names);
    return fetch("/assets/data/analyses-affiliations/shared.json", { signal: sig() })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (d) {
        names = Object.entries(d.nodes || {}).map(function (kv) {
          return { id: kv[0], label: (kv[1] && kv[1].label) || kv[0] };
        }).sort(function (a, b) { return a.label.localeCompare(b.label); });
        return names;
      });
  }

  /* ---- picker overlay ---- */
  var overlay = null;
  function closeOverlay() { if (overlay) { overlay.remove(); overlay = null; } }
  function openPicker(prefill) {
    if (overlay || !document.body) return;
    injectStyle();
    // if the roster fetch fails, the picker silently doesn't show — never block the page
    loadNames().then(function (list) { buildPicker(list, prefill || ""); }).catch(function () {});
  }

  function buildPicker(list, prefill) {
    overlay = document.createElement("div");
    overlay.className = "nid-scrim";
    overlay.innerHTML =
      '<div class="nid-card" role="dialog" aria-label="Who are you?">' +
      '<div class="nid-title">Who are you?</div>' +
      '<div class="nid-sub">Pick your name and this map opens to your seat. (No login &mdash; this just personalizes the view.)</div>' +
      '<input class="nid-input" type="search" placeholder="find your name&hellip;" autofocus autocomplete="off" spellcheck="false">' +
      '<ul class="nid-list"></ul>' +
      '<div class="nid-links"><a class="nid-browse">just browsing &rarr;</a>' +
      '<a href="/networks/affiliations/?edit=1&amp;join=1">I&#39;m new here &mdash; join the map &rarr;</a>' +
      '<a href="/networks/affiliations/?edit=1&amp;join=1">add someone else &rarr;</a></div></div>';
    document.body.appendChild(overlay);
    var input = overlay.querySelector(".nid-input");
    var ul = overlay.querySelector(".nid-list");
    var shown = list, active = -1;

    function renderList() {
      var q = input.value.trim().toLowerCase();
      shown = q ? list.filter(function (n) { return n.label.toLowerCase().indexOf(q) !== -1; }) : list;
      if (active >= shown.length) active = shown.length - 1;
      ul.innerHTML = shown.map(function (n, i) {
        return '<li data-i="' + i + '"' + (i === active ? ' class="on"' : "") + ">" + esc(n.label) + "</li>";
      }).join("") || '<li class="nid-none">no match &mdash; try fewer letters</li>';
    }
    ul.addEventListener("click", function (e) {
      var li = e.target.closest("li[data-i]");
      if (li) pick(shown[+li.dataset.i]);
    });
    input.addEventListener("input", function () { active = -1; renderList(); });
    overlay.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeOverlay(); // no dismissed flag — ask again next visit
      else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && shown.length) {
        e.preventDefault();
        active = (active + (e.key === "ArrowDown" ? 1 : -1) + shown.length) % shown.length;
        renderList();
        var on = ul.querySelector(".on");
        if (on) on.scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter" && shown.length) pick(shown[active === -1 ? 0 : active]);
    });
    overlay.querySelector(".nid-browse").addEventListener("click", function () {
      lsSet(DISMISS_KEY, "1"); closeOverlay();
    });
    input.value = prefill;
    renderList();
    input.focus();
  }

  /* ---- pick → info check ("here's what the map has for you") ---- */
  function pick(n) {
    if (!n || !overlay) return;
    lsSet(ID_KEY, JSON.stringify({ id: n.id, label: n.label }));
    var card = overlay.querySelector(".nid-card");
    card.innerHTML =
      '<div class="nid-title">Here&#39;s what the map has for ' + esc(n.label) + ":</div>" +
      '<div class="nid-sub nid-city"></div><div class="nid-rows nid-muted">loading&hellip;</div>' +
      '<div class="nid-pills"><button type="button" class="nid-pill nid-yes">looks right &#10003;</button>' +
      '<button type="button" class="nid-pill nid-fix">fix something &rarr;</button></div>';
    fetch("/assets/data/affiliations.json", { signal: sig() })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (d) { renderInfo(card, d, n); })
      .catch(function () {
        card.querySelector(".nid-rows").textContent = "couldn't load your row — you can still continue.";
      });
    card.querySelector(".nid-yes").addEventListener("click", function () {
      fetch(API_BASE + "/affiliations/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "aff_confirm", payload: { person: n.id }, editor: n.label }),
        signal: sig(),
      }).catch(function () {}); // fire-and-forget
      closeOverlay();
      renderChip(n);
      applyDefault(n.id, true);
    });
    card.querySelector(".nid-fix").addEventListener("click", function () {
      location.href = "/networks/affiliations/?edit=1&p=" + encodeURIComponent(n.id);
    });
  }

  function renderInfo(card, d, n) {
    var person = (d.people || []).find(function (p) { return p.id === n.id; });
    var links = (d.links || []).filter(function (l) { return l.person === n.id; });
    var orgLabel = {};
    (d.orgs || []).forEach(function (o) { orgLabel[o.id] = o.label; });
    var city = card.querySelector(".nid-city"), rows = card.querySelector(".nid-rows");
    if (person && person.city) city.textContent = person.city;
    rows.classList.remove("nid-muted");
    rows.innerHTML = links.map(function (l) {
      return '<div class="nid-row"><span class="nid-org">' + esc(orgLabel[l.org] || l.org) + "</span>" +
        ' <span class="nid-meta">&mdash; ' + esc(l.role || "") + " &middot; " + esc(l.years || "") + "</span></div>";
    }).join("") || '<div class="nid-row nid-meta">no career rows on the map yet.</div>';
  }

  /* ---- boot ---- */
  function init() {
    var ident = getIdentity();
    if (ident) renderChip(ident);
    else if (lsGet(DISMISS_KEY) !== "1") openPicker("");
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.NetworkIdentity = { get: getIdentity, open: openPicker, API_BASE: API_BASE };
})();
