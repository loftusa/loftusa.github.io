/* assets/js/analyses/bands-not-duets.js — recurring author-set mining: the standing teams */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("bands-not-duets", {
    prose: {
      intro:
        "<p>A co-authorship edge only remembers that two people once shared a byline. But a " +
        "paper isn’t a pair — it’s a whole roster, a set of people in a room. This panel asks " +
        "which exact <em>sets</em> keep recurring: not who collaborates, but who keeps showing " +
        "up together as a unit. Those are the standing bands behind the duets.</p>",
      how:
        "<p>An edge list is a lossy projection: it shatters every paper into pairwise links, so a " +
        "five-person team and a chain of five strangers can leave identical edges. A paper is " +
        "really a <em>hyperedge</em> — a set, not a pair — the same distinction as a circuit of " +
        "attention heads that only does its job together versus five heads you happened to ablate " +
        "one at a time.</p>" +
        "<p>To recover the sets, we mine for frequent itemsets: treat each paper’s author roster " +
        "as a basket and look for the exact member-combinations that appear on two or more " +
        "papers — like finding the repeated motif across many forward passes instead of reading " +
        "single activations. We allow a run to absorb a paper that adds at most one guest to the " +
        "core, so a standing trio that occasionally brings a fourth still reads as one band. Rank " +
        "the survivors by how many papers they share, and the longest-running circuits in the lab " +
        "fall out — teams the pairwise graph literally cannot represent.</p>",
      method:
        "<p>Frequent author-set (itemset) mining over the paper hypergraph — the basket-analysis " +
        "idea of Agrawal &amp; Srikant, “Fast Algorithms for Mining Association Rules,” VLDB 1994, " +
        "applied to co-authorship rather than supermarket carts. Treating papers as hyperedges and " +
        "asking which higher-order groups recur follows Benson, Abebe, Schaub, Jadbabaie &amp; " +
        "Kleinberg, “Simplicial closure and higher-order link prediction,” PNAS 2018 — the case " +
        "that pairwise edges discard real structure. Here: papers with 3–10 listed members " +
        "(skipping >25-author mega-papers); a <em>band</em> is an exact member-set on ≥2 distinct " +
        "papers; a run absorbs any paper that is a superset adding ≤1 person (the occasional " +
        "guest). Bands are ranked by run length, then year span; the top 10 are shown. Caveats: " +
        "the two source databases sometimes split one person into two spellings, so co-listed " +
        "aliases (e.g. “J. Vogelstein” / “Joshua T Vogelstein”) were merged to one node before " +
        "mining — a database artifact, not anyone’s doing; eLife “Author response” entries were " +
        "folded into the paper they review so a single work isn’t double-counted; and bands can " +
        "overlap, since a trio and the quartet it sometimes becomes are both real teams.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, bands = d.bands, C = shared.colors;
      var nb = bands.length;
      var yLo = d.years[0], yHi = d.years[1];

      var totalW = el.clientWidth || 680;
      var miniW = 200, gap = 24;
      var wide = totalW >= 660;
      var W = Math.max(360, Math.min(wide ? totalW - miniW - gap : totalW, 720));

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", gap + "px").style("align-items", "flex-start");
      // left is FIXED width W so the per-row barcodes and the shared axis stay aligned
      // (a flex-grown column would drift its internal columns out from under the axis ticks).
      var left = wrap.append("div").style("flex", "0 0 " + W + "px").style("max-width", "100%");
      var right = wrap.append("div").style("flex", "0 0 " + miniW + "px");

      // ---- column geometry (one HTML row per band; barcode + axis share one x-scale) ----
      var runW = 58;                              // "8 papers" label column
      var barW = Math.max(150, Math.round(W * 0.42));
      var namesW = W - barW - runW - 16;
      var pad = 8;
      var x = d3.scaleLinear().domain([yLo - 0.5, yHi + 0.5]).range([pad, barW - pad]);
      var maxN = d3.max(bands, function (b) { return b.n; });
      var inkOf = function (n) { return 0.4 + 0.6 * Math.sqrt(n / maxN); };

      // which nodes belong to any band / to band #1 (the anchor)
      var inBand = {}, lead = {};
      bands.forEach(function (b, i) {
        b.members.forEach(function (id) { inBand[id] = true; if (i === 0) lead[id] = true; });
      });

      // header strip aligned to the two data columns
      var head = left.append("div")
        .style("display", "flex").style("align-items", "baseline")
        .style("gap", "8px").style("margin-bottom", "2px");
      head.append("div").attr("class", "m-axis").style("flex", "0 0 " + namesW + "px")
        .style("color", C.muted).text("the standing teams");
      head.append("div").attr("class", "m-axis").style("flex", "0 0 " + barW + "px")
        .style("color", C.muted).text("one tick per paper, " + yLo + "–" + yHi);
      head.append("div").attr("class", "m-axis").style("flex", "0 0 " + runW + "px")
        .style("text-align", "right").style("color", C.muted).text("run");

      var rowsSel = [];   // per-row d3 selections for hover dimming

      bands.forEach(function (b, bi) {
        var row = left.append("div")
          .style("display", "flex").style("align-items", "center").style("gap", "8px")
          .style("padding", "5px 0")
          .style("border-top", "1px solid " + (bi === 0 ? C.hair : "#efe9dc"));
        rowsSel.push(row);

        // -- members: lab-colored dots + names, flex-wrap so any band size fits --
        var names = row.append("div")
          .style("flex", "0 0 " + namesW + "px").style("display", "flex")
          .style("flex-wrap", "wrap").style("gap", "3px 9px").style("align-items", "center");
        b.members.forEach(function (id) {
          var col = shared.colorOf(id);
          var chip = names.append("span")
            .style("display", "inline-flex").style("align-items", "center").style("gap", "4px")
            .style("white-space", "nowrap");
          chip.append("span")
            .style("width", "7px").style("height", "7px").style("border-radius", "50%")
            .style("background", col).style("opacity", bi === 0 ? 1 : 0.85)
            .style("flex", "0 0 auto");
          chip.append("span").attr("class", "m-label")
            .style("color", C.ink).style("opacity", bi === 0 ? 1 : 0.82)
            .html(shared.seatLink(id));
        });

        // -- barcode: one tick per paper on the shared year axis --
        var rh = 22;
        var svg = row.append("div").style("flex", "0 0 " + barW + "px")
          .append("svg").attr("width", barW).attr("height", rh);
        svg.append("line")                       // hairline year baseline
          .attr("x1", pad).attr("x2", barW - pad).attr("y1", rh - 5).attr("y2", rh - 5)
          .attr("stroke", C.hair).attr("stroke-width", 1);
        // spread same-year papers so ticks don't fully overlap
        var byYear = {};
        b.papers.forEach(function (p) { (byYear[p.year] = byYear[p.year] || []).push(p); });
        var placed = [];
        b.papers.forEach(function (p) {
          var sib = byYear[p.year], k = sib.indexOf(p), m = sib.length;
          var jit = m > 1 ? (k - (m - 1) / 2) * 3.2 : 0;
          placed.push({ p: p, cx: x(p.year) + jit });
        });
        var op = inkOf(b.n);
        svg.selectAll("line.t").data(placed).join("line").attr("class", "t")
          .attr("x1", function (t) { return t.cx; }).attr("x2", function (t) { return t.cx; })
          .attr("y1", 4).attr("y2", rh - 5)
          .attr("stroke", function (t) { return shared.colorOf(b.members[0]); })
          .attr("stroke-width", 2)
          .attr("stroke-opacity", function (t) { return t.p.guest ? 0.45 : op; });

        // -- run length, direct-labeled --
        row.append("div").attr("class", "m-axis")
          .style("flex", "0 0 " + runW + "px").style("text-align", "right")
          .style("color", bi === 0 ? C.ink : C.muted)
          .html("<strong>" + b.n + "</strong> paper" + (b.n === 1 ? "" : "s"));

        // -- hover: dim siblings, light the band on the mini-map, list the run --
        row.on("mouseenter", function (evt) {
          rowsSel.forEach(function (r, j) { r.style("opacity", j === bi ? 1 : 0.4); });
          highlightSet(b.members);
          shared.tooltip.show(tipHtml(b), evt);
        }).on("mousemove", function (evt) {
          shared.tooltip.show(tipHtml(b), evt);
        }).on("mouseleave", function () {
          rowsSel.forEach(function (r) { r.style("opacity", 1); });
          highlightSet(null);
          shared.tooltip.hide();
        });
      });

      // ---- shared year axis under the barcode column ----
      var axW = namesW + 8 + barW;
      var axSvg = left.append("div").style("margin-top", "2px")
        .append("svg").attr("width", axW).attr("height", 16);
      var x0 = namesW + 8;
      var ticks = [];
      for (var yr = yLo; yr <= yHi; yr++) if (yr % 2 === 0 || yr === yHi) ticks.push(yr);
      axSvg.selectAll("text").data(ticks).join("text").attr("class", "m-axis")
        .attr("x", function (yr) { return x0 + x(yr); }).attr("y", 11)
        .attr("text-anchor", "middle").attr("fill", C.muted)
        .text(function (yr) { return "’" + String(yr).slice(2); });

      var labNote = d.single_lab === d.n_bands
        ? "Every one of the <strong>" + d.n_bands + "</strong> standing bands sits inside a single " +
          "lab — recurring teams form <em>within</em> groups, not across them. "
        : "<strong>" + d.single_lab + "</strong> of " + d.n_bands + " standing bands sit inside a " +
          "single lab — recurring teams mostly form <em>within</em> groups. ";
      left.append("div").attr("class", "m-note").style("margin-top", "8px")
        .html(labNote + "A faint tick marks a paper where the core brought one guest; " +
          "darker ink runs longer.");

      // ---- mini-map: band people in lab color, band #1 at full ink ----
      var mini = shared.minimap(right.node(), function (id) {
        return inBand[id] ? shared.colorOf(id) : null;
      }, {
        opacityFn: function (id) { return lead[id] ? 1 : (inBand[id] ? 0.55 : 0.16); },
        radiusFn: function (id) { return lead[id] ? 3.6 : (inBand[id] ? 2.9 : 2); }
      });

      // highlight an arbitrary SET on the mini-map (the built-in highlight is single-id)
      function highlightSet(ids) {
        var circles = mini.svg.selectAll("circle");
        if (!circles.size()) return;
        circles.interrupt();
        if (!ids) {
          circles
            .attr("stroke", null)
            .attr("fill-opacity", function (n) { return lead[n.id] ? 1 : (inBand[n.id] ? 0.55 : 0.16); });
          return;
        }
        var set = {};
        ids.forEach(function (id) { set[id] = true; });
        circles
          .attr("stroke", function (n) { return set[n.id] ? C.ink : null; })
          .attr("stroke-width", 1.4)
          .attr("fill-opacity", function (n) { return set[n.id] ? 1 : 0.18; });
      }

      right.append("div").attr("class", "m-note").style("margin-top", "6px")
        .html("Band #1 — the longest-running team — held at full ink; every other band’s " +
          "members fade in. Hover a row to light up that team on the map.");

      function tipHtml(b) {
        var who = b.members.map(function (id) { return shared.esc(shared.labelOf(id)); }).join(" · ");
        var rows = b.papers.map(function (p) {
          var g = p.guest ? " <span style='color:" + C.muted + "'>+ " + shared.esc(p.guest) + "</span>" : "";
          return "<div style='margin-top:2px'><span style='color:" + C.muted + "'>" + p.year +
            "</span> &nbsp;" + shared.esc(p.title) + g + "</div>";
        }).join("");
        return "<strong>" + who + "</strong><br>" +
          "<span style='color:" + C.muted + "'>" + b.n + " papers, " +
          b.span[0] + "–" + b.span[1] + " · exact set on " + b.exact + "</span>" + rows;
      }
    }
  });
})();
