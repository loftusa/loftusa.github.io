/* assets/js/analyses-affiliations/same-rooms-no-paper.js — diff the rooms graph against the papers graph */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("same-rooms-no-paper", {
    prose: {
      intro:
        "<p>Two graphs live on the same 48 people. One is the <em>papers</em> graph — who has " +
        "shared a byline, straight from the publication record. The other is the <em>rooms</em> " +
        "graph — who has shared a lab, a degree program, an office, built from everyone's " +
        "careers. This panel diffs them. Most pairs who have shared a room have never shared a " +
        "paper, and that gap reads less like a verdict than like a list of collaborations still " +
        "on the table.</p>",
      how:
        "<p>Diffing two graphs on the same vertices is model diffing: same architecture, two " +
        "checkpoints, look where the weights disagree. Here the people are the architecture; " +
        "the rooms graph and the papers graph are the " +
        "checkpoints. Only about a quarter of the connected pairs appear in both graphs — " +
        "shared a room predicts surprisingly little about having shared a paper.</p>" +
        "<p>The dot columns then split the pairs by what kind of room it was, like reading the " +
        "diff layer by layer: share a lab and roughly 45% of pairs have a paper to show for it; " +
        "share only a campus or an online community and it is under 20%. The sister page's " +
        "<a href=\"/networks/analyses/#who-hasnt-met\">who-hasn't-met panel</a> is the " +
        "complementary lens — there, TF-IDF text similarity finds people who <em>write</em> " +
        "alike but never met; this one finds people who already met and never wrote.</p>",
      method:
        "<p>Edge-overlap statistics on a two-layer multiplex network (Kivelä, Arenas, Barthelemy, " +
        "Gleeson, Moreno &amp; Porter, “Multilayer networks,” J. Complex Networks 2014; Battiston, " +
        "Nicosia &amp; Latora, “Structural measures for multiplex networks,” Phys. Rev. E 2014); " +
        "agreement is the Jaccard index (Jaccard, 1901): the share of ever-connected pairs " +
        "present in both layers. Layer one is the direct co-authorship edges among the members (every indexed " +
        "co-authorship is present; one known proceedings-volume artifact is excluded in the graph " +
        "build). Layer two is the affiliation projection: a pair's weight sums room-type weights " +
        "(lab 3, program/company 2, community/university 1), and a shared lab <em>displaces</em> " +
        "the university that contains it — the nesting discount — so one shared room is never " +
        "double-counted as lab plus campus; a few weight-3 pairs are therefore stacks of smaller " +
        "rooms (company + campus) rather than labs. Honest caveats: the papers layer sees only " +
        "indexed publications, so work under review is invisible and an “open” pair may already " +
        "be closed; some open pairs involve a member with no indexed record " +
        "at all — they stay in the count but out of the named list; 3 of the 6 paper-without-room " +
        "pairs ride a single ~30-author position paper (“Open Problems in Mechanistic " +
        "Interpretability”); and for the 50 pairs in both graphs co-presence is not causation — " +
        "the room and the paper are often both downstream of the same advisor. Overlap years " +
        "are computed only where both people's stints carry dates.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, C = shared.colors;
      var BLUE = C.src.oa, GREEN = C.src.both, OCHRE = C.src.s2;  // shell palette, one source

      var totalW = el.clientWidth || 680;
      var miniW = 200, gap = 24;
      var wide = totalW >= 660;
      var W = Math.max(360, Math.min(wide ? totalW - miniW - gap : totalW, 640));

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", gap + "px").style("align-items", "flex-start");
      var left = wrap.append("div").style("flex", "0 0 " + W + "px").style("max-width", "100%");
      var right = wrap.append("div").style("flex", "0 0 " + miniW + "px");

      // ---- mini-map: ink deepens with how many of your rooms are still unconverted ----
      var counts = d.open_count_by_person;
      var maxOpen = 1;
      Object.keys(counts).forEach(function (k) { if (counts[k] > maxOpen) maxOpen = counts[k]; });
      var countOf = function (id) { return counts[id] || 0; };
      var baseOp = function (id) { return 0.25 + 0.75 * (countOf(id) / maxOpen); };
      var mapCircles = null;
      shared.minimap(right.node(), function (id) {
        return countOf(id) > 0 ? shared.colorOf(id) : null;
      }, {
        opacityFn: baseOp,
        onReady: function (api) { mapCircles = api.svg.selectAll("circle"); }
      });
      right.append("div").attr("class", "m-map-cap").style("max-width", miniW + "px")
        .text("ink deepens with open pairs — rooms shared, no paper yet. " +
          "Hover any dot or row to place both people.");

      function highlightSet(ids) {
        if (!mapCircles) return;
        mapCircles.interrupt();
        if (!ids) {
          mapCircles.attr("stroke", null)
            .attr("fill-opacity", function (n) { return baseOp(n.id); });
          return;
        }
        var set = {};
        ids.forEach(function (id) { set[id] = true; });
        mapCircles.attr("stroke", function (n) { return set[n.id] ? C.ink : null; })
          .attr("stroke-width", 1.4)
          .attr("fill-opacity", function (n) { return set[n.id] ? 1 : 0.15; });
      }

      // ---- (a) one segmented bar: the whole diff in 188 pair-widths ----
      var total = d.counts.open + d.counts.both + d.counts.inverse;
      var bx = d3.scaleLinear().domain([0, total]).range([0, W]);
      var barSvg = left.append("svg").attr("width", W).attr("height", 60);
      var segs = [
        { n: d.counts.open, c: BLUE },
        { n: d.counts.both, c: GREEN },
        { n: d.counts.inverse, c: OCHRE }
      ];
      var acc = 0;
      segs.forEach(function (s) { s.x0 = acc; acc += s.n; });
      barSvg.selectAll("rect").data(segs).join("rect")
        .attr("x", function (s) { return bx(s.x0); })
        .attr("y", 20).attr("height", 14)
        .attr("width", function (s) { return Math.max(1, bx(s.x0 + s.n) - bx(s.x0) - 1); })
        .attr("fill", function (s) { return s.c; });
      barSvg.append("text").attr("class", "m-label")
        .attr("x", 0).attr("y", 13).style("fill", BLUE)
        .text(d.counts.open + " rooms shared, no paper");
      barSvg.append("text").attr("class", "m-label")
        .attr("x", bx(segs[1].x0 + segs[1].n / 2)).attr("y", 13)
        .attr("text-anchor", "middle").style("fill", GREEN)
        .text(d.counts.both + " both");
      barSvg.append("text").attr("class", "m-label")
        .attr("x", W).attr("y", 48).attr("text-anchor", "end").style("fill", OCHRE)
        .text(d.counts.inverse + " — a paper, no room in common");
      barSvg.append("text").attr("class", "m-axis")
        .attr("x", 0).attr("y", 48)
        .text("all " + total + " pairs ever connected by either graph");

      // ---- (b) conversion by room type: open dots vs realized dots, per weight bin ----
      left.append("div").attr("class", "m-axis").style("color", C.muted)
        .style("margin", "10px 0 2px")
        .text("what kind of room — blue: still open, green: became a paper; % = conversion");

      var pairsByBin = {};
      d.pairs.forEach(function (p) {
        var k = p.kind + "|" + p.w;
        (pairsByBin[k] = pairsByBin[k] || []).push(p);   // payload order, no jitter
      });
      var groups = d.bins.map(function (b) {
        return {
          label: b.label, w: b.w,
          open: pairsByBin["open|" + b.w] || [],
          both: pairsByBin["both|" + b.w] || [],
          rate: b.both / (b.both + b.open)
        };
      });
      groups.push({ label: "paper, no room", inverse: d.inverse });

      var sp = 8.5, cols = 3, colsW = cols * sp, pairGap = 7;
      var rowsOf = function (n) { return Math.ceil(n / cols); };
      var maxRows = 1;
      groups.forEach(function (g) {
        var r = g.inverse ? rowsOf(g.inverse.length)
          : Math.max(rowsOf(g.open.length), rowsOf(g.both.length));
        if (r > maxRows) maxRows = r;
      });
      var topPad = 18, base = topPad + maxRows * sp, H = base + 34;
      var groupW = Math.max(68, Math.floor(W / groups.length));
      var dotSvg = left.append("svg").attr("width", W).attr("height", H);
      dotSvg.append("line").attr("x1", 0).attr("x2", W).attr("y1", base + 3).attr("y2", base + 3)
        .attr("stroke", C.hair).attr("stroke-width", 1);

      function pairTip(p) {
        var ov = p.overlap > 0
          ? ", " + p.overlap + " year" + (p.overlap === 1 ? "" : "s") + " overlapped" : "";
        var tail = p.kind === "open"
          ? "no joint paper indexed — still on the table"
          : p.n_papers + " joint paper" + (p.n_papers === 1 ? "" : "s");
        return "<strong>" + shared.esc(shared.labelOf(p.a)) + " × " +
          shared.esc(shared.labelOf(p.b)) + "</strong><br>" +
          "<span style='color:" + C.muted + "'>shared " +
          p.orgs.map(shared.esc).join(", ") + ov + " — </span>" + tail;
      }
      function invTip(v) {
        var rows = v.papers.map(function (t) {
          return "<div style='margin-top:2px'>" + shared.esc(t) + "</div>";
        }).join("");
        return "<strong>" + shared.esc(shared.labelOf(v.a)) + " × " +
          shared.esc(shared.labelOf(v.b)) + "</strong><br>" +
          "<span style='color:" + C.muted + "'>" + v.n_papers + " joint paper" +
          (v.n_papers === 1 ? "" : "s") + ", no room in common</span>" + rows;
      }
      function bindDot(sel, tip, who) {
        sel.style("cursor", "default")
          .on("mouseenter", function (evt, p) {
            highlightSet(who(p)); shared.tooltip.show(tip(p), evt);
          })
          .on("mousemove", function (evt, p) { shared.tooltip.show(tip(p), evt); })
          .on("mouseleave", function () { highlightSet(null); shared.tooltip.hide(); });
      }
      function stack(g, items, startX, fill, dashed, tip) {
        var dots = g.selectAll(null).data(items).join("circle")
          .attr("cx", function (p, i) { return startX + (i % cols) * sp + sp / 2; })
          .attr("cy", function (p, i) { return base - Math.floor(i / cols) * sp - sp / 2; })
          .attr("r", 3.1)
          .attr("fill", dashed ? C.bg : fill)
          .attr("stroke", dashed ? OCHRE : "transparent")
          .attr("stroke-width", dashed ? 1.2 : 4)
          .attr("stroke-dasharray", dashed ? "2,1.6" : null);
        bindDot(dots, tip, function (p) { return [p.a, p.b]; });
      }
      function twoLine(s) {
        if (s.length <= 9) return [s];
        var mid = Math.floor(s.length / 2), i = s.lastIndexOf(" ", mid);
        if (i < 0) i = s.indexOf(" ", mid);
        return i < 0 ? [s] : [s.slice(0, i), s.slice(i + 1)];
      }

      groups.forEach(function (g, gi) {
        var gx = gi * groupW + groupW / 2;
        var grp = dotSvg.append("g");
        if (g.inverse) {
          stack(grp, g.inverse, gx - colsW / 2, null, true, invTip);
        } else {
          stack(grp, g.open, gx - pairGap / 2 - colsW, BLUE, false, pairTip);
          stack(grp, g.both, gx + pairGap / 2, GREEN, false, pairTip);
          var r = Math.max(rowsOf(g.open.length), rowsOf(g.both.length));
          grp.append("text").attr("class", "m-label")
            .attr("x", gx).attr("y", base - r * sp - 6).attr("text-anchor", "middle")
            .style("fill", GREEN).text(shared.fmt.pct(g.rate));
        }
        twoLine(g.label).forEach(function (ln, li) {
          grp.append("text").attr("class", "m-axis")
            .attr("x", gx).attr("y", base + 16 + li * 12).attr("text-anchor", "middle")
            .style("fill", g.inverse ? OCHRE : C.muted).text(ln);
        });
      });

      left.append("div").attr("class", "m-axis").style("color", C.muted)
        .style("margin-top", "4px").style("max-width", "62ch").style("line-height", "1.5")
        .text("Each dot is one pair, anonymous until hovered — 132 names would be a wall, " +
          "not a chart. 13 of the open pairs involve a member with no indexed publication " +
          "record; they stay in the count but sit out the list below. And the record only " +
          "sees published work — an open pair may already have a paper in review.");

      // ---- (c) nominations: the open pairs with the most room behind them ----
      left.append("div").attr("class", "m-label")
        .style("color", C.ink).style("margin", "16px 0 2px")
        .text("Still on the table — the eight open pairs with the most shared room behind them");

      var rowsSel = [];
      d.nominations.forEach(function (nm, ni) {
        var row = left.append("div")
          .style("display", "flex").style("align-items", "baseline").style("gap", "10px")
          .style("padding", "4px 0")
          .style("border-top", "1px solid " + (ni === 0 ? C.hair : "#efe9dc"));
        rowsSel.push(row);
        var names = row.append("div").style("flex", "1 1 auto")
          .style("display", "flex").style("flex-wrap", "wrap")
          .style("gap", "3px 8px").style("align-items", "center");
        [nm.a, nm.b].forEach(function (id, i) {
          if (i === 1) names.append("span").attr("class", "m-axis").style("color", C.muted).text("×");
          var chip = names.append("span")
            .style("display", "inline-flex").style("align-items", "center")
            .style("gap", "4px").style("white-space", "nowrap");
          chip.append("span")
            .style("width", "7px").style("height", "7px").style("border-radius", "50%")
            .style("background", shared.colorOf(id)).style("flex", "0 0 auto");
          chip.append("span").attr("class", "m-label").style("color", C.ink)
            .html(shared.seatLink(id));
        });
        row.append("div").attr("class", "m-axis").style("color", C.muted)
          .style("flex", "0 1 auto").style("text-align", "right")
          .text(nm.orgs.join(", ") + " · " + nm.overlap + " yr" +
            (nm.overlap === 1 ? "" : "s") + " overlapped");
        var tip = function () {
          return "<strong>" + shared.esc(shared.labelOf(nm.a)) + " × " +
            shared.esc(shared.labelOf(nm.b)) + "</strong><br>" +
            "<span style='color:" + C.muted + "'>shared " + nm.orgs.map(shared.esc).join(", ") +
            ", " + nm.overlap + " year" + (nm.overlap === 1 ? "" : "s") +
            " overlapped — </span>no joint paper indexed, still on the table";
        };
        row.on("mouseenter", function (evt) {
          rowsSel.forEach(function (r, j) { r.style("opacity", j === ni ? 1 : 0.45); });
          highlightSet([nm.a, nm.b]); shared.tooltip.show(tip(), evt);
        }).on("mousemove", function (evt) { shared.tooltip.show(tip(), evt); })
          .on("mouseleave", function () {
            rowsSel.forEach(function (r) { r.style("opacity", 1); });
            highlightSet(null); shared.tooltip.hide();
          });
      });

      left.append("div").attr("class", "m-axis").style("color", C.muted)
        .style("margin-top", "6px").style("max-width", "62ch").style("line-height", "1.5")
        .text("Ranked by how much room a pair shared, then by years of overlap, among the " +
          d.counts.open_nameable + " open pairs where both people have an indexed record. " +
          "The site's author appears twice in eight — build the matchmaking table, become " +
          "its output.");
    }
  });
})();
