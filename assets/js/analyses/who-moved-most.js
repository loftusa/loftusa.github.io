/* assets/js/analyses/who-moved-most.js — OMNI temporal drift: who moved the most, and when */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("who-moved-most", {
    prose: {
      intro:
        "<p>Some people occupy the same corner of this network for years; others arrive at the " +
        "edge and end up near the middle. This panel gives every person one position in the " +
        "network for each year since 2019, measures how far that position moves year over year, " +
        "and ranks the movers — with each person’s breakout year, the single biggest jump, " +
        "marked in full ink.</p>",
      how:
        "<p>“Position per year” has a catch: embed each year’s graph separately and " +
        "the coordinates land in different spaces — like comparing activations across two training " +
        "seeds, where the axes mean different things and cross-run distances are noise. The omnibus " +
        "embedding (OMNI) is the alignment trick: stack all eight yearly graphs into one big block " +
        "matrix and decompose it <em>jointly</em>, so every person gets one coordinate per year in " +
        "a single shared space — the same move as aligning latent spaces across model checkpoints " +
        "so you can watch a representation evolve.</p>" +
        "<p>In that shared space, movement is signal: a person’s drift in a year is the " +
        "distance between this year’s position and last year’s, total drift sums those, " +
        "and the breakout year is the largest single jump. One honest wrinkle: positions here grow " +
        "with collaboration volume, so a big jump can mean a changed neighborhood, a publishing " +
        "burst with the same collaborators, or both — the tie counts in each tooltip tell you " +
        "which.</p>",
      method:
        "<p>Omnibus embedding: Levin, Athreya, Tang, Lyzinski &amp; Priebe, “A central limit " +
        "theorem for an omnibus embedding of multiple random graphs”, IEEE ICDM Workshops 2017, " +
        "via graspologic’s <code>OmnibusEmbed</code> (4 components, deterministic full SVD) on " +
        "cumulative co-authorship-weighted adjacencies for 2019–2026 over a fixed 112-vertex set " +
        "(pre-2019 cumulative graphs have ≤ 13 edges and were dropped). Edge weights use " +
        "fractional co-authorship counting — each paper adds 1/n_authors to each of its " +
        "co-author pairs (suggested by Stella Biderman) — so a small-team paper deepens a tie " +
        "far more than a 30-author one. Drift is the Euclidean " +
        "displacement between consecutive latent positions; the 35 roster members with at least one " +
        "recorded co-authorship are ranked. And a treat: one of the method’s inventors is a node " +
        "on this very map — Carey E. Priebe sits in the Vogelstein cluster, so OMNI is " +
        "here measuring the orbit of its own co-creator’s collaborators. Caveats: cumulative " +
        "graphs only accrete edges, so drift captures growth and re-orientation, never churn; " +
        "latent-position norm scales with weighted degree, so long-running prolific collaborations " +
        "(the Vogelstein side especially) drift farthest; and “new co-author ties” are " +
        "counted at the node level — the source databases occasionally split one person into two " +
        "spellings, so a tie count can overstate distinct humans by one. That is the databases’ " +
        "doing, not anyone’s publication record.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, movers = d.movers, C = shared.colors;
      var k = d.drift_years.length;
      var n = movers.length;
      var totalW = el.clientWidth || 640;
      var miniW = 200, gap = 24;
      var wide = totalW >= 640;
      var W = Math.max(340, Math.min(wide ? totalW - miniW - gap : totalW, 760));

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", gap + "px").style("align-items", "flex-start");
      var left = wrap.append("div").style("flex", "1 1 " + W + "px").style("min-width", "320px");
      var right = wrap.append("div").style("flex", "0 0 " + miniW + "px");

      // ---- geometry: [names | dot ranking | per-year bar sparkline] ----
      var rowH = 30, topPad = 54;
      var H = topPad + n * rowH + 20;
      var gutter = Math.min(150, Math.max(118, Math.round(W * 0.32)));
      var sparkW = 110, sparkX0 = W - sparkW - 8, sparkX1 = W - 8;
      var dotX0 = gutter + 10, dotX1 = sparkX0 - 58;
      var maxTotal = d3.max(movers, function (m) { return m.total; });
      var x = d3.scaleLinear().domain([0, maxTotal]).range([dotX0, dotX1]);
      var step = (sparkX1 - sparkX0) / k, barW = Math.max(3, step - 3);
      var rowY = function (i) { return topPad + i * rowH + rowH / 2; };
      var inkOp = function (m) { return 0.45 + 0.55 * Math.sqrt(m.total / maxTotal); };

      var svg = left.append("svg").attr("width", W).attr("height", H);

      // direct column headers, no legend
      svg.append("text").attr("class", "m-axis").attr("x", dotX0).attr("y", 14)
        .attr("fill", C.muted)
        .text("total drift, " + d.years[0] + "→" + d.years[d.years.length - 1]);
      svg.append("text").attr("class", "m-axis").attr("x", sparkX1).attr("y", 14)
        .attr("text-anchor", "end").attr("fill", C.muted)
        .text("drift per year");

      // one hairline at zero is the only axis the dot plot needs
      svg.append("line")
        .attr("x1", dotX0).attr("x2", dotX0)
        .attr("y1", topPad - 4).attr("y2", topPad + n * rowH + 2)
        .attr("stroke", C.hair).attr("stroke-width", 1);

      // ---- rows ----
      var rows = svg.append("g").selectAll("g").data(movers).join("g");

      rows.append("text").attr("class", "m-label")
        .attr("x", gutter).attr("y", function (m, i) { return rowY(i) + 4; })
        .attr("text-anchor", "end").attr("fill", C.ink)
        .text(function (m) { return shared.labelOf(m.id); });

      rows.append("line")
        .attr("x1", dotX0).attr("x2", function (m) { return x(m.total); })
        .attr("y1", function (m, i) { return rowY(i); })
        .attr("y2", function (m, i) { return rowY(i); })
        .attr("stroke", C.hair).attr("stroke-width", 1);

      rows.append("circle")
        .attr("cx", function (m) { return x(m.total); })
        .attr("cy", function (m, i) { return rowY(i); })
        .attr("r", 4.5)
        .attr("fill", function (m) { return shared.colorOf(m.id); })
        .attr("fill-opacity", inkOp);

      rows.append("text").attr("class", "m-axis")
        .attr("x", function (m) { return x(m.total) + 8; })
        .attr("y", function (m, i) { return rowY(i) + 4; })
        .attr("fill", C.muted)
        .text(function (m) { return shared.fmt.sig(m.total, 2); });

      // ---- per-year drift as mini bars, breakout year at full ink ----
      var barX = function (t) { return sparkX0 + t * step; };
      rows.each(function (m, i) {
        var g = d3.select(this);
        var base = rowY(i) + 9, hMax = 17;
        var ymax = d3.max(m.per_year, function (p) { return p.d; }) || 1e-9;
        var color = shared.colorOf(m.id);
        g.selectAll(null).data(m.per_year).join("rect")
          .attr("x", function (p, t) { return barX(t); })
          .attr("width", barW)
          .attr("y", function (p) { return base - Math.max(0.5, (p.d / ymax) * hMax); })
          .attr("height", function (p) { return Math.max(0.5, (p.d / ymax) * hMax); })
          .attr("fill", color)
          .attr("fill-opacity", function (p) { return p.year === m.breakout ? 1 : 0.28; });
      });

      // annotation on the top row: what the breakout actually was
      var b0 = movers[0].per_year.findIndex(function (p) { return p.year === movers[0].breakout; });
      if (d.annotation && W >= 480) {
        var bx = barX(b0) + barW / 2;
        svg.append("text").attr("class", "m-axis")
          .attr("x", Math.min(bx, sparkX1)).attr("y", 32)
          .attr("text-anchor", "end").attr("fill", C.ink)
          .text(d.annotation);
        svg.append("line")
          .attr("x1", bx).attr("x2", bx)
          .attr("y1", 36).attr("y2", rowY(0) - 10)
          .attr("stroke", C.hair).attr("stroke-width", 1);
      }

      // year ticks under the last row of bars, for orientation
      [0, k - 1].forEach(function (t) {
        svg.append("text").attr("class", "m-axis")
          .attr("x", barX(t) + barW / 2).attr("y", topPad + n * rowH + 14)
          .attr("text-anchor", "middle").attr("fill", C.muted)
          .text("’" + String(d.drift_years[t]).slice(2));
      });

      // ---- mini-map: lab colors, the movers at full ink ----
      var moverIds = {};
      movers.forEach(function (m) { moverIds[m.id] = true; });
      var mini = shared.minimap(right.node(), function (id) { return shared.colorOf(id); }, {
        opacityFn: function (id) { return moverIds[id] ? 0.95 : 0.22; },
        radiusFn: function (id) { return moverIds[id] ? 3.4 : 2; }
      });

      // dominant lab among the movers, computed from the data (not hardcoded)
      var labName = function (ci) {
        var com = null;
        (shared.communities || []).forEach(function (c) { if (c.id === ci) com = c; });
        return com ? com.label : "unaffiliated";
      };
      var byCom = {};
      movers.forEach(function (m) {
        var c = shared.communityOf(m.id);
        byCom[c] = (byCom[c] || 0) + 1;
      });
      var topCom = null;
      Object.keys(byCom).forEach(function (c) {
        if (topCom === null || byCom[c] > byCom[topCom]) topCom = c;
      });
      var miniNote = "The " + n + " biggest movers at full ink; everyone else fades.";
      if (topCom !== null && byCom[topCom] >= Math.ceil(n / 2)) {
        miniNote += " " + byCom[topCom] + " of " + n + " sit in the " +
          labName(+topCom) + " cluster, where long-running collaborations keep stacking weight.";
      }
      right.append("div").attr("class", "m-note").style("margin-top", "6px").text(miniNote);

      left.append("div").attr("class", "m-note").style("margin-top", "4px")
        .text("Drift partly tracks publication volume — a breakout year is usually also a " +
          "prolific one. " + d.n_eligible + " roster members with ≥ 1 recorded co-authorship " +
          "were ranked; the median total drift among them is just " +
          shared.fmt.sig(d.median_total, 1) + ", so movement on this scale is rare.");

      // ---- hover: dim other rows, light the person up on the mini-map ----
      function tipHtml(m) {
        var b = m.per_year.findIndex(function (p) { return p.year === m.breakout; });
        var bp = m.per_year[b];
        var prevW = b === 0 ? m.w0 : m.per_year[b - 1].w;
        return "<strong>" + shared.esc(shared.labelOf(m.id)) + "</strong>" +
          " · " + shared.esc(labName(shared.communityOf(m.id))) + "<br>" +
          "total drift " + shared.fmt.sig(m.total, 2) + " over " + k + " years<br>" +
          "breakout " + m.breakout + ": drift " + shared.fmt.sig(bp.d, 2) + ", " +
          (bp.new > 0
            ? "+" + bp.new + " new co-author tie" + (bp.new === 1 ? "" : "s") + " on this map,"
            : "no new ties — existing ones deepened,") +
          "<br>total tie strength " + prevW + " → " + bp.w;
      }
      rows.append("rect")
        .attr("x", 0).attr("y", function (m, i) { return topPad + i * rowH; })
        .attr("width", W).attr("height", rowH).attr("fill", "transparent")
        .on("mouseenter", function (evt, m) {
          var idx = movers.indexOf(m);
          rows.interrupt().attr("opacity", function (mm, j) { return j === idx ? 1 : 0.35; });
          mini.highlight(m.id);
          shared.tooltip.show(tipHtml(m), evt);
        })
        .on("mousemove", function (evt, m) { shared.tooltip.show(tipHtml(m), evt); })
        .on("mouseleave", function () {
          rows.interrupt().attr("opacity", 1);
          mini.highlight(null);
          shared.tooltip.hide();
        });
    }
  });
})();
