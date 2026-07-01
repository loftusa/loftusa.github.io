/* assets/js/analyses/small-world.js */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("small-world", {
    prose: {
      intro:
        "<p>&ldquo;It&rsquo;s a small world&rdquo; is a measurable claim. A network earns the " +
        "name when it pulls off two things at once: your collaborators&rsquo; collaborators " +
        "know each other (tight triangles), yet anyone can reach anyone else in a few hops. " +
        "This panel scores the map on both — and the scoring metric arrives with an easter " +
        "egg: one of its inventors is a green dot on this very map.</p>",
      how:
        "<p>Watts and Strogatz framed every network as sitting on a dial between two poles. " +
        "At one end is an orderly ring lattice — triangles abound but reaching the far side " +
        "takes forever; at the other is pure chance wiring — any two nodes sit a couple of " +
        "hops apart, but mutual-collaborator triangles vanish. A small world cheats the " +
        "trade-off: lattice-grade clustering with chance-grade path lengths.</p>" +
        "<p>Small-world propensity turns that into a number the way you&rsquo;d evaluate a " +
        "probe: the raw score means nothing until you bracket it with baselines. So the map " +
        "is rebuilt twice — shuffled into a degree-preserving random network (the chance " +
        "pole) and frozen into a ring lattice carrying the same tie strengths (the order " +
        "pole) — and the score is one minus how far real clustering and real path length " +
        "each fall short of their ideal pole. This map nails half the bargain — three " +
        "handshakes link most connected pairs — but keeps only modest clustering, landing " +
        "at 0.45: half a small world, leaning toward chance.</p>",
      method:
        "<p>Small-world propensity (SWP) for weighted networks: Muldoon, Bridgeford &amp; " +
        "Bassett, &ldquo;Small-World Propensity and Weighted Brain Networks&rdquo;, " +
        "<em>Scientific Reports</em> 6, 22057 (2016), building on Watts &amp; Strogatz, " +
        "&ldquo;Collective dynamics of &lsquo;small-world&rsquo; networks&rdquo;, " +
        "<em>Nature</em> 393 (1998). That Bridgeford is <strong>Eric Bridgeford</strong> — " +
        "a green node in the Vogelstein cluster of this very map: his own collaboration " +
        "network, graded by his own metric. Computed on the weighted largest component " +
        "(99 of 125 people, 312 ties). Edges use fractional co-authorship counting (each " +
        "paper adds 1/n_authors per pair; suggested by Stella Biderman), so a duo paper " +
        "carries more tie strength than a consortium paper: C is Onnela&rsquo;s weighted " +
        "clustering, normalized by the heaviest tie on the map (Carey E. Priebe &amp; Hayden " +
        "S. Helm, tie strength 6.2 across 33 shared papers); L is the mean shortest path " +
        "with distance 1/strength &mdash; a strong duo bond is a short hop, a mega-paper " +
        "tie a long one. " +
        "Random null: degree-preserving double-edge-swap of the binarized graph (10&times; " +
        "edges swaps) with the original weight multiset reshuffled onto the surviving edges, " +
        "averaged over 20 seeded draws. Lattice null: a ring lattice with the same node and " +
        "edge count, sorted weights placed nearest-first. Both nulls are deterministic " +
        "approximations of Muldoon et&nbsp;al.&rsquo;s construction (they reorder weights " +
        "toward the adjacency-matrix diagonal). &Delta;C&nbsp;=&nbsp;0.76, " +
        "&Delta;L&nbsp;=&nbsp;0.15, SWP&nbsp;=&nbsp;1&nbsp;&minus;&nbsp;&radic;((&Delta;C" +
        "&sup2;&nbsp;+&nbsp;&Delta;L&sup2;)/2)&nbsp;=&nbsp;0.45. The per-lab axis normalizes " +
        "all three labs by that same map-wide maximum so the dots share one scale; score " +
        "each lab against its own strongest tie instead and the ranking flips to the Bau lab " +
        "(0.025) over Vogelstein (0.015) — &ldquo;tightest&rdquo; depends on whether " +
        "the 33-paper Priebe&ndash;Helm bond sets the yardstick, so both numbers ship in " +
        "the data. Hop counts use the " +
        "unweighted component; 13 of the 48 list members have no resolved coauthor tie " +
        "inside this map — a statement about the databases, never about a person — so the " +
        "histogram covers the connected 35.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data;
      var GUT = 88, RPAD = 14;

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "16px 30px").style("align-items", "flex-start");
      var chartBox = wrap.append("div").style("flex", "1 1 340px").style("min-width", "300px");
      var mapBox = wrap.append("div").style("flex", "0 0 auto").style("width", "204px");

      // ---- the shared mini-map, colored by lab ------------------------------------
      var map = shared.minimap(mapBox.append("div").node(), function (id) {
        var c = shared.communityOf(id);
        return (c === 0 || c === 1 || c === 2) ? shared.colors.community[c] : null;
      }, { width: 204, height: 163 });
      mapBox.append("div").attr("class", "m-note").style("margin-top", "6px")
        .html("The map, colored by lab; grey is the periphery. " +
          "<span class=\"sw-eric\" style=\"border-bottom:1px dotted " + shared.colors.muted +
          "; cursor:help\">" + shared.esc(d.eric.label) + "</span> — co-inventor of the " +
          "small-world score used here — is one of the green dots.");

      var W = Math.max(chartBox.node().clientWidth || 0, 300);
      var section = function (title) {
        var s = chartBox.append("div").style("margin", "0 0 20px 0");
        s.append("div").attr("class", "m-label")
          .style("color", shared.colors.ink).style("font-weight", "600")
          .style("margin-bottom", "4px").text(title);
        return s;
      };

      // ================= 1. the order-to-chance scale ==============================
      var s1 = section("where the map sits between order and chance");
      var svg1 = s1.append("svg").attr("width", W).attr("height", 84);
      var xs = d3.scaleLinear().domain([0, 1]).range([GUT, W - RPAD]);

      svg1.append("text").attr("class", "m-axis").attr("x", xs(0)).attr("y", 12)
        .attr("fill", shared.colors.muted).text("orderly lattice");
      svg1.append("text").attr("class", "m-axis").attr("x", xs(1)).attr("y", 12)
        .attr("text-anchor", "end").attr("fill", shared.colors.muted).text("pure chance");

      var rows1 = [
        { name: "clustering", pos: d.dC, ideal: 0, y: 38,
          tip: "<strong>weighted clustering</strong><br>this map " + d.C.toFixed(4) +
            " &middot; random null " + d.C_rand.toFixed(4) + " &middot; ring lattice " +
            d.C_latt.toFixed(4) + "<br><span style=\"color:" + shared.colors.muted + "\">" +
            shared.fmt.pct(d.dC) + " of the way from lattice to random</span>" },
        { name: "path length", pos: 1 - d.dL, ideal: 1, y: 66,
          tip: "<strong>mean weighted path</strong> (distance = 1/tie strength)<br>this map " +
            d.L.toFixed(2) + " &middot; random null " + d.L_rand.toFixed(2) +
            " &middot; ring lattice " + d.L_latt.toFixed(2) + "<br><span style=\"color:" +
            shared.colors.muted + "\">only " + shared.fmt.pct(d.dL) +
            " of the way back from random to lattice</span>" }
      ];
      var g1 = svg1.selectAll("g.sw-row").data(rows1).join("g");
      g1.append("line")
        .attr("x1", xs(0)).attr("x2", xs(1))
        .attr("y1", function (r) { return r.y; }).attr("y2", function (r) { return r.y; })
        .attr("stroke", shared.colors.hair).attr("stroke-width", 1);
      g1.append("text").attr("class", "m-label")
        .attr("x", GUT - 10).attr("y", function (r) { return r.y + 4; })
        .attr("text-anchor", "end").attr("fill", shared.colors.ink)
        .text(function (r) { return r.name; });
      g1.append("line") // the dotted gap to the ideal pole — its length IS the metric
        .attr("x1", function (r) { return xs(r.ideal); })
        .attr("x2", function (r) { return xs(r.pos); })
        .attr("y1", function (r) { return r.y; }).attr("y2", function (r) { return r.y; })
        .attr("stroke", shared.colors.muted).attr("stroke-width", 1.2)
        .attr("stroke-dasharray", "1.5,3.5");
      g1.append("text").attr("class", "m-axis")
        .attr("x", function (r) {
          var mid = (xs(r.ideal) + xs(r.pos)) / 2;
          return Math.max(GUT + 26, Math.min(mid, W - 40));
        })
        .attr("y", function (r) { return r.y - 7; })
        .attr("text-anchor", "middle").attr("fill", shared.colors.muted)
        .text(function (r) { return "gap " + Math.abs(r.pos - r.ideal).toFixed(2); });
      g1.append("circle") // the ideal pole, hollow
        .attr("cx", function (r) { return xs(r.ideal); })
        .attr("cy", function (r) { return r.y; }).attr("r", 4)
        .attr("fill", shared.colors.bg).attr("stroke", shared.colors.muted)
        .attr("stroke-width", 1.1);
      g1.append("circle") // where this map actually sits
        .attr("cx", function (r) { return xs(r.pos); })
        .attr("cy", function (r) { return r.y; }).attr("r", 4.5)
        .attr("fill", shared.colors.ink).attr("fill-opacity", 0.92);
      g1.append("rect")
        .attr("x", 0).attr("width", W)
        .attr("y", function (r) { return r.y - 13; }).attr("height", 26)
        .attr("fill", "transparent")
        .on("mouseenter", function (evt, r) { shared.tooltip.show(r.tip, evt); })
        .on("mousemove", function (evt, r) { shared.tooltip.show(r.tip, evt); })
        .on("mouseleave", function () { shared.tooltip.hide(); });
      s1.append("div").attr("class", "m-note")
        .html("Small-world propensity <strong style=\"color:" + shared.colors.ink + "\">" +
          d.swp.toFixed(2) + "</strong> &mdash; one minus the typical (root-mean-square) " +
          "dotted gap. A perfect small world (SWP&nbsp;1) pins clustering at the lattice " +
          "end <em>and</em> paths at the chance end; this map wins its chance-short paths " +
          "but keeps only modest clustering.");

      // ================= 2. handshake histogram ====================================
      var s2 = section("handshakes between any two list members in the connected core");
      var svg2 = s2.append("svg").attr("width", W).attr("height", 130);
      var bins = d.hops.bins, nPairs = d.hops.n_pairs;
      var base = 98, top = 22;
      var step = (W - RPAD - GUT) / bins.length;
      var bw = Math.min(40, step * 0.62);
      var cx2 = function (i) { return GUT + step * (i + 0.5); };
      var yc = d3.scaleLinear()
        .domain([0, d3.max(bins, function (b) { return b[1]; })]).range([base, top]);
      var cum = 0;
      var bars = bins.map(function (b, i) {
        cum += b[1];
        return { hop: b[0], n: b[1], i: i, cum: cum };
      });
      svg2.append("line").attr("x1", GUT).attr("x2", W - RPAD)
        .attr("y1", base).attr("y2", base)
        .attr("stroke", shared.colors.hair).attr("stroke-width", 1);
      var g2 = svg2.selectAll("g.sw-bar").data(bars).join("g");
      g2.append("rect")
        .attr("x", function (b) { return cx2(b.i) - bw / 2; })
        .attr("y", function (b) { return yc(b.n); })
        .attr("width", bw).attr("height", function (b) { return base - yc(b.n); })
        .attr("fill", shared.colors.ink)
        .attr("fill-opacity", function (b) { return b.hop === d.hops.median ? 0.8 : 0.5; });
      g2.append("text").attr("class", "m-axis")
        .attr("x", function (b) { return cx2(b.i); })
        .attr("y", function (b) { return yc(b.n) - 4; })
        .attr("text-anchor", "middle").attr("fill", shared.colors.muted)
        .text(function (b) { return b.n; });
      g2.append("text").attr("class", "m-axis")
        .attr("x", function (b) { return cx2(b.i); }).attr("y", base + 14)
        .attr("text-anchor", "middle")
        .attr("fill", function (b) {
          return b.hop === d.hops.median ? shared.colors.ink : shared.colors.muted;
        })
        .text(function (b) { return b.hop; });
      svg2.selectAll("text.sw-med").data(bars.filter(function (b) {
        return b.hop === d.hops.median;
      })).join("text").attr("class", "m-label sw-med")
        .attr("x", function (b) { return cx2(b.i); }).attr("y", base + 27)
        .attr("text-anchor", "middle").attr("fill", shared.colors.ink).text("median");
      var tip2 = function (b) {
        return "<strong>" + b.hop + (b.hop === 1 ? " handshake" : " handshakes") +
          "</strong> apart: " + b.n + " pairs (" + shared.fmt.pct(b.n / nPairs) +
          ")<br><span style=\"color:" + shared.colors.muted + "\">" +
          shared.fmt.pct(b.cum / nPairs) + " of pairs are within " + b.hop + "</span>";
      };
      g2.append("rect")
        .attr("x", function (b) { return cx2(b.i) - step / 2; })
        .attr("y", top - 8).attr("width", step).attr("height", base - top + 22)
        .attr("fill", "transparent")
        .on("mouseenter", function (evt, b) { shared.tooltip.show(tip2(b), evt); })
        .on("mousemove", function (evt, b) { shared.tooltip.show(tip2(b), evt); })
        .on("mouseleave", function () { shared.tooltip.hide(); });
      s2.append("div").attr("class", "m-note")
        .html("One handshake = one coauthorship. <strong style=\"color:" + shared.colors.ink +
          "\">" + d.hops.median + " handshakes reach " + shared.fmt.pct(d.hops.cum_at_median) +
          "</strong> of the " + shared.fmt.num(nPairs) + " pairs among the " + d.hops.n_members +
          " list members in the connected core (the databases give the other " +
          (d.hops.n_list_total - d.hops.n_members) +
          " no in-map coauthor tie &mdash; a fact about the records, not the people).");

      // ================= 3. lab clustering, one shared scale =======================
      var s3 = section("within-lab clustering, one shared scale");
      var svg3 = s3.append("svg").attr("width", W).attr("height", 92);
      var labs = d.labs;
      var tight = labs.filter(function (l) { return l.id === d.tightest_lab; })[0];
      var xs3 = d3.scaleLinear()
        .domain([0, d3.max(labs, function (l) { return l.C; }) * 1.22])
        .range([GUT, W - RPAD]);
      var y3 = 56;
      svg3.append("line").attr("x1", xs3(0)).attr("x2", xs3.range()[1])
        .attr("y1", y3).attr("y2", y3)
        .attr("stroke", shared.colors.hair).attr("stroke-width", 1);
      svg3.append("text").attr("class", "m-axis").attr("x", xs3(0)).attr("y", y3 + 16)
        .attr("text-anchor", "middle").attr("fill", shared.colors.muted).text("0");
      svg3.append("text").attr("class", "m-axis").attr("x", GUT - 10).attr("y", y3 + 4)
        .attr("text-anchor", "end").attr("fill", shared.colors.ink).text("clustering");
      var g3 = svg3.selectAll("g.sw-lab").data(labs).join("g");
      g3.append("circle")
        .attr("cx", function (l) { return xs3(l.C); }).attr("cy", y3).attr("r", 5.5)
        .attr("fill", function (l) { return shared.colors.community[l.id]; })
        .attr("fill-opacity", 0.95);
      var lab3 = g3.append("text").attr("class", "m-label")
        .attr("x", function (l) { // clamp so long names stay inside the svg
          var half = (l.label.length + 6) * 3.3;
          return Math.max(half + 2, Math.min(xs3(l.C), W - half - 2));
        })
        .attr("y", function (l) { return l.id === 0 ? y3 + 27 : y3 - 13; })
        .attr("text-anchor", "middle")
        .attr("fill", function (l) { return shared.colors.community[l.id]; })
        .text(function (l) { return l.label; });
      lab3.append("tspan").attr("fill", shared.colors.muted)
        .text(function (l) { return " " + l.C.toFixed(3); });
      var tip3 = function (l) {
        return "<strong>" + shared.esc(l.label) + "</strong> <span style=\"color:" +
          shared.colors.muted + "\">" + l.n + " people &middot; " + l.edges +
          " within-lab ties</span><br>clustering " + l.C.toFixed(4) +
          " on the shared scale (" + l.C_own.toFixed(4) +
          " against its own strongest tie)<br><span style=\"color:" + shared.colors.muted +
          "\">strongest tie " + l.w_max_papers + " shared papers (strength " +
          l.w_max.toFixed(1) + ") &middot; mean weighted path " +
          l.L.toFixed(2) + " across its connected " + l.lcc_n + "</span>";
      };
      g3.append("rect")
        .attr("x", function (l) { return xs3(l.C) - 16; }).attr("y", y3 - 22)
        .attr("width", 32).attr("height", 50).attr("fill", "transparent")
        .on("mouseenter", function (evt, l) { shared.tooltip.show(tip3(l), evt); })
        .on("mousemove", function (evt, l) { shared.tooltip.show(tip3(l), evt); })
        .on("mouseleave", function () { shared.tooltip.hide(); });
      s3.append("div").attr("class", "m-note").style("margin-bottom", "0")
        .html("Tightest knit on the common scale: the <strong style=\"color:" +
          shared.colors.community[tight.id] + "\">" + shared.esc(tight.label) +
          "</strong> lab, whose triangles ride repeat collaborations of up to " +
          tight.w_max_papers +
          " shared papers. It is also <span class=\"sw-eric\" style=\"border-bottom:1px dotted " +
          shared.colors.muted + "; cursor:help\">" + shared.esc(d.eric.label) +
          "</span>&rsquo;s home lab &mdash; the tightest cluster contains the co-author of " +
          "the metric doing the judging. (Scored against each lab&rsquo;s own strongest tie " +
          "instead, the Bau lab ranks first; the footnote has both readings.)");

      // ---- Eric Bridgeford hover: find the metric's co-author on the mini-map ------
      var ericNode = shared.nodes.get(d.eric.id);
      var ericLab = (shared.communities || []).filter(function (c) {
        return c.id === d.eric.community;
      })[0];
      var ericTip = "<strong>" + shared.esc(shared.labelOf(d.eric.id)) + "</strong> " +
        "<span style=\"color:" + shared.colors.muted + "\">" +
        shared.esc(ericLab ? ericLab.label : "") +
        " &middot; " + (ericNode ? ericNode.degree : "") + " coauthor ties here</span><br>" +
        "co-author of the small-world-propensity paper (Muldoon, Bridgeford &amp; Bassett " +
        "2016) &mdash; his own metric, measuring his own network.";
      wrap.selectAll("span.sw-eric")
        .on("mouseenter", function (evt) {
          map.highlight(d.eric.id);
          shared.tooltip.show(ericTip, evt);
        })
        .on("mousemove", function (evt) { shared.tooltip.show(ericTip, evt); })
        .on("mouseleave", function () {
          map.highlight(null);
          shared.tooltip.hide();
        });
    }
  });
})();
