/* assets/js/analyses/who-hasnt-met.js */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("who-hasnt-met", {
    prose: {
      intro:
        "<p>Some of the likeliest coauthorships on this map haven&rsquo;t happened yet. " +
        "This panel reads what every member writes about — nothing but their paper titles — " +
        "and asks two questions: do people who write alike actually end up collaborating " +
        "here, and if so, which like-minded pairs are still missing their first paper " +
        "together?</p>",
      how:
        "<p>Step one: embed every member by their words. Each person&rsquo;s paper titles " +
        "become a TF-IDF vector — the humble bag-of-words embedding that predates " +
        "transformers — and the cosine between two people measures how alike their research " +
        "vocabulary is. Step two: measure how far apart the same two people sit in the " +
        "coauthorship graph, in hops (1 = coauthors, 2 = share a coauthor, and so on).</p>" +
        "<p>That gives two geometries over the same people — a text space and a graph space " +
        "— and the test asks whether neighborhoods in one match neighborhoods in the other, " +
        "the same question CKA or representational-alignment checks ask of two checkpoints&rsquo; " +
        "latent spaces. Here the answer is an emphatic yes: people who write alike sit " +
        "measurably closer in the graph. The payoff is the residual — pairs that are nearest " +
        "neighbors in text space yet two or more hops apart in the graph: that corner of the " +
        "chart is a list of collaborations waiting to happen.</p>",
      method:
        "<p>Distance correlation, the dependence test of Sz&eacute;kely, Rizzo &amp; Bakirov, " +
        "&ldquo;Measuring and testing dependence by correlation of distances&rdquo;, " +
        "<em>Annals of Statistics</em> 35(6), 2007 — zero only under independence, so it " +
        "catches nonlinear association that Pearson misses — and the workhorse generalized " +
        "by Multiscale Graph Correlation (Vogelstein et al., &ldquo;Discovering and " +
        "deciphering relationships across disparate data modalities&rdquo;, <em>eLife</em>, " +
        "2019). It runs here via the <em>hyppo</em> package (Panda, Mehta et al., " +
        "&ldquo;hyppo: A multivariate hypothesis testing Python package&rdquo;). This page " +
        "cannot resist pointing out that <strong>Joshua Vogelstein</strong> and " +
        "<strong>Ronak Mehta</strong> are both nodes on the very map being tested — the " +
        "dependence test and the software running this panel were built by people a few " +
        "hops from everyone on it. Details: similarity is cosine over title TF-IDF " +
        "(500 terms); distance is unweighted shortest-path hops; Dcorr on the 595 pairs " +
        "over 35 members, 1,000 permutations, p = 1/1001 — the smallest value a " +
        "1,000-permutation test can report. Six members have no resolved coauthor edge in " +
        "these databases, so graph distance is undefined for them and their pairs sit out " +
        "the test — a statement about database coverage, never about a person. Pairs that " +
        "share a person are not independent, so a stricter Mantel-style permutation over " +
        "people (not pairs) was run as a check: same verdict, p = 1/1001. One " +
        "high-similarity pair is left off the nominations because the databases do record " +
        "a joint paper that this map&rsquo;s edge filters dropped. The nominations are " +
        "invitations, not gaps — &ldquo;no recorded paper together&rdquo; is the only " +
        "claim being made.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data;
      var members = d.members;
      var noms = d.nominations;

      // pair objects; nomKey marks the nominated ("waiting to happen") pairs
      var nomKey = {};
      noms.forEach(function (n, k) { nomKey[n.a + "|" + n.b] = k; });
      var pairs = d.pairs.map(function (p, k) {
        var a = members[p[0]], b = members[p[1]];
        var nk = nomKey[a + "|" + b];
        if (nk === undefined) nk = nomKey[b + "|" + a];
        return { a: a, b: b, sim: p[2], hops: p[3], i: k, nom: nk === undefined ? null : noms[nk] };
      });
      var pText = d.test.p < 0.001 ? "p < 0.001" : "p = " + shared.fmt.sig(d.test.p);

      // ---- layout: scatter left, minimap + nomination list right ----
      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "14px 28px").style("align-items", "flex-start");
      var chartBox = wrap.append("div").style("flex", "1 1 340px").style("min-width", "300px");
      var sideBox = wrap.append("div").style("flex", "0 0 232px");

      // ---- scatter: similarity (x) vs hops apart (y, jittered rows) ----
      var W = Math.max(chartBox.node().clientWidth || 0, 300);
      var H = 330, top = 30, bottom = 40, left = 64, right = 14;
      var simMax = 0.56;
      var x = d3.scaleLinear().domain([0, simMax]).range([left, W - right]);
      var y = d3.scaleLinear().domain([0.6, 6.6]).range([H - bottom, top]);
      var bandH = Math.abs(y(2) - y(1));
      var jit = function (k) { return (((k + 1) * 0.6180339887498949) % 1 - 0.5) * 0.62 * bandH; };
      var px = function (p) { return x(p.sim); };
      var py = function (p) { return y(p.hops) + jit(p.i); };

      var svg = chartBox.append("svg").attr("width", W).attr("height", H);

      // the corner where text space and graph space disagree — lightly shaded
      svg.append("rect")
        .attr("x", x(d.nom_cutoff)).attr("y", top - 12)
        .attr("width", x(simMax) - x(d.nom_cutoff)).attr("height", y(1.5) - (top - 12))
        .attr("fill", shared.colors.hair).attr("fill-opacity", 0.33);
      svg.append("text").attr("class", "m-anno")
        .attr("x", x(simMax)).attr("y", top - 16).attr("text-anchor", "end")
        .text("collaborations waiting to happen");

      // hairline x axis + ticks
      svg.append("line")
        .attr("x1", left).attr("x2", W - right).attr("y1", H - bottom + 12).attr("y2", H - bottom + 12)
        .attr("stroke", shared.colors.hair);
      [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach(function (t) {
        svg.append("text").attr("class", "m-axis")
          .attr("x", x(t)).attr("y", H - bottom + 24).attr("text-anchor", "middle")
          .text(t === 0 ? "0" : ("." + Math.round(t * 10)));
      });
      svg.append("text").attr("class", "m-axis")
        .attr("x", (left + W - right) / 2).attr("y", H - 4).attr("text-anchor", "middle")
        .text("topic similarity — cosine between what their paper titles say →");

      // y rows: hops apart, coauthors at the bottom
      for (var h = 1; h <= 6; h++) {
        svg.append("text").attr("class", "m-axis")
          .attr("x", left - 8).attr("y", y(h) + 4).attr("text-anchor", "end")
          .style("fill", h === 1 ? shared.colors.ink : null)
          .text(h === 1 ? "coauthors" : h);
      }
      svg.append("text").attr("class", "m-axis")
        .attr("x", left - 8).attr("y", top - 4).attr("text-anchor", "end")
        .text("hops apart");

      // all 595 pairs
      svg.append("g").selectAll("circle").data(pairs).join("circle")
        .attr("cx", px).attr("cy", py).attr("r", 2.2)
        .attr("fill", shared.colors.ink).attr("fill-opacity", 0.35)
        .attr("pointer-events", "none");

      // the fit line, annotated with the verdict
      var fitX1 = 0, fitX2 = Math.min((1 - d.fit.intercept) / d.fit.slope, simMax);
      var fitY = function (s) { return y(d.fit.intercept + d.fit.slope * s); };
      svg.append("line")
        .attr("x1", x(fitX1)).attr("y1", fitY(fitX1))
        .attr("x2", x(fitX2)).attr("y2", fitY(fitX2))
        .attr("stroke", shared.colors.ink).attr("stroke-width", 1).attr("stroke-opacity", 0.55);
      var midS = (fitX1 + fitX2) / 2;
      var ang = Math.atan2(fitY(fitX2) - fitY(fitX1), x(fitX2) - x(fitX1)) * 180 / Math.PI;
      svg.append("text").attr("class", "m-anno")
        .attr("transform", "translate(" + x(midS) + "," + fitY(midS) + ") rotate(" + ang + ")")
        .attr("text-anchor", "middle").attr("dy", -7)
        .text("similar topics, closer collaboration — " + pText);

      // nominated pairs: solid dots + direct labels where there is room
      var nomPts = pairs.filter(function (p) { return p.nom; });
      svg.append("g").selectAll("circle").data(nomPts).join("circle")
        .attr("cx", px).attr("cy", py).attr("r", 4)
        .attr("fill", shared.colors.ink).attr("fill-opacity", 0.95)
        .attr("stroke", shared.colors.bg).attr("stroke-width", 1.2)
        .attr("pointer-events", "none");
      var lastName = function (id) { return shared.labelOf(id).split(" ").pop(); };
      nomPts.filter(function (p) { return p.nom === noms[0] || p.nom === noms[1]; })
        .forEach(function (p) {
          var rightSide = px(p) + 150 < W;
          var lx = px(p) + (rightSide ? 8 : -8);
          svg.append("text").attr("class", "m-label")
            .attr("x", lx).attr("y", py(p) - 3)
            .attr("text-anchor", rightSide ? "start" : "end")
            .text(lastName(p.a) + " × " + lastName(p.b));
          svg.append("text").attr("class", "m-axis")
            .attr("x", lx).attr("y", py(p) + 10).attr("text-anchor", rightSide ? "start" : "end")
            .text("both: " + p.nom.terms.slice(0, 2).join(", "));
        });

      chartBox.append("div")
        .style("font-size", "12px").style("color", shared.colors.muted)
        .style("line-height", "1.5").style("max-width", "62ch").style("margin-top", "6px")
        .html("Each dot is one pair of members: " + d.test.n_pairs + " pairs over " +
          members.length + " people. Six members have no resolved coauthor edge on this " +
          "map, so graph distance is undefined for them and their pairs sit out the test " +
          "— a fact about database coverage, not about them. Distance correlation " +
          shared.esc(String(d.test.dcorr)) + ", " + pText +
          " — checked twice, permuting pairs and permuting people.");

      // ---- minimap: where the nominees live ----
      var nomineeSet = {};
      noms.forEach(function (n) { nomineeSet[n.a] = 1; nomineeSet[n.b] = 1; });
      var nNominees = Object.keys(nomineeSet).length;
      var mapCircles = null;
      var mapHolder = sideBox.append("div").node();
      shared.minimap(mapHolder, function (id) {
        return nomineeSet[id] ? shared.colorOf(id) : null;
      }, {
        width: 200, height: 160,
        radiusFn: function (id) { return nomineeSet[id] ? 4 : (shared.isList(id) ? 3 : 2); },
        onReady: function (api) { mapCircles = api.svg.selectAll("circle"); }
      });
      sideBox.append("div").attr("class", "m-map-cap").style("max-width", "200px")
        .text("the " + nNominees + " people named in the " + noms.length +
          " nominations — hover a pair to find both");

      var highlightPair = function (a, b) {
        if (!mapCircles) return;
        mapCircles.interrupt();
        if (a == null) {
          mapCircles.attr("stroke", null).attr("fill-opacity", 0.9);
        } else {
          mapCircles
            .attr("stroke", function (n) { return (n.id === a || n.id === b) ? shared.colors.ink : null; })
            .attr("stroke-width", 1.4)
            .attr("fill-opacity", function (n) { return (n.id === a || n.id === b) ? 1 : 0.35; });
        }
      };

      // ---- nomination list: the labels, written out as invitations ----
      sideBox.append("div").attr("class", "m-kicker").style("margin", "16px 0 4px")
        .text("Collaborations waiting to happen");
      var focus = svg.append("circle").attr("r", 6).attr("fill", "none")
        .attr("stroke", shared.colors.ink).attr("stroke-width", 1.5).style("display", "none");
      var setFocus = function (p) {
        if (!p) { focus.style("display", "none"); highlightPair(null); return; }
        focus.style("display", null).attr("cx", px(p)).attr("cy", py(p));
        highlightPair(p.a, p.b);
      };
      var tipFor = function (p) {
        var html = "<strong>" + shared.esc(shared.labelOf(p.a)) + " × " +
          shared.esc(shared.labelOf(p.b)) + "</strong><br>topic similarity " +
          shared.esc(String(p.sim)) + " · " +
          (p.hops === 1 ? "already coauthors" : p.hops + " hops apart");
        if (p.nom) {
          html += "<br>both write about: " + shared.esc(p.nom.terms.join(", ")) +
            "<br><span style=\"color:" + shared.colors.muted +
            "\">no recorded paper together — yet</span>";
        }
        return html;
      };
      noms.forEach(function (n, k) {
        var p = nomPts.filter(function (q) { return q.nom === n; })[0];
        var row = sideBox.append("div")
          .style("padding", "4px 6px").style("margin", "0 -6px").style("cursor", "default");
        row.append("div")
          .style("font-size", "12.5px").style("color", shared.colors.ink)
          .html((k + 1) + ". " + shared.seatLink(n.a) + " × " + shared.seatLink(n.b));
        row.append("div")
          .style("font-size", "11.5px").style("color", shared.colors.muted)
          .style("line-height", "1.4")
          .text("both: " + n.terms.join(", ") + " · " + n.hops + " hops apart");
        row.on("mouseenter", function (evt) {
          row.style("background", shared.colors.hair);
          setFocus(p); shared.tooltip.show(tipFor(p), evt);
        }).on("mousemove", function (evt) { shared.tooltip.show(tipFor(p), evt); })
          .on("mouseleave", function () {
            row.style("background", null); setFocus(null); shared.tooltip.hide();
          });
      });

      // ---- hover anywhere on the scatter: nearest pair via Delaunay ----
      var delaunay = d3.Delaunay.from(pairs, px, py);
      svg.append("rect")
        .attr("x", left).attr("y", top - 12)
        .attr("width", W - right - left).attr("height", H - bottom + 12 - (top - 12))
        .attr("fill", "transparent")
        .on("mousemove", function (evt) {
          var m = d3.pointer(evt);
          var p = pairs[delaunay.find(m[0], m[1])];
          var dx = px(p) - m[0], dy = py(p) - m[1];
          if (dx * dx + dy * dy > 24 * 24) {
            setFocus(null); shared.tooltip.hide(); return;
          }
          setFocus(p); shared.tooltip.show(tipFor(p), evt);
        })
        .on("mouseleave", function () { setFocus(null); shared.tooltip.hide(); });
    }
  });
})();
