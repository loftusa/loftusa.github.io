/* assets/js/analyses/stars-and-tribes.js */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("stars-and-tribes", {
    prose: {
      intro:
        "<p>Every page of this project paints people in three lab colors — but where do the " +
        "colors come from? Turn the network into vectors and the answer hangs on one " +
        "preprocessing choice: read the raw who-wrote-with-whom matrix and the geometry sorts " +
        "stars from everyone else; divide each person&rsquo;s paper volume out first and the " +
        "same math finds the tribes. Both clouds below are the same 99 people and the same " +
        "edges — only the normalization differs. And once the tribes are visible, a harder " +
        "question: are there really three of them, or just two?</p>",
      how:
        "<p>A spectral embedding is representation learning without the gradient descent: take " +
        "the coauthorship matrix, keep its top six eigenvectors, and every person becomes a " +
        "vector — like reading activations off a chosen layer. And as with layers, " +
        "<em>where</em> you read decides <em>what</em> you see. Read the raw adjacency matrix " +
        "and the leading direction tracks sheer connection volume (r&nbsp;=&nbsp;0.79 with " +
        "weighted degree): the left cloud strings everyone out by prominence while the three " +
        "labs dissolve into one faint blob.</p>" +
        "<p>Normalize by degree first — the graph Laplacian — and that volume signal is divided " +
        "away, leaving the <em>direction</em> of your edges rather than their count: who your " +
        "coauthors are, not how many. Fit a three-component Gaussian mixture to each embedding " +
        "and the contrast is stark — on the right cloud it reproduces the map&rsquo;s lab " +
        "colors for 98 of 99 people (the lone exception is a deliberate editorial assignment), " +
        "on the left it agrees with them at chance level. Neither " +
        "reading is wrong — centrality and community are both genuinely in the matrix, and " +
        "which truth you get depends on the normalization: the two-truths phenomenon.</p>" +
        "<p>Zoom out, though, and a third reading suggests itself: maybe the truer structure " +
        "is two blocks, not three — one interpretability core-and-periphery (EleutherAI and " +
        "the Bau lab together) plus a separate NeuroData block. The circumstantial evidence " +
        "is real: EleutherAI and the Vogelstein lab share zero direct edges; inside the interp " +
        "side, a person&rsquo;s tie volume and their embedding radius rise together " +
        "(r&nbsp;=&nbsp;0.63), the signature of one core with a fringe; and the " +
        "EleutherAI&ndash;Bau seam carries 11% of the interp side&rsquo;s tie weight while the " +
        "Bau&ndash;Vogelstein seam carries under 1% of its own side&rsquo;s. But put the two " +
        "readings head to head — fit a degree-corrected blockmodel with memberships fixed each " +
        "way and compare fits, like comparing two architectures on the same data with a " +
        "penalty for extra parameters — and the three labs win by &Delta;BIC&nbsp;&asymp;&nbsp;48: " +
        "even after the star-versus-fringe gradient is divided out, EleutherAI and the Bau lab " +
        "keep measurably different company. The verdict line under the clouds keeps score.</p>",
      method:
        "<p>Adjacency vs. Laplacian spectral embedding with Gaussian-mixture clustering, after " +
        "Priebe, Park, Vogelstein <em>et al.</em>, &ldquo;On a two-truths phenomenon in " +
        "spectral graph clustering&rdquo;, <em>PNAS</em> 116(13), 2019, which showed ASE " +
        "recovers core&ndash;periphery/degree structure while LSE recovers affinity " +
        "communities. The easter egg: three of that paper&rsquo;s authors — Carey E. Priebe, " +
        "Joshua Vogelstein, and Eric Bridgeford — are dots in these clouds (all three labeled " +
        "above), so the two-truths authors are here clustered by their own method; the paper " +
        "itself is one of the records in this dataset. Details: weighted largest connected " +
        "component (99 of 125 people); edges use fractional co-authorship counting (each paper " +
        "adds 1/n_authors per pair; suggested by Stella Biderman); ASE d&nbsp;=&nbsp;6 (full " +
        "SVD), dim&nbsp;1 oriented positive with weighted degree, Pearson r&nbsp;=&nbsp;0.79; " +
        "LSE d&nbsp;=&nbsp;6 (R-DAD regularized Laplacian), rows normalized to the unit sphere, " +
        "GMM(k&nbsp;=&nbsp;3, fixed seed) anchor-mapped to lab ids — ARI&nbsp;=&nbsp;1.0 " +
        "against the map&rsquo;s colors, vs ARI&nbsp;=&nbsp;0.015 for the identical GMM on " +
        "ASE. The two-vs-three-block verdict fixes memberships ({EleutherAI&thinsp;+&thinsp;Bau, " +
        "Vogelstein} vs the three labs) and compares degree-corrected stochastic-blockmodel " +
        "fits by the Karrer&ndash;Newman profile log-likelihood (<em>Phys. Rev. E</em> 83, " +
        "2011) with a BIC penalty — k(k+1)/2 block parameters plus n&minus;k degree parameters " +
        "over n(n&minus;1)/2 dyads; on fractional weights this Poisson likelihood is a " +
        "quasi-likelihood, since the weights are no longer integer counts. Honest caveats: the " +
        "map&rsquo;s colors were themselves assigned by this same Laplacian-plus-mixture " +
        "recipe, so ARI&nbsp;1.0 is the page agreeing with its own cookbook rather than " +
        "independent ground truth — the informative gap is that the adjacency reading scores " +
        "~0 on the identical target; the two spots where the map&rsquo;s color was once set " +
        "by an editorial override are now what the model says natively; and the left " +
        "cloud&rsquo;s axes are &radic;-scaled for legibility.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, st = d.stats;
      var nodes = d.nodes, n = nodes.length;
      var TT_AUTHORS = { "carey e priebe": 1, "j vogelstein": 1, "joshua t vogelstein": 1, "eric bridgeford": 1 };

      var labName = function (c) {
        var m = (shared.communities || []).filter(function (x) { return x.id === c; })[0];
        return m ? m.label : "—";
      };
      var labColor = function (c) { return shared.colors.community[c % shared.colors.community.length]; };

      // prominence rank (1 = most prominent) from ASE dim 1
      var rank = {};
      nodes.slice().sort(function (a, b) { return b.ase[0] - a.ase[0]; })
        .forEach(function (p, i) { rank[p.id] = i + 1; });
      var aseOpacity = function (p) { return 0.14 + 0.8 * (1 - (rank[p.id] - 1) / (n - 1)); };
      var byId = {};
      nodes.forEach(function (p) { byId[p.id] = p; });

      // ---- layout: two clouds side by side, minimap + note beneath ----
      var W = Math.max(el.clientWidth || 0, 300);
      var GAP = 26;
      var cloudW = W >= 620 ? Math.min(372, Math.floor((W - GAP) / 2)) : Math.min(420, W);
      var cloudH = 308;
      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "18px " + GAP + "px").style("align-items", "flex-start");

      function cloudBox(knob, phrase) {
        var box = wrap.append("div").style("flex", "0 0 " + cloudW + "px");
        box.append("div").style("font-size", "12.5px").style("font-weight", "600")
          .style("color", shared.colors.ink).text(knob);
        box.append("div").style("font-size", "11.5px").style("color", shared.colors.muted)
          .style("margin-bottom", "4px").text(phrase);
        return box.append("svg").attr("width", cloudW).attr("height", cloudH);
      }

      // direct-label placement: radial candidate sweep, rightmost dots first, 3px gaps,
      // axis-caption bands reserved, leader hairline when the label lands far from its dot.
      // (Offsets validated offline against the shipped coordinates at widths 300-372.)
      function placeLabels(svg, pts) {
        var boxes = pts.map(function (p) { return [p.x - 5, p.y - 5, 10, 10]; });
        boxes.push([0, cloudH - 20, cloudW, 20]); // bottom axis caption band
        boxes.push([0, 0, 76, 15]);               // top-left axis caption
        var RAD = [10, 16, 24, 34, 46, 60], ANG = 16;
        pts.slice().sort(function (a, b) { return b.x - a.x; }).forEach(function (p) {
          var w = 6.4 * p.text.length + 6, h = 14, pick = null;
          for (var ri = 0; ri < RAD.length && !pick; ri++) {
            for (var k = 0; k < ANG && !pick; k++) {
              var ang = -Math.PI / 2 + (k * 2 * Math.PI) / ANG;
              var c = Math.cos(ang), s = Math.sin(ang), r = RAD[ri];
              var tx = p.x + r * c, ty = p.y + r * s + 4;
              var anch = c > 0.38 ? "start" : c < -0.38 ? "end" : "middle";
              var x0 = anch === "start" ? tx : anch === "end" ? tx - w : tx - w / 2;
              var y0 = ty - h + 3;
              if (x0 < 2 || x0 + w > cloudW - 2 || y0 < 2 || y0 + h > cloudH - 2) continue;
              var clash = boxes.some(function (b) {
                return !(x0 + w + 3 < b[0] || b[0] + b[2] + 3 < x0 ||
                  y0 + h + 3 < b[1] || b[1] + b[3] + 3 < y0);
              });
              if (!clash) pick = { tx: tx, ty: ty, anch: anch, box: [x0, y0, w, h], r: r, c: c, s: s };
            }
          }
          if (!pick) pick = { tx: p.x + 7, ty: p.y - 6, anch: "start", box: [p.x + 7, p.y - 17, w, h], r: 10, c: 1, s: 0 };
          boxes.push(pick.box);
          if (pick.r >= 24) {
            svg.append("line")
              .attr("x1", p.x + 5 * pick.c).attr("y1", p.y + 5 * pick.s)
              .attr("x2", p.x + (pick.r - 6) * pick.c).attr("y2", p.y + (pick.r - 6) * pick.s)
              .attr("stroke", shared.colors.muted).attr("stroke-width", 0.7)
              .attr("stroke-opacity", 0.55);
          }
          svg.append("text").attr("class", "m-label")
            .attr("x", pick.tx).attr("y", pick.ty)
            .attr("text-anchor", pick.anch).text(p.text);
        });
      }

      var labelText = function (id) {
        return shared.labelOf(id) + (TT_AUTHORS[id] ? " ✦" : "");
      };

      var PAD = { l: 14, r: 14, t: 16, b: 26 };

      // ---- left cloud: ASE dims 1–2, ink, opacity = prominence (√-scaled axes) ----
      var sq = function (v) { return Math.sign(v) * Math.sqrt(Math.abs(v)); };
      var aseSvg = cloudBox("Raw adjacency", "sorted by prominence — the labs vanish");
      var ax = d3.scaleLinear()
        .domain([0, d3.max(nodes, function (p) { return sq(p.ase[0]); })])
        .range([PAD.l, cloudW - PAD.r]);
      var ayExt = d3.extent(nodes, function (p) { return sq(p.ase[1]); });
      var ay = d3.scaleLinear().domain(ayExt).range([cloudH - PAD.b, PAD.t]);

      aseSvg.append("line")
        .attr("x1", PAD.l).attr("x2", cloudW - PAD.r)
        .attr("y1", cloudH - PAD.b + 8).attr("y2", cloudH - PAD.b + 8)
        .attr("stroke", shared.colors.hair).attr("stroke-width", 1);
      aseSvg.append("text").attr("class", "m-axis")
        .attr("x", cloudW - PAD.r).attr("y", cloudH - 6).attr("text-anchor", "end")
        .text("more prominent → (1st adjacency axis, √ scale)");
      aseSvg.append("text").attr("class", "m-axis")
        .attr("x", PAD.l - 4).attr("y", 11).text("2nd axis ↑");

      var aseSorted = nodes.slice().sort(function (a, b) { return a.ase[0] - b.ase[0]; });
      var aseDots = aseSvg.append("g").selectAll("circle").data(aseSorted).join("circle")
        .attr("cx", function (p) { return ax(sq(p.ase[0])); })
        .attr("cy", function (p) { return ay(sq(p.ase[1])); })
        .attr("r", 3.5)
        .attr("fill", shared.colors.ink)
        .attr("fill-opacity", aseOpacity);
      placeLabels(aseSvg, d.labeled.map(function (id) {
        var p = byId[id];
        return { x: ax(sq(p.ase[0])), y: ay(sq(p.ase[1])), text: labelText(id) };
      }));

      // ---- right cloud: LSE best 2 dims, lab colors from the GMM clusters ----
      var lseSvg = cloudBox("Degree-normalized Laplacian", "sorted by company kept — they snap into view");
      var lxExt = d3.extent(nodes, function (p) { return p.lse[0]; });
      var lyExt = d3.extent(nodes, function (p) { return p.lse[1]; });
      var lx = d3.scaleLinear().domain(lxExt).range([PAD.l, cloudW - PAD.r]);
      var ly = d3.scaleLinear().domain(lyExt).range([cloudH - PAD.b, PAD.t]);

      lseSvg.append("line")
        .attr("x1", PAD.l).attr("x2", cloudW - PAD.r)
        .attr("y1", cloudH - PAD.b + 8).attr("y2", cloudH - PAD.b + 8)
        .attr("stroke", shared.colors.hair).attr("stroke-width", 1);
      lseSvg.append("text").attr("class", "m-axis")
        .attr("x", cloudW - PAD.r).attr("y", cloudH - 6).attr("text-anchor", "end")
        .text("Laplacian dim " + (st.lse_dims[0] + 1) + " → (volume divided out)");
      lseSvg.append("text").attr("class", "m-axis")
        .attr("x", PAD.l - 4).attr("y", 11).text("dim " + (st.lse_dims[1] + 1) + " ↑");

      var lseSorted = nodes.slice().sort(function (a, b) { return a.wd - b.wd; });
      var lseDots = lseSvg.append("g").selectAll("circle").data(lseSorted).join("circle")
        .attr("cx", function (p) { return lx(p.lse[0]); })
        .attr("cy", function (p) { return ly(p.lse[1]); })
        .attr("r", 3.5)
        .attr("fill", function (p) { return labColor(p.g); })
        .attr("fill-opacity", 0.85);
      placeLabels(lseSvg, d.labeled.map(function (id) {
        var p = byId[id];
        return { x: lx(p.lse[0]), y: ly(p.lse[1]), text: labelText(id) };
      }));

      // ---- one-line verdict: 2 blocks (interp vs NeuroData) or 3 (the labs)? ----
      var seam = d.seam;
      wrap.append("div")
        .style("flex", "1 1 100%").style("max-width", "76ch")
        .style("font-size", "12.5px").style("line-height", "1.5")
        .style("color", shared.colors.ink)
        .html("<strong>Two blocks or three?</strong> <span style=\"color:" +
          shared.colors.muted + "\">Merge EleutherAI&thinsp;+&thinsp;Bau into one interp " +
          "block and refit a degree-corrected blockmodel: the three labs win by " +
          "&Delta;BIC&nbsp;" + Math.round(seam.dbic) + " — even though EleutherAI and the " +
          "Vogelstein lab share " + (seam.ev_edges === 0 ? "zero" : seam.ev_edges) +
          " direct edges, the two interp labs keep different company.</span>");

      // ---- beneath: the standard minimap recolored by the Laplacian clusters + the note ----
      var foot = wrap.append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "8px 24px").style("align-items", "flex-start")
        .style("flex", "1 1 100%");
      var mapCell = foot.append("div").style("flex", "0 0 170px");
      var map = shared.minimap(mapCell.node(), function (id) {
        var p = byId[id];
        return p ? labColor(p.g) : null;
      }, { width: 170, height: 136 });
      mapCell.append("div").style("font-size", "11px").style("color", shared.colors.muted)
        .style("text-align", "center").style("margin-top", "2px")
        .text("the right cloud's clusters, on the map");
      foot.append("div")
        .style("flex", "1 1 260px").style("max-width", "62ch")
        .style("font-size", "12.5px").style("line-height", "1.5")
        .style("color", shared.colors.muted)
        .html("Same " + n + " people in both clouds; hover any dot to find its twin. Fitting the " +
          "same 3-cluster mixture to each embedding: the right cloud reproduces the map&rsquo;s " +
          "lab colors at ARI&nbsp;" + shared.fmt.sig(st.ari_lse) + " (" + (n - st.n_disagree) +
          " of " + n + " people" + (st.n_disagree === 1 ?
            "; the one holdout is a spot where the map&rsquo;s color was set by hand" : "") +
          ") — the left cloud manages ARI&nbsp;" + shared.fmt.sig(st.ari_ase) +
          ", i.e. chance. In fact the map&rsquo;s colors were chosen by the right-hand recipe, " +
          "so this panel is the page opening its own cookbook. ✦&nbsp;marks authors of the " +
          "two-truths paper itself — mapped by their own method.");

      // ---- hover: tooltip + twin pulse in the other cloud + minimap highlight ----
      function resetDots() {
        aseDots.interrupt(); lseDots.interrupt();
        aseDots.attr("r", 3.5).attr("stroke", null).attr("fill-opacity", aseOpacity);
        lseDots.attr("r", 3.5).attr("stroke", null).attr("fill-opacity", 0.85);
      }
      function pulse(sel) {
        sel.transition().duration(320).attr("r", 8)
          .transition().duration(320).attr("r", 4.5)
          .on("end", function () { pulse(sel); });
      }
      function tipFor(p) {
        var html = "<div class=\"t-name\">" + shared.esc(shared.labelOf(p.id)) + "</div>" +
          "<div class=\"t-sub\">" + shared.esc(labName(p.c)) + " · tie strength " +
          shared.fmt.sig(p.wd) + " (fractional)</div>" +
          (rank[p.id] <= 10
            ? "prominence: #" + rank[p.id] + " of " + n + " on the left axis<br>"
            : "left cloud: in the quiet blob, where the labs blur together<br>") +
          "company kept: the " + shared.esc(labName(p.g)) + " cluster" +
          (p.g === p.c ? " — same as the map"
            : " — disagrees with the map&rsquo;s color here");
        if (TT_AUTHORS[p.id]) {
          html += "<br><span style=\"color:" + shared.colors.muted +
            "\">✦ author of the two-truths paper this panel reenacts</span>";
        }
        return html;
      }
      // invisible, larger hit targets on top of dots + labels (small dots are hard to hover)
      function hitLayer(svg, dataArr, cxFn, cyFn) {
        return svg.append("g").selectAll("circle").data(dataArr).join("circle")
          .attr("cx", cxFn).attr("cy", cyFn)
          .attr("r", 9).attr("fill", "transparent");
      }
      function wireHover(hits, ownDots, twinDots) {
        hits
          .on("mouseenter", function (evt, p) {
            resetDots();
            ownDots.filter(function (q) { return q.id === p.id; })
              .attr("stroke", shared.colors.ink).attr("stroke-width", 1.4)
              .attr("fill-opacity", 1);
            var twin = twinDots.filter(function (q) { return q.id === p.id; });
            twin.attr("stroke", shared.colors.ink).attr("stroke-width", 1.4)
              .attr("fill-opacity", 1);
            pulse(twin);
            map.highlight(p.id);
            shared.tooltip.show(tipFor(p), evt);
          })
          .on("mousemove", function (evt, p) { shared.tooltip.show(tipFor(p), evt); })
          .on("mouseleave", function () {
            resetDots();
            map.highlight(null);
            shared.tooltip.hide();
          });
      }
      wireHover(hitLayer(aseSvg, nodes,
        function (p) { return ax(sq(p.ase[0])); },
        function (p) { return ay(sq(p.ase[1])); }), aseDots, lseDots);
      wireHover(hitLayer(lseSvg, nodes,
        function (p) { return lx(p.lse[0]); },
        function (p) { return ly(p.lse[1]); }), lseDots, aseDots);
    }
  });
})();
