/* assets/js/analyses/two-witnesses.js */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("two-witnesses", {
    prose: {
      intro:
        "<p>Every connection on this map was reported by two independent record-keepers " +
        "— Semantic Scholar and OpenAlex — which crawl the literature separately and never " +
        "confer. Treat them as two witnesses to the same events: where their accounts " +
        "agree, the map is solid; where they differ, you have found a bug in a database, " +
        "never a fact about a person. This panel cross-examines them — globally, with a " +
        "formal two-sample test, and then person by person.</p>",
      how:
        "<p>Think of the two databases as two training runs over the same data: same " +
        "underlying process, different noise. You don&rsquo;t compare runs by diffing raw " +
        "weights — you embed both into a latent space, align them (Procrustes, the same " +
        "trick used to align embeddings across seeds or checkpoints), and measure what " +
        "alignment can&rsquo;t explain. That is exactly the latent-position test: " +
        "spectrally embed each witness&rsquo;s graph, find the best rotation between the " +
        "two embeddings, and compare the leftover gap to a null built by re-sampling " +
        "graphs from each witness&rsquo;s own account 200 times. Here the witnesses pass " +
        "emphatically — the observed gap is smaller than every single one of the 400 " +
        "re-tellings, so statistically this is one network, told twice.</p>" +
        "<p>The person-level score is simpler: each person&rsquo;s row of the adjacency " +
        "matrix is their collaboration fingerprint, and the cosine between their two " +
        "fingerprints — like cosine between the same token&rsquo;s embedding in two " +
        "models — localizes exactly whom the witnesses disagree about. Every low score " +
        "here traces to a filing artifact — a coverage gap, an unmerged preprint, a " +
        "split author profile — which makes this less a verdict than a debugging tool " +
        "for the map itself.</p>",
      method:
        "<p>Global test: the semiparametric two-sample latent-position test of Tang, " +
        "Athreya, Sussman, Lyzinski &amp; Priebe, &ldquo;A semiparametric two-sample " +
        "hypothesis testing problem for random graphs&rdquo;, <em>Journal of " +
        "Computational and Graphical Statistics</em> 26(2), 2017 (graspologic " +
        "implementation): adjacency spectral embedding (d&nbsp;=&nbsp;6, Zhu&ndash;Ghodsi " +
        "elbow), orthogonal-Procrustes alignment, rotation test case, parametric " +
        "bootstrap with 200 resamples per side (p-value floor &asymp; 0.005). Carey E. " +
        "Priebe, the paper&rsquo;s senior author, is a node on this very map — and his " +
        "edge with Joshua T. Vogelstein is one of the two records the databases dispute " +
        "most. The test&rsquo;s bootstrap resamples binary Bernoulli graphs from the " +
        "estimated edge-probability matrix, so it runs on the binarized layers — the " +
        "regime it was built for. Edges here use fractional co-authorship counting (each " +
        "paper adds 1/n<sub>authors</sub> per co-author pair; suggested by Stella Biderman), " +
        "so the raw weights are small: only ~5% of that probability matrix now falls outside " +
        "[0,&nbsp;1] (the old integer counts pushed ~23% out), yet a rank-transformed " +
        "weighted graph is still incommensurable with the binary null — it puts every " +
        "bootstrap draw above the observed (p&nbsp;=&nbsp;1.0 by regime mismatch, not " +
        "agreement). Binarizing sidesteps both, and because the topology is identical across " +
        "weightings this verdict does not move. Caveat: the two layers witness the " +
        "<em>same</em> underlying papers, so they are positively dependent and the " +
        "independent-samples null is conservative — the observed statistic (2.15) fell below " +
        "all 400 null draws (2.36&ndash;5.32), so &ldquo;same network&rdquo; is an emphatic " +
        "verdict, not a marginal one. Person-level scores are cosine similarities between the " +
        "fractional-weighted adjacency rows, so a few-author paper weighs more than a " +
        "many-author one: a single unmatched small paper can pull a score down even where the " +
        "databases otherwise agree (e.g. C. McDougall, who differs by just one paper). People " +
        "indexed by only one database get flagged, not scored. The per-person " +
        "spirit follows Bridgeford et al., &ldquo;Eliminating accidental deviations to " +
        "minimize generalization error and maximize replicability: applications in " +
        "connectomics&rdquo;, <em>PLOS Computational Biology</em>, 2021 — " +
        "discriminability, i.e. whether repeated measurements of the same networks can " +
        "be told apart. Eric Bridgeford is on this map too.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data;
      var tailN = d.tail_n;
      var people = d.people.slice().sort(function (a, b) {
        return a.cos - b.cos || (a.id < b.id ? -1 : 1);
      });
      var tail = people.slice(0, tailN);
      var inTail = {};
      tail.forEach(function (p) { inTail[p.id] = p; });

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "14px 28px").style("align-items", "flex-start");
      var chartBox = wrap.append("div").style("flex", "1 1 340px").style("min-width", "300px");
      var sideBox = wrap.append("div").style("flex", "0 0 auto").style("width", "200px");

      // ---- minimap anchor: the eight disputed records, in lab colors ----
      var mm = shared.minimap(sideBox.node(), function (id) {
        return inTail[id] ? shared.colorOf(id) : null;
      }, {
        radiusFn: function (id) { return inTail[id] ? 4 : 2.1; },
        opacityFn: function (id) { return inTail[id] ? 1 : 0.5; }
      });
      sideBox.append("div").attr("class", "m-map-cap")
        .text("the eight disputed records, on the map");
      sideBox.append("div").attr("class", "m-map-cap")
        .style("margin-top", "8px").style("text-align", "left")
        .text("Formal verdict: the gap between the two accounts (" +
          d.test.stat + ") is smaller than all " + (2 * d.test.n_boot) +
          " bootstrap re-tellings (" + d.test.null_min + "–" + d.test.null_max +
          "). p = " + d.test.p.toFixed(2) + ": one network, told twice.");

      // ---- sorted strip plot: agreement per person, disputed tail pulled out ----
      var W = Math.max(chartBox.node().clientWidth || 0, 300);
      var wide = W >= 520;
      var mL = 34, mR = 12, mT = 16, mB = 30, H = 330;
      var x = d3.scaleLinear().domain([0, people.length - 1]).range([mL + 8, W - mR - 6]);
      var y = d3.scaleLinear().domain([0, 1]).range([H - mB, mT]);
      var svg = chartBox.append("svg").attr("width", W).attr("height", H);

      // hairline y-axis, three ticks, one reference line at perfect agreement
      var ax = svg.append("g");
      ax.append("line").attr("x1", mL).attr("x2", mL).attr("y1", y(1)).attr("y2", y(0))
        .attr("stroke", shared.colors.hair);
      [0, 0.5, 1].forEach(function (t) {
        ax.append("line").attr("x1", mL - 3).attr("x2", mL).attr("y1", y(t)).attr("y2", y(t))
          .attr("stroke", shared.colors.hair);
        ax.append("text").attr("class", "m-axis").attr("x", mL - 6).attr("y", y(t) + 3)
          .attr("text-anchor", "end").text(t === 0.5 ? ".5" : t);
      });
      svg.append("line").attr("x1", mL).attr("x2", W - mR).attr("y1", y(1)).attr("y2", y(1))
        .attr("stroke", "#efe9dc");
      svg.append("text").attr("class", "m-axis").attr("x", mL + 8).attr("y", y(1) - 5)
        .text("1 = the two databases tell an identical story about you");
      svg.append("text").attr("class", "m-axis").attr("x", mL + 8).attr("y", H - 8)
        .text("← the " + tailN + " records the witnesses dispute");
      svg.append("text").attr("class", "m-axis").attr("x", W - mR).attr("y", H - 8)
        .attr("text-anchor", "end")
        .text(people.length + " people known to both, sorted by agreement →");

      // the quiet band, labeled directly
      svg.append("text").attr("class", "m-anno")
        .attr("x", x(Math.round(people.length * 0.62)))
        .attr("y", y(0.83))
        .attr("text-anchor", "middle")
        .text("the quiet band: " + Math.round(100 * d.band.frac_above_90) +
          "% agree above 0.9");

      var dots = svg.append("g").selectAll("circle").data(people).join("circle")
        .attr("cx", function (p, i) { return x(i); })
        .attr("cy", function (p) { return y(p.cos); })
        .attr("r", function (p) { return inTail[p.id] ? 4 : 2.6; })
        .attr("fill", function (p) { return shared.colorOf(p.id); })
        .attr("fill-opacity", function (p) { return inTail[p.id] ? 1 : 0.45; });

      // ---- annotate the tail: name + likely cause, hairline leaders ----
      var lx = x(tailN + 1.5);
      var ly = y(0) + 4;                       // greedy stack, bottom-up, >=16px apart
      tail.forEach(function (p, i) {
        ly = Math.min(y(p.cos) + 4, ly - 17);
        var t = svg.append("text").attr("class", "m-anno")
          .attr("x", lx + 14).attr("y", ly);
        t.append("tspan").text(shared.labelOf(p.id));
        if (wide) t.append("tspan").attr("fill", shared.colors.muted)
          .text(" · " + p.cause);
        svg.append("line")
          .attr("x1", x(i) + 5).attr("y1", y(p.cos))
          .attr("x2", lx + 10).attr("y2", ly - 4)
          .attr("stroke", shared.colors.hair).attr("stroke-width", 1);
      });

      // ---- hover: tooltip + minimap highlight ----
      var step = (x(1) - x(0));
      var tipFor = function (p) {
        var lab = shared.communityOf(p.id), labName = "";
        (shared.communities || []).forEach(function (c) { if (c.id === lab) labName = c.label; });
        var html = "<span class=\"t-name\">" + shared.esc(shared.labelOf(p.id)) + "</span>" +
          "<div class=\"t-sub\">" + shared.esc(labName || "unaffiliated") +
          " · " + p.ds2 + " connection" + (p.ds2 === 1 ? "" : "s") +
          " in Semantic Scholar · " + p.doa + " in OpenAlex</div>" +
          "agreement between the two accounts: <strong>" + p.cos.toFixed(2) + "</strong>";
        if (p.note) {
          html += "<div class=\"t-sub\" style=\"max-width:260px\">" +
            shared.esc(p.cause) + " — " + shared.esc(p.note) + "</div>";
        }
        return html;
      };
      svg.append("g").selectAll("rect").data(people).join("rect")
        .attr("x", function (p, i) { return x(i) - step / 2; })
        .attr("y", mT - 6).attr("width", step).attr("height", H - mT - mB + 12)
        .attr("fill", "transparent")
        .on("mouseenter", function (evt, p) {
          dots.interrupt();
          dots.attr("stroke", function (q) { return q.id === p.id ? shared.colors.ink : null; })
            .attr("stroke-width", 1.4)
            .attr("fill-opacity", function (q) {
              return q.id === p.id ? 1 : (inTail[q.id] ? 0.55 : 0.18);
            });
          mm.highlight(p.id);
          shared.tooltip.show(tipFor(p), evt);
        })
        .on("mousemove", function (evt, p) { shared.tooltip.show(tipFor(p), evt); })
        .on("mouseleave", function () {
          dots.interrupt();
          dots.attr("stroke", null)
            .attr("fill-opacity", function (q) { return inTail[q.id] ? 1 : 0.45; });
          mm.highlight(null);
          shared.tooltip.hide();
        });

      // ---- footnotes: the biggest disputes + the single-witness records ----
      var dsp = d.dispute;
      var names = function (rows) {
        return rows.map(function (r) { return shared.seatLink(r.id); }).join(", ");
      };
      var s2only = d.single.filter(function (r) { return r.witness === "s2"; });
      var oaonly = d.single.filter(function (r) { return r.witness === "oa"; });
      d3.select(el).append("div").attr("class", "m-note").html(
        "The ledger&rsquo;s two biggest disputes, each off by seven papers: <strong>" +
        shared.seatLink(dsp.a) + " &amp; " + shared.seatLink(dsp.b) +
        "</strong> — " + dsp.s2 + " shared papers says Semantic Scholar, " + dsp.oa +
        " says OpenAlex — and " + shared.seatLink(dsp.tie.a) + " &amp; " +
        shared.seatLink(dsp.tie.b) + " (" + dsp.tie.s2 + " vs " + dsp.tie.oa +
        "). Both are filing artifacts: unmerged preprint versions and split profiles, " +
        "not anyone&rsquo;s doing.<br><br>" +
        "Single-witness records — no agreement score is possible, and that itself is a " +
        "coverage finding: only Semantic Scholar knows the connections of " +
        names(s2only) + " here; only OpenAlex knows " + names(oaonly) + "&rsquo;s."
      );
    }
  });
})();
