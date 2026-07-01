/* assets/js/analyses/dual-citizens.js — soft community membership (LSE + GMM responsibilities) */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("dual-citizens", {
    prose: {
      intro:
        "<p>Every person on this map wears exactly one lab color, but collaboration is rarely " +
        "that tidy — some people genuinely have a foot in two groups. So instead of asking the " +
        "model <em>who belongs where</em>, this panel asks <em>how sure it is</em>: each " +
        "person's fractional membership in all three labs, most divided people first.</p>",
      how:
        "<p>The lab colors across this site come from one pipeline: embed the network so every " +
        "person becomes a point in 7-dimensional space (people who collaborate a lot land close " +
        "together), then fit a 3-component Gaussian mixture over those points. A hard community " +
        "label is just the argmax of that mixture — but the model underneath assigns every " +
        "person a <em>responsibility</em> for each cluster, and reading those instead of the " +
        "argmax is like reading the full softmax instead of the top-1 token.</p>" +
        "<p>Each row below shows one person's <em>runner-up share</em> — the slice of them the " +
        "model would hand to a second lab — on a log scale, because the split is so lopsided " +
        "nothing else would survive the plot. For years it ran at temperature zero: everyone at " +
        "least 99.99% one lab. That has just broken. Two people now sit visibly off the wall — " +
        "Logan Smith reads about 98% EleutherAI, 2% Bau (one part in 52), and Arvind Narayanan " +
        "about 99.6% Bau, 0.4% EleutherAI (one part in 240) — the map's first real dual citizens. " +
        "Behind them the trace collapses as before: by the sixth row the runner-up share is parts " +
        "per ten-million, and for 42 of the 107 it underflows floating point to exactly zero. All " +
        "the ambiguity still sits on the EleutherAI–Bau border — the Vogelstein frontier's largest " +
        "leak is one part in 10⁴³.</p>",
      method:
        "<p>Soft assignments from the site's own community pipeline: the weighted graph's " +
        "largest connected component (107 of 130 people), Laplacian spectral embedding " +
        "(graspologic, R-DAD, d&nbsp;=&nbsp;7), rows projected onto the unit sphere, then a " +
        "3-component full-covariance Gaussian mixture (seed 42). The fractions shown are " +
        "<code>GaussianMixture.predict_proba</code> — posterior responsibilities, with " +
        "components remapped to lab ids by anchor members. The fractional-membership reading " +
        "follows mixed membership stochastic blockmodels (Airoldi, Blei, Fienberg &amp; Xing, " +
        "<em>JMLR</em> 2008); the embed-then-cluster recipe is the random dot product graph " +
        "survey (Athreya, Fishkind, Tang, Priebe et&nbsp;al., <em>JMLR</em> 2018) — co-written " +
        "by a connector on this very map: Carey E. Priebe sits in the Vogelstein cluster, two " +
        "co-author hops from Joshua Vogelstein himself. Edges use fractional co-authorship " +
        "counting (each paper adds 1/n_authors per pair; suggested by Stella Biderman), which " +
        "for years kept big-team papers — EleutherAI's signature genre — from dominating the " +
        "border, so the embedding separated the labs almost perfectly. Two new core members " +
        "(Antonio Mari, Eric Todd) thickened the Bau side enough to re-open a hairline crack: " +
        "Logan Smith and Arvind Narayanan are the first people the model won't call cleanly. " +
        "Model and site colors now agree for 105 of 107 connected people; the two disagreements " +
        "are the deliberate site overrides that force Kola Ayonrinde and Jesse Hoogland into the " +
        "Bau cluster, where the raw model still reads their ties as EleutherAI. (Sheridan Feucht, " +
        "the override that used to stand out here, the model now reproduces on its own.) Caveat: " +
        "GMM posteriors on a 7-d embedding of 107 points are characteristically " +
        "overconfident, so read the tail figures as model arithmetic, not measured sociology — " +
        "&ldquo;one part in 46 billion&rdquo; means the model cannot tell that person apart " +
        "from a core member, nothing finer.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data;
      var C = shared.colors.community;
      var mixedSet = {};
      d.mixed.forEach(function (id) { mixedSet[id] = true; });

      function labLabel(c) {
        var lab = (shared.communities || []).filter(function (x) { return x.id === c; })[0];
        return lab ? lab.label : ["EleutherAI", "David Bau", "Joshua Vogelstein"][c];
      }
      var SUP = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵",
                  "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻" };
      function pow10(e) {
        return "10" + String(e).split("").map(function (ch) { return SUP[ch] || ""; }).join("");
      }
      function oneIn(s) {              // runner-up share -> "1 in 29,000" / "1 in 95 billion"
        if (!s) return "exactly 0";
        var n = 1 / s;
        var m = Math.pow(10, Math.floor(Math.log10(n)) - 1);
        n = Math.round(n / m) * m;     // two significant figures
        if (n >= 1e12) return "1 in " + +(n / 1e12).toPrecision(2) + " trillion";
        if (n >= 1e9) return "1 in " + +(n / 1e9).toPrecision(2) + " billion";
        if (n >= 1e6) return "1 in " + +(n / 1e6).toPrecision(2) + " million";
        return "1 in " + n.toLocaleString("en-US");
      }
      function tipHtml(id, rec) {      // rec = {c: top lab, j: runner-up lab, s: runner-up share}
        var rows = [0, 1, 2].map(function (c) {
          var v = c === rec.c ? ">99.99%"
                : (c === rec.j && rec.s > 0) ? "trace: " + oneIn(rec.s)
                : "0";
          return '<span style="color:' + C[c] + '">●</span> ' +
            shared.esc(labLabel(c)) + " " + v;
        });
        return "<strong>" + shared.esc(shared.labelOf(id)) + "</strong><br>" + rows.join("<br>");
      }

      // ---- layout: trace plot (flexible) + minimap column ----
      el.style.display = "flex";
      el.style.flexWrap = "wrap";
      el.style.gap = "20px";
      el.style.alignItems = "flex-start";
      var plotDiv = document.createElement("div");
      plotDiv.style.flex = "1 1 360px";
      plotDiv.style.minWidth = "320px";
      var mapDiv = document.createElement("div");
      mapDiv.style.flex = "0 0 210px";
      el.appendChild(plotDiv);
      el.appendChild(mapDiv);

      var mm = shared.minimap(mapDiv, function (id) {
        var n = d.nodes[id];
        return n ? C[n.c] : null;          // model's argmax lab; outside the LCC -> periphery
      }, {
        opacityFn: function (id) { return mixedSet[id] ? 0.95 : 0.35; },
        radiusFn: function (id) { return mixedSet[id] ? 3.4 : (shared.isList(id) ? 3.2 : 2.2); },
        onReady: function (api) {
          api.svg.selectAll("circle")
            .on("mouseenter", function (evt, n) {
              var rec = d.nodes[n.id];
              if (!rec) return;
              shared.tooltip.show(tipHtml(n.id, rec), evt);
              api.highlight(n.id);
            })
            .on("mousemove", function (evt, n) {
              var rec = d.nodes[n.id];
              if (rec) shared.tooltip.show(tipHtml(n.id, rec), evt);
            })
            .on("mouseleave", function () { shared.tooltip.hide(); api.highlight(null); });
        }
      });
      var cap = document.createElement("div");
      cap.className = "m-note";
      cap.style.maxWidth = "200px";
      cap.style.marginTop = "6px";
      cap.textContent = "All " + d.stats.n_lcc + " connected people, colored by the model's " +
        "strongest lab; the " + d.mixed.length + " most divided at full ink. Hover a dot for " +
        "its exact split.";
      mapDiv.appendChild(cap);

      // ---- the trace plot: each person's runner-up share on a log axis. Every
      //      trace in the data sits on the EleutherAI–Bau border (asserted at
      //      build time); the dot color is the runner-up lab. ----
      var availW = el.clientWidth || 700;
      var W = Math.max(340, Math.min(availW - 240, 620));
      if (availW < 580) W = Math.max(320, availW - 10);   // narrow: map wraps below
      var nameW = Math.min(150, Math.max(104, Math.round(W * 0.24)));
      var lblW = 92;
      var barW = Math.max(150, W - nameW - lblW - 16);
      var rowH = 24, topM = 40;
      var axisY = topM + d.mixed.length * rowH;
      var H = axisY + 44;
      var svg = d3.select(plotDiv).append("svg").attr("width", W).attr("height", H);
      var x0 = nameW + 8;
      var x = d3.scaleLog().domain([1e-12, 1]).range([x0, x0 + barW]);

      // column header (direct label, no legend) — name the runner-up lab if uniform
      var jSet = {};
      d.mixed.forEach(function (id) { jSet[d.nodes[id].j] = true; });
      var jUniq = Object.keys(jSet).map(Number);
      svg.append("text").attr("class", "m-axis")
        .attr("x", x0).attr("y", 12).attr("fill", jUniq.length === 1 ? C[jUniq[0]] : shared.colors.ink)
        .text("runner-up share, log scale" +
          (jUniq.length === 1 ? " — every trace points to " + labLabel(jUniq[0]) : ""));

      // reference lines: the 0.01% bar nobody clears (labeled with the ticks below),
      // and a true 50/50 dual citizen (labeled on top, alone, to avoid collisions)
      [1e-4, 0.5].forEach(function (v) {
        svg.append("line")
          .attr("x1", x(v)).attr("x2", x(v))
          .attr("y1", topM - 8).attr("y2", axisY)
          .attr("stroke", shared.colors.muted).attr("stroke-width", 1)
          .attr("stroke-dasharray", "2,3").attr("stroke-opacity", 0.6);
      });
      svg.append("text").attr("class", "m-note")
        .attr("x", x(0.5) + 4).attr("y", topM - 14)
        .attr("text-anchor", "end").attr("fill", shared.colors.muted)
        .text("50/50 — a true dual citizen");
      svg.append("text").attr("class", "m-axis")
        .attr("x", x(1e-4)).attr("y", axisY + 15).attr("text-anchor", "middle")
        .attr("fill", shared.colors.muted)
        .text("0.01%");

      // hairline axis + power-of-ten ticks
      svg.append("line")
        .attr("x1", x0).attr("x2", x0 + barW).attr("y1", axisY).attr("y2", axisY)
        .attr("stroke", shared.colors.hair).attr("stroke-width", 1);
      [-10, -8, -6, -2].forEach(function (e) {
        var xv = x(Math.pow(10, e));
        svg.append("line")
          .attr("x1", xv).attr("x2", xv).attr("y1", axisY).attr("y2", axisY + 4)
          .attr("stroke", shared.colors.hair).attr("stroke-width", 1);
        svg.append("text").attr("class", "m-axis")
          .attr("x", xv).attr("y", axisY + 15).attr("text-anchor", "middle")
          .attr("fill", shared.colors.muted)
          .text(pow10(e));
      });
      svg.append("text").attr("class", "m-note")
        .attr("x", x0).attr("y", axisY + 32).attr("fill", shared.colors.muted)
        .text("two dots now clear the 0.01% line; a true 50/50 split is still a decade and a half further right");

      var rows = svg.selectAll("g.dcrow").data(d.mixed).join("g").attr("class", "dcrow")
        .attr("transform", function (id, i) { return "translate(0," + (topM + i * rowH) + ")"; });

      rows.append("text").attr("class", "m-label")
        .attr("x", nameW).attr("y", rowH / 2 + 4).attr("text-anchor", "end")
        .attr("fill", shared.colors.ink)
        .text(function (id) { return shared.labelOf(id); });

      rows.append("line")                      // light track so the eye can cross 7 decades
        .attr("x1", x0).attr("x2", x0 + barW)
        .attr("y1", rowH / 2).attr("y2", rowH / 2)
        .attr("stroke", "#efe9dc").attr("stroke-width", 1);
      rows.append("circle")
        .attr("cx", function (id) { return x(d.nodes[id].s); })
        .attr("cy", rowH / 2).attr("r", 4)
        .attr("fill", function (id) { return C[d.nodes[id].j]; })
        .attr("fill-opacity", 0.9);

      rows.append("text").attr("class", "m-axis")
        .attr("x", x0 + barW + 8).attr("y", rowH / 2 + 4)
        .attr("fill", function (id) { return C[d.nodes[id].j]; })
        .text(function (id) { return oneIn(d.nodes[id].s); });

      // generous hover targets: whole row -> focus row, tooltip, minimap highlight
      rows.append("rect")
        .attr("x", 0).attr("y", 0).attr("width", W).attr("height", rowH)
        .attr("fill", "transparent")
        .on("mouseenter", function (evt, id) {
          var i = d.mixed.indexOf(id);
          rows.interrupt().attr("opacity", function (q, j) { return j === i ? 1 : 0.45; });
          shared.tooltip.show(tipHtml(id, d.nodes[id]), evt);
          mm.highlight(id);
        })
        .on("mousemove", function (evt, id) { shared.tooltip.show(tipHtml(id, d.nodes[id]), evt); })
        .on("mouseleave", function () {
          rows.interrupt().attr("opacity", 1);
          shared.tooltip.hide();
          mm.highlight(null);
        });

      var foot = document.createElement("div");
      foot.className = "m-note";
      foot.style.marginTop = "8px";
      foot.textContent = "Below the top two, these rows fall away fast — beneath one part in a " +
        "million by the fourth, and for " + d.stats.n_zero + " of " + d.stats.n_lcc + " people " +
        "the runner-up share underflows floating point to exactly zero. The biggest leak " +
        "into the " + labLabel(2) + " cluster from outside is one part in " +
        pow10(Math.round(Math.log10(1 / d.stats.vog_spill))) + ": every crack there is, the two " +
        "real dual citizens included, sits on the one seam, " + labLabel(0) + "–" + labLabel(1) + ".";
      plotDiv.appendChild(foot);
    }
  });
})();
