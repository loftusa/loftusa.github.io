/* assets/js/analyses/lab-ledger.js */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("lab-ledger", {
    prose: {
      intro:
        "<p>Collapse everyone in the three labs into their lab — the unaffiliated periphery " +
        "holds no edges here — and 312 coauthorship edges " +
        "become a nine-number ledger: how much each lab trades with itself and with the " +
        "other two. The question is whether any of that trade is more than what each " +
        "lab&rsquo;s sheer size and star power would produce on their own. The two-block " +
        "reading is already visible in the raw counts: of the 312 edges, not one runs " +
        "directly between EleutherAI and the Vogelstein lab, while the interpretability " +
        "seam between EleutherAI and the Bau lab carries 37 direct edges with a summed " +
        "tie strength of 5.4.</p>",
      how:
        "<p>Raw totals mislead here, because a lab full of prolific hubs racks up " +
        "cross-lab weight by volume alone — the way a high-norm activation dominates a " +
        "raw dot product whether or not it points anywhere interesting. So we normalize: " +
        "each cell of the ledger is divided by the weight you&rsquo;d expect if every " +
        "person kept exactly their total collaboration weight but rewired <em>who</em> they " +
        "collaborate with completely at random.</p>" +
        "<p>That denominator is the degree correction in a degree-corrected stochastic " +
        "blockmodel — the graph analogue of normalizing activation magnitudes before " +
        "comparing directions, cosine similarity instead of raw dot products. What " +
        "survives is pure preference: a cell at 1.0 means hub sizes alone explain the " +
        "traffic, above 1.0 means the labs seek each other out, below 1.0 means a wall. " +
        "On this map the diagonal glows at 2.3&ndash;4.5&times; chance and almost everything " +
        "off it sits far below — except one channel.</p>",
      method:
        "<p>Degree-corrected stochastic blockmodel block matrix &mdash; Karrer &amp; " +
        "Newman, &ldquo;Stochastic blockmodels and community structure in networks&rdquo;, " +
        "<em>Physical Review E</em> 83, 016107 (2011) &mdash; with the block assignment " +
        "fixed to the page&rsquo;s three communities rather than re-fit. Observed " +
        "W<sub>ab</sub> sums edge weights between labs; edges use fractional " +
        "co-authorship counting &mdash; each paper adds 1/n<sub>authors</sub> to each of " +
        "its co-author pairs (suggested by Stella Biderman) &mdash; so one many-author " +
        "paper no longer counts as several units: the EleutherAI&ndash;Bau cell&rsquo;s " +
        "5.4 units of tie strength come from 21 distinct papers. The null is the " +
        "weighted configuration model, E<sub>ab</sub> = d<sub>a</sub>d<sub>b</sub>/2m off " +
        "the diagonal and d<sub>a</sub><sup>2</sup>/4m on it, so the printed ratio " +
        "W<sub>ab</sub>/E<sub>ab</sub> is proportional to the DCSBM mixing-matrix MLE. " +
        "The ledger&rsquo;s thinnest channel, Bau&ndash;Vogelstein, holds three recorded " +
        "papers &mdash; Agents of Chaos, plus NNsight/NDIF counted twice under two title " +
        "variants the databases never merged &mdash; and every one of its ten edges " +
        "touches Alexander R. Loftus, whom this re-clustering itself moved to the " +
        "Vogelstein side. Caveats: the communities were clustered from this same graph, " +
        "so diagonal ratios above 1 are partly by construction &mdash; the off-diagonal " +
        "comparisons are the informative part; and a zero is a statement about what the " +
        "databases recorded, never about the people.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data;
      var labs = d.labs;
      var C = shared.colors;
      var labColor = function (i) { return C.community[labs[i].id % C.community.length]; };

      var maxRatio = d3.max(d.ratio.flat());
      var tint = d3.scaleLinear().domain([0, maxRatio]).range([0, 0.2]);
      var fmtRatio = function (r) {
        if (r === 0) return "0";
        if (r >= 10) return r.toFixed(0);
        if (r >= 1) return r.toFixed(1);
        if (r >= 0.095) return r.toFixed(2);
        return r.toFixed(3);
      };

      var w = el.clientWidth || 720;
      var labelCol = 116;
      var avail = Math.min(w, 760) - labelCol - (w >= 640 ? 240 : 0);
      var cellW = Math.max(72, Math.min(120, Math.floor(avail / 3) - 6));
      var cellH = Math.max(52, Math.round(cellW * 0.6));

      var root = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "12px 36px").style("align-items", "flex-start");

      // ---- corner anchor: the standard mini-map, colored by lab -----------------
      var right = root.append("div").style("order", "2");
      var mini = shared.minimap(right.node(), function (id) {
        var c = shared.communityOf(id);
        return c >= 0 ? C.community[c % C.community.length] : null;
      });
      right.append("div").attr("class", "m-note").style("max-width", "200px")
        .text("The three labs, same colors as the ledger. Hover a cross-lab cell to " +
          "light up its biggest carrier.");

      // ---- the 3×3 table-graphic: the printed ratios ARE the ink ----------------
      var left = root.append("div").style("order", "1");
      var grid = left.append("div")
        .style("display", "grid")
        .style("grid-template-columns", labelCol + "px repeat(3, " + cellW + "px)")
        .style("gap", "5px");

      grid.append("div"); // empty corner
      labs.forEach(function (lab, j) {
        grid.append("div").attr("class", "m-label")
          .style("color", labColor(j))
          .style("align-self", "end").style("text-align", "center")
          .style("padding-bottom", "2px")
          .text(lab.short);
      });

      var carrierTip = function (i, j) {
        var list = d.carriers[Math.min(i, j) + "" + Math.max(i, j)];
        if (!list || !list.length) return "";
        var top = list[0];
        if (top.share === 1) {
          return "<br><span style=\"color:" + C.muted + "\">every link in this cell " +
            "touches </span><strong>" + shared.esc(shared.labelOf(top.id)) + "</strong>";
        }
        return "<br><span style=\"color:" + C.muted + "\">most tie strength carried by</span> " +
          list.map(function (p) {
            return shared.esc(shared.labelOf(p.id)) + " (" + shared.fmt.num(p.w) + ")";
          }).join(", ");
      };
      var cellTip = function (i, j) {
        var within = i === j;
        var head = within
          ? "within " + shared.esc(labs[i].short)
          : shared.esc(labs[i].short) + " &harr; " + shared.esc(labs[j].short);
        var nPairs = within ? labs[i].n * (labs[i].n - 1) / 2 : labs[i].n * labs[j].n;
        var html = "<strong>" + head + "</strong><br>" +
          "tie strength " + shared.fmt.num(d.observed[i][j]) +
          (d.observed[i][j] > 0
            ? " &mdash; " + d.papers[i][j] + " distinct papers across " +
              d.edges[i][j] + " direct edges"
            : "") +
          "<br>random rewiring expects " + shared.fmt.num(d.expected[i][j]) +
          " &rarr; ratio <strong>" + fmtRatio(d.ratio[i][j]) + "</strong><br>" +
          "<span style=\"color:" + C.muted + "\">tie strength " +
          shared.fmt.sig(d.percap[i][j], 2) +
          " per possible pair (" + shared.fmt.num(nPairs) + " pairs)</span>";
        if (d.observed[i][j] === 0) {
          html += "<br><span style=\"color:" + C.muted + "\">no recorded papers — " +
            "every path between these labs runs through the Bau-lab side of the map</span>";
        } else if (!within) {
          html += carrierTip(i, j);
        }
        return html;
      };
      var topCarrierId = function (i, j) {
        if (i === j || d.observed[i][j] === 0) return null;
        var list = d.carriers[Math.min(i, j) + "" + Math.max(i, j)];
        return list && list.length ? list[0].id : null;
      };

      labs.forEach(function (rowLab, i) {
        var rl = grid.append("div")
          .style("align-self", "center").style("text-align", "right")
          .style("padding-right", "10px").style("line-height", "1.25");
        rl.append("div").attr("class", "m-label")
          .style("color", labColor(i)).text(rowLab.short);
        rl.append("div").attr("class", "m-axis").style("color", C.muted)
          .text(rowLab.n + " people");

        labs.forEach(function (colLab, j) {
          var r = d.ratio[i][j];
          var cell = grid.append("div")
            .style("height", cellH + "px")
            .style("display", "flex")
            .style("align-items", "center").style("justify-content", "center")
            .style("background", function () {
              var ink = d3.color(C.ink);
              return "rgba(" + ink.r + "," + ink.g + "," + ink.b + "," + tint(r).toFixed(3) + ")";
            })
            .style("border", i === j ? "1px solid " + C.hair : "1px solid transparent")
            .style("border-radius", "2px")
            .style("cursor", "default");
          var num = cell.append("div")
            .style("font-size", "18px").style("font-weight", "600")
            .style("color", r === 0 ? C.muted : C.ink)
            .style("font-variant-numeric", "tabular-nums")
            .text(fmtRatio(r));
          if (r !== 0) {
            num.append("span")
              .style("font-size", "12px").style("font-weight", "400")
              .style("color", C.muted).text("×");
          }
          cell.on("mousemove", function (evt) {
            shared.tooltip.show(cellTip(i, j), evt);
            mini.highlight(topCarrierId(i, j));
          }).on("mouseleave", function () {
            shared.tooltip.hide();
            mini.highlight(null);
          });
        });
      });

      left.append("div").attr("class", "m-note")
        .style("max-width", (labelCol + 3 * cellW + 15) + "px")
        .style("margin-top", "10px")
        .html("<strong>1.0</strong> = exactly what hub sizes alone predict. Darker = " +
          "further above chance; diagonal (outlined) = within-lab.");
    }
  });
})();
