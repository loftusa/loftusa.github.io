/* assets/js/analyses-affiliations/the-pipeline.js — careers as a five-token language:
   org-type bigram matrix, the academia↔company flows, and the recent crossings */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("the-pipeline", {
    prose: {
      intro:
        "<p>Strip away the org names and every career in this chat reads like a sentence " +
        "over a five-word vocabulary: <em>university, lab, company, program, community</em>. " +
        "This panel learns the grammar of those sentences — given the word you are standing " +
        "on, which word tends to come next? Mostly the answer is <em>company</em>: that is " +
        "the gravity of the field right now. The moves back the other way are the rarer, " +
        "braver token, and they get equal billing here.</p>",
      how:
        "<p>This is literally a bigram language model with a five-token vocabulary. Take each " +
        "person's dated affiliations, sort them by start year, read the sequence of " +
        "institution <em>types</em> as a token stream, and count next-token frequencies — " +
        "exactly how you would estimate bigram statistics from a corpus. Two affiliations " +
        "that start the same year are one state held jointly, not a move, so they contribute " +
        "no transition.</p>" +
        "<p>Each row of the matrix is then the next-token distribution given where you " +
        "stand — its next-token distribution. The <em>university</em> row puts " +
        "most of its probability on <em>company</em>, and the <em>program</em> " +
        "row is sharper still: half of everything that follows a fellowship or research " +
        "program is a company. The two arrows pull those flows out of the matrix — 25 " +
        "crossings from academia into companies, 7 going back.</p>",
      method:
        "<p>First-order Markov chain over institution types, with maximum-likelihood " +
        "transition probabilities — i.e. row-normalized bigram counts (Anderson &amp; " +
        "Goodman, &ldquo;Statistical Inference about Markov Chains,&rdquo; <em>Annals of " +
        "Mathematical Statistics</em>, 1957); reading careers as such chains follows the " +
        "mover&ndash;stayer tradition (Blumen, Kogan &amp; McCarthy, <em>The Industrial " +
        "Mobility of Labor as a Probability Process</em>, 1955). Caveats: only dated " +
        "affiliations are visible — undated ones (sparse public CVs) leave no " +
        "transitions, so every count is an undercount; same-year starts are treated as " +
        "simultaneous holdings, not moves; the diagonal is real movement too (company&rarr;company " +
        "transitions are job changes, not noise); and first-order means " +
        "memoryless, which careers are not. n is small — read the rows as observed " +
        "frequencies, not destiny.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, C = shared.colors;
  // org-type palette — keep in sync across the analyses-affiliations panels + the map page
      var TYPE_COLORS = {
        lab: "#8c510a", program: "#2166ac", company: "#7b3294",
        community: "#35978f", university: "#98917f"
      };
      var types = d.types, counts = d.counts, probs = d.probs, rows = d.row_sums;

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "16px 28px").style("align-items", "flex-start");
      var matrixBox = wrap.append("div").style("flex", "0 0 auto");
      var arrowBox = wrap.append("div")
        .style("flex", "1 1 200px").style("min-width", "190px").style("max-width", "280px");
      var sideBox = wrap.append("div").style("flex", "0 0 200px");
      var crossBox = wrap.append("div").style("flex", "1 1 100%");

      // ---- mini-map: everyone tinted by the type of their latest dated affiliation ----
      var baseOp = function (id) { return d.latest_type[id] ? 0.92 : 0.35; };
      var mini = shared.minimap(sideBox.node(), function (id) {
        var t = d.latest_type[id];
        return t ? TYPE_COLORS[t] : null;
      }, { opacityFn: baseOp });
      var capWords = types.map(function (t) {
        return "<span style='color:" + TYPE_COLORS[t] + "'>" + t + "</span>";
      }).join(" · ");
      sideBox.append("div").attr("class", "m-map-cap")
        .html("each person tinted by their latest start:<br>" + capWords);

      // highlight an arbitrary SET on the mini-map (built-in highlight is single-id)
      function highlightSet(ids) {
        var circles = mini.svg.selectAll("circle");
        if (!circles.size()) return;
        circles.interrupt();
        if (!ids) {
          circles.attr("stroke", null)
            .attr("fill-opacity", function (n) { return baseOp(n.id); });
          return;
        }
        var set = {};
        ids.forEach(function (id) { set[id] = true; });
        circles
          .attr("stroke", function (n) { return set[n.id] ? C.ink : null; })
          .attr("stroke-width", 1.4)
          .attr("fill-opacity", function (n) { return set[n.id] ? 1 : 0.18; });
      }

      // ---- 5×5 transition matrix: ink opacity ∝ row probability, count printed ----
      var cell = 44, gL = 96, gT = 58, gR = 64;
      var mW = gL + 5 * cell + gR, mH = gT + 5 * cell + 8;
      var svg = matrixBox.append("svg").attr("width", mW).attr("height", mH);

      types.forEach(function (t, j) {        // column heads: dot + rotated label
        var cx = gL + j * cell + cell / 2;
        svg.append("circle").attr("cx", cx).attr("cy", gT - 8).attr("r", 3)
          .attr("fill", TYPE_COLORS[t]);
        svg.append("text").attr("class", "m-axis")
          .attr("transform", "translate(" + (cx + 2) + "," + (gT - 16) + ") rotate(-35)")
          .attr("text-anchor", "start").text(t);
      });
      svg.append("text").attr("class", "m-axis")
        .attr("x", gL + 5 * cell + 10).attr("y", gT - 8).text("moves out");

      types.forEach(function (t, i) {        // row heads: label + dot, then the cells
        var cy = gT + i * cell + cell / 2;
        svg.append("text").attr("class", "m-axis")
          .attr("x", gL - 18).attr("y", cy + 4).attr("text-anchor", "end").text(t);
        svg.append("circle").attr("cx", gL - 10).attr("cy", cy).attr("r", 3)
          .attr("fill", TYPE_COLORS[t]);
        svg.append("text").attr("class", "m-axis")
          .attr("x", gL + 5 * cell + 10).attr("y", cy + 4).text(rows[i]);
      });

      function cellTip(i, j) {
        var key = types[i] + ">" + types[j];
        var head = "<span class='t-name'>" + types[i] + " &rarr; " + types[j] + "</span>" +
          "<div class='t-sub'>" + counts[i][j] + " of " + rows[i] + " moves out of a " +
          types[i] + " · p = " + probs[i][j].toFixed(2) + "</div>";
        var ex = d.cells[key] || [];
        var body = ex.map(function (e) {
          return "<div style='margin-top:2px'><span style='color:" + C.muted + "'>" +
            "’" + String(e.y0).slice(2) + "→’" + String(e.y1).slice(2) +
            "</span> &nbsp;" + shared.esc(shared.labelOf(e.id)) + " — " +
            shared.esc(e.from) + " → " + shared.esc(e.to) + "</div>";
        }).join("");
        if (counts[i][j] > ex.length) {
          body += "<div class='t-sub'>… and " + (counts[i][j] - ex.length) + " more</div>";
        }
        if (!counts[i][j]) body = "<div class='t-sub'>no recorded moves</div>";
        return head + body;
      }

      types.forEach(function (ft, i) {
        types.forEach(function (tt, j) {
          var x = gL + j * cell, y = gT + i * cell;
          var rect = svg.append("rect")
            .attr("x", x + 0.5).attr("y", y + 0.5)
            .attr("width", cell - 1).attr("height", cell - 1)
            .attr("fill", C.ink).attr("fill-opacity", probs[i][j] * 0.92)
            .attr("stroke", "#efe9dc").attr("stroke-width", 1);
          svg.append("text")
            .attr("class", counts[i][j] ? "m-label" : "m-axis")
            .attr("x", x + cell / 2).attr("y", y + cell / 2 + 4)
            .attr("text-anchor", "middle").style("pointer-events", "none")
            .text(counts[i][j]);
          rect.on("mouseenter", function (evt) {
            rect.attr("stroke", C.ink).attr("stroke-width", 1.4);
            highlightSet((d.cells[ft + ">" + tt] || []).map(function (e) { return e.id; }));
            shared.tooltip.show(cellTip(i, j), evt);
          }).on("mousemove", function (evt) {
            shared.tooltip.show(cellTip(i, j), evt);
          }).on("mouseleave", function () {
            rect.attr("stroke", "#efe9dc").attr("stroke-width", 1);
            highlightSet(null);
            shared.tooltip.hide();
          });
        });
      });
      matrixBox.append("div").attr("class", "m-axis")
        .style("color", C.muted).style("margin-top", "2px")
        .text("rows: where you are · columns: where the next start lands · " +
          "darker ink = larger share of that row");

      // ---- the two flows, pulled out: band thickness ∝ moves (Minard-style) ----
      var aW = Math.max(190, Math.min(arrowBox.node().clientWidth || 220, 260));
      var aSvg = arrowBox.append("svg").attr("width", aW).attr("height", 126);
      function arrow(yc, th, rightward, color) {
        var h = th / 2, hd = Math.max(12, th * 0.55), wg = Math.max(4, th * 0.3);
        var x0 = 8, x1 = aW - 10, pts;
        if (rightward) {
          pts = [[x0, yc - h], [x1 - hd, yc - h], [x1 - hd, yc - h - wg], [x1, yc],
                 [x1 - hd, yc + h + wg], [x1 - hd, yc + h], [x0, yc + h]];
        } else {
          pts = [[x1, yc - h], [x0 + hd, yc - h], [x0 + hd, yc - h - wg], [x0, yc],
                 [x0 + hd, yc + h + wg], [x0 + hd, yc + h], [x1, yc + h]];
        }
        return aSvg.append("polygon")
          .attr("points", pts.map(function (p) { return p.join(","); }).join(" "))
          .attr("fill", color).attr("fill-opacity", 0.82);
      }
      aSvg.append("text").attr("class", "m-label").attr("x", 8).attr("y", 16)
        .text("into companies — " + d.acad_to_co + " moves");
      var fwd = arrow(40, d.acad_to_co, true, TYPE_COLORS.company);
      var back = arrow(92, d.co_to_acad, false, TYPE_COLORS.lab);
      aSvg.append("text").attr("class", "m-label").attr("x", aW - 10).attr("y", 116)
        .attr("text-anchor", "end").text("back to academia — " + d.co_to_acad);

      var fwdTip = "<span class='t-name'>academia &rarr; company</span>" +
        "<div class='t-sub'>university → company " + counts[0][2] +
        " · lab → company " + counts[1][2] + "</div>";
      var backTip = "<span class='t-name'>company &rarr; academia</span>" +
        "<div class='t-sub'>company → university " + counts[2][0] +
        " · company → lab " + counts[2][1] + "</div>";
      [[fwd, fwdTip], [back, backTip]].forEach(function (pair) {
        pair[0].on("mouseenter", function (evt) { shared.tooltip.show(pair[1], evt); })
          .on("mousemove", function (evt) { shared.tooltip.show(pair[1], evt); })
          .on("mouseleave", function () { shared.tooltip.hide(); });
      });
      arrowBox.append("div").attr("class", "m-axis")
        .style("color", C.muted).style("display", "block").style("margin-top", "4px")
        .text("band thickness ∝ moves · both directions are real careers — " +
          "the return trip is just the rarer token");

      // ---- recent crossings: academia → company, 2024 on ----
      crossBox.append("div").attr("class", "m-axis")
        .style("color", C.muted).style("margin-bottom", "2px")
        .text("recent crossings — out of a university or lab, into a company, since 2024");
      var moverRows = [];
      d.movers.forEach(function (m, mi) {
        var row = crossBox.append("div")
          .style("display", "flex").style("flex-wrap", "wrap")
          .style("align-items", "baseline").style("gap", "2px 10px")
          .style("padding", "4px 0")
          .style("border-top", "1px solid " + (mi === 0 ? C.hair : "#efe9dc"));
        moverRows.push(row);
        row.append("span").attr("class", "m-axis")
          .style("flex", "0 0 30px").style("color", C.muted)
          .text("’" + String(m.year).slice(2));
        row.append("span").attr("class", "m-label")
          .style("flex", "0 0 170px").style("color", C.ink)
          .html(shared.seatLink(m.id));
        var route = row.append("span").attr("class", "m-label");
        route.append("span").style("color", C.muted)
          .text(m.from + " → ");
        route.append("span").style("color", C.ink).style("font-weight", "600")
          .text(m.to);
        var tip = "<span class='t-name'>" + shared.esc(shared.labelOf(m.id)) + "</span>" +
          "<div class='t-sub'>" + shared.esc(m.from) + " → " + shared.esc(m.to) +
          ", " + m.year + " — a new chapter</div>";
        row.on("mouseenter", function (evt) {
          moverRows.forEach(function (r, j) { r.style("opacity", j === mi ? 1 : 0.45); });
          mini.highlight(m.id);
          shared.tooltip.show(tip, evt);
        }).on("mousemove", function (evt) {
          shared.tooltip.show(tip, evt);
        }).on("mouseleave", function () {
          moverRows.forEach(function (r) { r.style("opacity", 1); });
          mini.highlight(null);
          shared.tooltip.hide();
        });
      });
      var tally = {};
      d.movers.forEach(function (m) { tally[m.to] = (tally[m.to] || 0) + 1; });
      var top = Object.keys(tally).sort(function (a, b) { return tally[b] - tally[a]; })[0];
      if (tally[top] >= 2) {
        crossBox.append("div").attr("class", "m-note")
          .html("<strong>" + shared.esc(top) + "</strong> alone accounts for " + tally[top] +
            " of these " + d.movers.length + " crossings — one company quietly collecting " +
            "this chat's alumni. Hover a row to find that person on the map.");
      }
    }
  });
})();
