/* assets/js/analyses/who-holds-it-together.js */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("who-holds-it-together", {
    prose: {
      intro:
        "<p>Every network has people it could lose without noticing — and a few it could not. " +
        "Here we remove each of the 125 people on this map, one at a time, and measure how much " +
        "of the connection between the three labs disappears with them. A high score is not about " +
        "publishing the most; it is about sitting exactly where the labs touch.</p>",
      how:
        "<p>This is the ablation move from interpretability, applied to people: knock out one " +
        "component, re-run, and measure what the system can no longer do. The component is a " +
        "person; the lost capability is lab-to-lab connection capacity. Squash each lab into a " +
        "single node, give every coauthorship edge a capacity equal to its tie strength — each " +
        "shared paper counts 1/n_authors, so a three-author paper binds a pair far tighter than " +
        "a fifty-author one — and ask how much collaboration capacity can flow between each pair " +
        "of labs: the classic max-flow problem. Delete one person, recompute, and the drop is " +
        "their removal cost.</p>" +
        "<p>Like a good head ablation, the score is honest about redundancy: a well-connected hub " +
        "whose links are duplicated elsewhere scores low because the network simply routes around " +
        "them, while a quietly placed person on the only path scores high. The baseline already " +
        "gives away one surprise — in this data EleutherAI and the Vogelstein lab have no direct " +
        "papers at all, so everything between them runs through the Bau-lab side of the map.</p>",
      method:
        "<p>Mediator ablation via s&ndash;t maximum flow on the weighted coauthorship graph " +
        "(NetworkX preflow-push). Edges use fractional co-authorship counting — each paper adds " +
        "1/n<sub>authors</sub> to each of its co-author pairs (suggested by Stella Biderman; " +
        "Newman, <em>Phys. Rev. E</em> 64, 2001 is the 1/(n&minus;1) variant) — so capacities " +
        "are tie strengths, not paper counts. Each lab is contracted to " +
        "a supernode with parallel capacities summed; between-lab connectivity is the sum of " +
        "max-flow over the three lab pairs, which by max-flow/min-cut equals the cheapest edge " +
        "cut separating each pair. Removal cost of person v is 1 &minus; flow(G&minus;v)/flow(G), " +
        "with a per-pair breakdown. The setup follows the attack-tolerance tradition of Albert, " +
        "Jeong &amp; Barab&aacute;si, &ldquo;Error and attack tolerance of complex networks&rdquo;, " +
        "<em>Nature</em> 406 (2000), and the key-player problem of Borgatti, &ldquo;Identifying " +
        "sets of key players in a social network&rdquo;, <em>Computational &amp; Mathematical " +
        "Organization Theory</em> 12 (2006). Secondary check: change in algebraic connectivity " +
        "(Fiedler value &lambda;&#8322;) of the largest component — which can <em>rise</em> after " +
        "a removal, when severing a weakly attached branch leaves a denser surviving core. " +
        "Caveats: capacities are built from distinct recorded papers with no recency weighting; " +
        "a zero here is a statement about the databases, never about a person; people with no " +
        "resolved coauthor edge in this graph necessarily score zero.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data;
      var people = d.people.slice(0, 12);
      var top1 = people[0];
      var maxCost = d3.max(people, function (p) { return p.cost; });

      var labName = function (c) {
        var m = (shared.communities || []).filter(function (x) { return x.id === c; })[0];
        return m ? m.label : "—";
      };
      var pairName = function (i) {
        return labName(d.pairs[i][0]) + " ↔ " + labName(d.pairs[i][1]);
      };
      var pctLoss = function (x) {
        var r = Math.round(100 * x);
        return r === 0 ? "0%" : "−" + r + "%";
      };
      // a lab is fully severed by p if every pair touching it loses ~100% of its flow
      var seversLab = function (p) {
        for (var c = 0; c < 3; c++) {
          var ok = true, touches = false;
          for (var i = 0; i < d.pairs.length; i++) {
            if (d.pairs[i][0] !== c && d.pairs[i][1] !== c) continue;
            if (d.baseline.flows[i] <= 0) continue;
            touches = true;
            if (p.per_pair[i] < 0.999) ok = false;
          }
          if (touches && ok) return c;
        }
        return null;
      };

      // ---- layout: lollipop left, before/after minimap pair beside rank #1 ----
      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "16px 30px").style("align-items", "flex-start");
      var chartBox = wrap.append("div").style("flex", "1 1 340px").style("min-width", "300px");
      var mapsBox = wrap.append("div").style("flex", "0 0 auto");

      // ---- the minimap pair: as recorded vs. without rank #1 ----
      var MW = 170, MH = 136;
      var mapsRow = mapsBox.append("div").style("display", "flex").style("gap", "12px");
      var mapCell = function (caption) {
        var cell = mapsRow.append("div").style("width", MW + "px");
        var holder = cell.append("div").node();
        cell.append("div").attr("class", "m-note")
          .style("text-align", "center").style("margin-top", "2px").text(caption);
        return holder;
      };
      var leftMap = shared.minimap(mapCell("as recorded"), shared.colorOf,
        { width: MW, height: MH });
      shared.minimap(mapCell("without " + shared.labelOf(top1.id)), shared.colorOf, {
        width: MW, height: MH,
        opacityFn: function (id) { return id === top1.id ? 0.04 : 0.9; },
        onReady: function (api) {
          api.svg.selectAll("line").attr("stroke-opacity", function (l) {
            return (l.source === top1.id || l.target === top1.id) ? 0.04 : 0.8;
          });
        }
      });
      mapsBox.append("div").attr("class", "m-note")
        .style("max-width", (2 * MW + 12) + "px").style("margin-top", "6px")
        .html("Right: " + shared.seatLink(top1.id) + "&rsquo;s node and edges dimmed — " +
          pctLoss(top1.cost) + " of the network&rsquo;s between-lab capacity (tie strength " +
          shared.fmt.num(d.baseline.total) + "). Hover a name below to find them on the left map.");

      // ---- ranked lollipop ----
      var W = Math.max(chartBox.node().clientWidth || 0, 300);
      var wide = W >= 540;
      var nameW = 142, rightPad = wide ? 168 : 56;
      var rowH = 27, padTop = 8;
      var H = padTop + rowH * people.length + 8;
      var x = d3.scaleLinear().domain([0, maxCost]).range([nameW + 10, W - rightPad]);
      var y = function (i) { return padTop + rowH * i + rowH / 2; };

      if (!wide) {
        chartBox.append("div").attr("class", "m-note").style("margin-bottom", "4px")
          .text("share of between-lab flow lost when this person is removed");
      }
      var svg = chartBox.append("svg").attr("width", W).attr("height", H);

      // a single hairline at zero is the only axis this chart needs
      svg.append("line")
        .attr("x1", x(0)).attr("x2", x(0)).attr("y1", padTop - 2).attr("y2", H - 6)
        .attr("stroke", shared.colors.hair).attr("stroke-width", 1);

      var rows = svg.selectAll("g.row").data(people).join("g");
      rows.append("text").attr("class", "m-label")
        .attr("x", nameW).attr("y", function (p, i) { return y(i) + 4; })
        .attr("text-anchor", "end").attr("fill", shared.colors.ink)
        .text(function (p) { return shared.labelOf(p.id); });
      rows.append("line")
        .attr("x1", x(0)).attr("x2", function (p) { return x(p.cost); })
        .attr("y1", function (p, i) { return y(i); })
        .attr("y2", function (p, i) { return y(i); })
        .attr("stroke", function (p) { return shared.colorOf(p.id); })
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", function (p) { return 0.25 + 0.5 * (p.cost / maxCost); });
      var dots = rows.append("circle")
        .attr("cx", function (p) { return x(p.cost); })
        .attr("cy", function (p, i) { return y(i); }).attr("r", 5)
        .attr("fill", function (p) { return shared.colorOf(p.id); })
        .attr("fill-opacity", function (p) { return 0.45 + 0.55 * (p.cost / maxCost); });
      var vals = rows.append("text").attr("class", "m-axis")
        .attr("x", function (p) { return x(p.cost) + 9; })
        .attr("y", function (p, i) { return y(i) + 4; })
        .attr("fill", shared.colors.ink)
        .text(function (p) { return pctLoss(p.cost); });
      if (wide) {
        vals.filter(function (p, i) { return i === 0; })
          .append("tspan").attr("fill", shared.colors.muted).text(" of between-lab flow");
        vals.filter(function (p, i) { return i > 0 && seversLab(p) !== null; })
          .append("tspan").attr("fill", shared.colors.muted)
          .text(function (p) { return " · sole bridge to the " + labName(seversLab(p)) + " lab"; });
      }

      // ---- hover: tooltip + left-minimap highlight ----
      var tipFor = function (p) {
        var c = shared.communityOf(p.id);
        var html = "<strong>" + shared.esc(shared.labelOf(p.id)) + "</strong> " +
          "<span style=\"color:" + shared.colors.muted + "\">" +
          shared.esc(c >= 0 ? labName(c) : "unaffiliated") + " · degree " + p.degree +
          "</span><br>" + pctLoss(p.cost) + " of between-lab max-flow (tie strength " +
          shared.fmt.num(d.baseline.total) + " → " +
          shared.fmt.num(d.baseline.total * (1 - p.cost)) + ")";
        for (var i = 0; i < d.pairs.length; i++) {
          html += "<br><span style=\"color:" + shared.colors.muted + "\">" +
            shared.esc(pairName(i)) + "</span> " + pctLoss(p.per_pair[i]);
        }
        var sev = seversLab(p);
        if (sev !== null) {
          html += "<br><span style=\"color:" + shared.colors.muted +
            "\">without them the " + shared.esc(labName(sev)) +
            " lab detaches from the map entirely</span>";
        } else if (p.ac_drop > 0.005) {
          html += "<br><span style=\"color:" + shared.colors.muted +
            "\">second check — algebraic connectivity: " + pctLoss(p.ac_drop) + "</span>";
        }
        return html;
      };
      rows.append("rect")
        .attr("x", 0).attr("width", W)
        .attr("y", function (p, i) { return y(i) - rowH / 2; }).attr("height", rowH)
        .attr("fill", "transparent")
        .on("mouseenter", function (evt, p) {
          dots.interrupt();
          dots.attr("stroke", function (q) { return q.id === p.id ? shared.colors.ink : null; })
            .attr("stroke-width", 1.4)
            .attr("fill-opacity", function (q) {
              return q.id === p.id ? 1 : 0.2 + 0.3 * (q.cost / maxCost);
            });
          leftMap.highlight(p.id);
          shared.tooltip.show(tipFor(p), evt);
        })
        .on("mousemove", function (evt, p) { shared.tooltip.show(tipFor(p), evt); })
        .on("mouseleave", function () {
          dots.interrupt();
          dots.attr("stroke", null)
            .attr("fill-opacity", function (q) { return 0.45 + 0.55 * (q.cost / maxCost); });
          leftMap.highlight(null);
          shared.tooltip.hide();
        });
    }
  });
})();
