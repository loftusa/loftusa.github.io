/* assets/js/analyses-affiliations/range.js — perplexity of each career's institution-type mix */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("range", {
    prose: {
      intro:
        "<p>The affiliation records contain five kinds of room: labs, universities, companies, " +
        "programs, and communities. Some careers have slept in all five; others have made one " +
        "kind of room thoroughly home. This panel measures both — how wide a career ranges and " +
        "how deep it anchors — without ranking anyone down for either. (The author concedes " +
        "this measure was not chosen to flatter him. It does anyway.)</p>",
      how:
        "<p>Score a career exactly the way you'd score a language model on held-out text. The " +
        "five institution types are the vocabulary; a person's affiliations are the tokens; " +
        "take the Shannon entropy of their type distribution and exponentiate it. The result " +
        "is a perplexity between 1 and 5: a 1 means a single type explains the whole record, " +
        "a 5 means all five types are equally yours.</p>" +
        "<p>Ecologists rediscovered the same number as the <em>Hill number</em> — the effective " +
        "number of species in a habitat. Here it is the effective number of worlds a person " +
        "inhabits. Depth scores too: a perplexity-1 career sustained across many organizations " +
        "is the deep-anchor quadrant, honored on the x-axis rather than penalized on the y.</p>",
      method:
        "<p>Shannon entropy of each person's affiliation-type distribution, converted to a Hill " +
        "number of order q&nbsp;=&nbsp;1, i.e. exp(H) — the perplexity of the distribution " +
        "(Hill, &ldquo;Diversity and Evenness: A Unifying Notation,&rdquo; Ecology 1973; Jost, " +
        "&ldquo;Entropy and diversity,&rdquo; Oikos 2006). Counts are distinct canonical " +
        "organizations per type from the hand-reviewed sweep, not time-weighted: a decade-long " +
        "PhD and a summer program each count once. Sweep coverage is uneven — a thin public CV " +
        "reads as a low organization count, so people with fewer than three recorded " +
        "affiliations are muted in the chart and excluded from every claim; that is sparsity " +
        "of the record, not a verdict on the person. Type labels are the build's canon: a " +
        "university-hosted lab counts as a lab. People landing on the exact same " +
        "(organizations, perplexity) point are fanned out slightly so no one hides behind " +
        "anyone else.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, C = shared.colors;
  // org-type palette — keep in sync across the analyses-affiliations panels + the map page
      var TYPE_COLORS = {
        lab: "#8c510a", program: "#2166ac", company: "#7b3294",
        community: "#35978f", university: "#98917f"
      };
      var PLURAL = {
        lab: "labs", program: "programs", company: "companies",
        community: "communities", university: "universities"
      };

      var byId = {};
      d.people.forEach(function (p) { byId[p.id] = p; });
      var annotated = {};
      d.annotated.forEach(function (id) { annotated[id] = true; });

      var totalW = el.clientWidth || 680;
      var miniW = 200, gap = 24;
      var wide = totalW >= 660;
      var W = Math.max(360, Math.min(wide ? totalW - miniW - gap : totalW, 640));
      var H = 380;
      var padL = 30, padR = 14, padT = 22, padB = 34;

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", gap + "px").style("align-items", "flex-start");
      var left = wrap.append("div").style("flex", "0 0 " + W + "px").style("max-width", "100%");
      var right = wrap.append("div").style("flex", "0 0 " + miniW + "px");

      var svg = left.append("svg").attr("width", W).attr("height", H);
      var x = d3.scaleLinear().domain([0.5, 13.8]).range([padL, W - padR]);
      var y = d3.scaleLinear().domain([0.7, 5.35]).range([H - padB, padT]);

      // ---- hairline axes -----------------------------------------------------------
      svg.append("line")
        .attr("x1", padL).attr("x2", W - padR).attr("y1", H - padB).attr("y2", H - padB)
        .attr("stroke", C.hair).attr("stroke-width", 1);
      var xTicks = d3.range(1, 14);
      svg.selectAll("text.xt").data(xTicks).join("text").attr("class", "m-axis xt")
        .attr("x", function (v) { return x(v); }).attr("y", H - padB + 14)
        .attr("text-anchor", "middle").attr("fill", C.muted).text(String);
      svg.append("text").attr("class", "m-axis")
        .attr("x", (padL + W - padR) / 2).attr("y", H - 4)
        .attr("text-anchor", "middle").attr("fill", C.muted)
        .text("distinct organizations on record");
      var yTicks = [1, 2, 3, 4, 5];
      svg.selectAll("line.yt").data(yTicks).join("line").attr("class", "yt")
        .attr("x1", padL - 4).attr("x2", padL).attr("y1", y).attr("y2", y)
        .attr("stroke", C.hair).attr("stroke-width", 1);
      svg.selectAll("text.yt").data(yTicks).join("text").attr("class", "m-axis yt")
        .attr("x", padL - 7).attr("y", function (v) { return y(v) + 3.5; })
        .attr("text-anchor", "end").attr("fill", C.muted).text(String);

      // ---- the ceiling: perplexity 5 -------------------------------------------------
      svg.append("line")
        .attr("x1", padL).attr("x2", W - padR).attr("y1", y(5)).attr("y2", y(5))
        .attr("stroke", C.hair).attr("stroke-width", 1);
      svg.append("text").attr("class", "m-axis")
        .attr("x", padL).attr("y", y(5) - 5).attr("fill", C.muted)
        .text("perplexity 5 — the ceiling");

      // ---- quadrant names, both positive ---------------------------------------------
      svg.append("text").attr("class", "m-anno")
        .attr("x", x(13.6)).attr("y", y(4.0)).attr("text-anchor", "end")
        .text("wide travelers");
      svg.append("text").attr("class", "m-anno")
        .attr("x", x(13.6)).attr("y", y(1.0) + 4).attr("text-anchor", "end")
        .text("deep anchors");

      // ---- dots -----------------------------------------------------------------------
      function cxOf(p) { return x(p.n) + p.dx; }
      function cyOf(p) { return y(p.n1) + p.dy; }
      function baseOpacity(p) { return p.low ? 0.3 : 0.85; }

      var dots = svg.append("g").selectAll("circle").data(d.people).join("circle")
        .attr("cx", cxOf).attr("cy", cyOf)
        .attr("r", function (p) { return annotated[p.id] ? 4.2 : 3.4; })
        .attr("fill", function (p) { return shared.colorOf(p.id); })
        .attr("fill-opacity", baseOpacity);

      // ---- direct labels on the annotated four only -----------------------------------
      var name = shared.labelOf;
      var ANNO = {
        "alex loftus": { lines: [name("alex loftus") + " — nearly maxed out"], anchor: "end", ox: -8, oy: 16 },
        "roy rinberg": { lines: [name("roy rinberg")], anchor: "start", ox: 8, oy: 4 },
        "jeremy howard": { lines: [name("jeremy howard")], anchor: "start", ox: 8, oy: 4 },
        "zeno kujawa": {
          lines: [name("zeno kujawa"), "four chapters, every one a university", "— all-in on the academy"],
          anchor: "start", ox: 8, oy: -34
        }
      };
      d.annotated.forEach(function (id) {
        var p = byId[id], a = ANNO[id];
        if (!p || !a) return;
        a.lines.forEach(function (line, i) {
          svg.append("text").attr("class", i === 0 ? "m-label" : "m-anno")
            .attr("x", cxOf(p) + a.ox).attr("y", cyOf(p) + a.oy + i * 13)
            .attr("text-anchor", a.anchor)
            .text(line);
        });
      });

      // ---- hover: tooltip + the shared mini-map ---------------------------------------
      function tally(types) {
        var entries = Object.keys(types).map(function (t) { return [t, types[t]]; });
        entries.sort(function (a, b) { return b[1] - a[1] || (a[0] < b[0] ? -1 : 1); });
        return entries.map(function (e) {
          var word = e[1] === 1 ? e[0] : PLURAL[e[0]];
          return "<span style='color:" + TYPE_COLORS[e[0]] + "'>" + e[1] + " " + word + "</span>";
        }).join(" · ");
      }
      function tipHtml(p) {
        var k = Object.keys(p.types).length;
        var lowNote = p.low
          ? "<br><span style='color:" + C.muted + "'>fewer than 3 affiliations on record — " +
            "too thin to score</span>"
          : "";
        return "<strong>" + shared.esc(name(p.id)) + "</strong><br>" +
          "<span style='color:" + C.muted + "'>perplexity " + p.n1.toFixed(1) +
          " over " + k + (k === 1 ? " type" : " types") + "</span><br>" +
          tally(p.types) + lowNote;
      }
      svg.append("g").selectAll("circle").data(d.people).join("circle")
        .attr("cx", cxOf).attr("cy", cyOf).attr("r", 9)
        .attr("fill", "transparent")
        .on("mouseenter", function (evt, p) {
          dots.interrupt();
          dots.attr("stroke", function (q) { return q.id === p.id ? C.ink : null; })
            .attr("stroke-width", 1.4)
            .attr("fill-opacity", function (q) { return q.id === p.id ? 1 : baseOpacity(q) * 0.45; });
          mini.highlight(p.id);
          shared.tooltip.show(tipHtml(p), evt);
        })
        .on("mousemove", function (evt, p) { shared.tooltip.show(tipHtml(p), evt); })
        .on("mouseleave", function () {
          dots.interrupt();
          dots.attr("stroke", null).attr("fill-opacity", baseOpacity);
          mini.highlight(null);
          shared.tooltip.hide();
        });

      left.append("div").attr("class", "m-note").style("margin-top", "6px")
        .html("Muted dots: fewer than three recorded affiliations — too thin to score, so they " +
          "sit out of every claim. That is sparsity of the public record, not a verdict on " +
          "anyone. Dots landing on the same exact point are fanned apart so no one hides " +
          "behind anyone else.");

      // ---- mini-map: ink scales with breadth ------------------------------------------
      var mini = shared.minimap(right.node(), function (id) {
        return shared.colorOf(id);
      }, {
        opacityFn: function (id) {
          var p = byId[id];
          return p ? Math.max(0.25, (p.n1 - 1) / 4) : 0.25;
        }
      });
      right.append("div").attr("class", "m-note").style("margin-top", "6px")
        .html("Everyone in their usual seat, ink scaled to breadth: the more kinds of " +
          "institution a career spans, the more solid the dot. Hover a dot on the chart to " +
          "find that person here.");
    }
  });
})();
