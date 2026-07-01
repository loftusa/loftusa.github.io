/* assets/js/analyses/which-lab-claims-you.js — one-hot GEE lab-affinity, drawn as a ternary plot */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("which-lab-claims-you", {
    prose: {
      intro:
        "<p>Three labs anchor this map, but most people on it sit on no lab&rsquo;s roster &mdash; " +
        "outside collaborators, one-paper co-authors, even the PIs themselves (rosters list members, " +
        "so David Bau is on nobody&rsquo;s list). This panel lets the network settle it: judging purely " +
        "by who you write papers with, which lab claims you? Most verdicts are unanimous; 23 people " +
        "are split between labs &mdash; and the single most contested name on the map is now " +
        "<strong>David Bau</strong> himself.</p>",
      how:
        "<p>Each of the 35 roster members gets a one-hot label for their lab; everyone else starts " +
        "blank. Then we run exactly one round of label propagation through the weighted co-authorship " +
        "graph: every person sums their co-authors&rsquo; labels, weighted by tie strength and " +
        "normalized by roster size, so the 4-person EleutherAI roster isn&rsquo;t simply outvoted by " +
        "the 17-person Bau roster. In ML terms this is a 1-layer GNN with frozen one-hot weights " +
        "&mdash; a single message-passing step, nothing trained.</p>" +
        "<p>L1-normalize each person&rsquo;s row and you get an affinity distribution over the three " +
        "labs &mdash; read it like softmax probabilities over three classes. The triangle <em>is</em> " +
        "that distribution: corners are labs, a dot hugging a corner is claimed outright, a dot out on " +
        "an edge is split between two labs. The price of one layer is a one-hop receptive field: " +
        "17 people whose co-authors here carry no lab label sit outside it, and stay grey.</p>",
      method:
        "<p>One-Hot Graph Encoder Embedding &mdash; Shen, Wang &amp; Priebe, <em>IEEE TPAMI</em> 2023: " +
        "Z&nbsp;=&nbsp;AW, where W is the n&times;3 class-indicator matrix column-normalized by class " +
        "size; rows of Z for unseeded vertices, L1-normalized, are the simplex coordinates plotted " +
        "here, computed on the full weighted graph including bridge-path edges. Edge weights use " +
        "fractional co-authorship counting &mdash; each paper adds 1/n_authors per pair (suggested by " +
        "Stella Biderman) &mdash; so a two-person paper outweighs a fifty-author one. Best of all, " +
        "C.&nbsp;E.&nbsp;Priebe is a node on this very map: run through his own method, the " +
        "co-inventor comes back <strong>100% Vogelstein&nbsp;lab</strong>, on 62 paper-links to that " +
        "roster. Caveats: &ldquo;labs&rdquo; here means the page&rsquo;s three clusters (for " +
        "people the clustering also labeled, GEE agrees 56 of 63 times), and per-class normalization " +
        "means a unit of tie strength pulls about six times as hard toward the 3-person EleutherAI " +
        "roster as toward the 18-person Bau roster. An earlier cut of the clustering filed Sheridan " +
        "Feucht &mdash; a Bau-lab student &mdash; under EleutherAI, which briefly tipped David Bau " +
        "himself 53/46 toward EleutherAI; with her corrected to the Bau roster, the network returns " +
        "him 99% to the lab that bears his name.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, C = shared.colors, esc = shared.esc;
      var labs = d.labs, people = d.people;

      var root = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "18px").style("align-items", "flex-start");
      var left = root.append("div").style("flex", "1 1 380px").style("min-width", "320px");
      var right = root.append("div").style("flex", "0 0 215px");

      // ---- ternary geometry: Bau (1) top, EleutherAI (0) bottom-left, Vogelstein (2) bottom-right
      var W = Math.min(left.node().clientWidth || el.clientWidth || 560, 560);
      var M = { top: 44, bottom: 46, side: 26 };
      var side = W - 2 * M.side;
      var H = M.top + side * Math.sqrt(3) / 2 + M.bottom;
      var corner = [];
      corner[1] = [W / 2, M.top];
      corner[0] = [M.side, H - M.bottom];
      corner[2] = [W - M.side, H - M.bottom];
      var centroid = [
        (corner[0][0] + corner[1][0] + corner[2][0]) / 3,
        (corner[0][1] + corner[1][1] + corner[2][1]) / 3
      ];
      function bary(aff) {
        var x = 0, y = 0;
        for (var k = 0; k < 3; k++) { x += aff[k] * corner[k][0]; y += aff[k] * corner[k][1]; }
        return [x, y];
      }

      var svg = left.append("svg").attr("width", W).attr("height", H);

      // triangle outline + claim boundaries (centroid -> edge midpoints), hairline only
      svg.append("path")
        .attr("d", "M" + corner[0] + "L" + corner[1] + "L" + corner[2] + "Z")
        .attr("fill", "none").attr("stroke", C.hair).attr("stroke-width", 1);
      for (var k = 0; k < 3; k++) {
        var mid = [(corner[(k + 1) % 3][0] + corner[(k + 2) % 3][0]) / 2,
                   (corner[(k + 1) % 3][1] + corner[(k + 2) % 3][1]) / 2];
        svg.append("line")
          .attr("x1", centroid[0]).attr("y1", centroid[1])
          .attr("x2", mid[0]).attr("y2", mid[1])
          .attr("stroke", "#efe9dc").attr("stroke-width", 1);
      }

      // corner labels: lab name in lab color + roster size and claim count (no legend anywhere)
      [0, 1, 2].forEach(function (lab) {
        var p = corner[lab], top = lab === 1;
        var anchor = top ? "middle" : (lab === 0 ? "start" : "end");
        var g = svg.append("g");
        g.append("text").attr("class", "m-label")
          .attr("x", p[0]).attr("y", p[1] + (top ? -26 : 20)).attr("text-anchor", anchor)
          .attr("fill", C.community[lab]).style("font-weight", "600")
          .text(labs[lab]);
        g.append("text").attr("class", "m-axis")
          .attr("x", p[0]).attr("y", p[1] + (top ? -13 : 33)).attr("text-anchor", anchor)
          .attr("fill", C.muted)
          .text(d.roster_sizes[lab] + " on roster · claims " + d.counts[lab]);
      });

      // ---- deterministic fan-out for coincident points (corners hold dozens of 100% verdicts)
      var groups = d3.group(people, function (p) { return p.affinity.join(","); });
      var mass = function (p) { return p.raw[0] + p.raw[1] + p.raw[2]; };
      var pts = [];
      groups.forEach(function (members) {
        members = members.slice().sort(function (a, b) {
          return mass(b) - mass(a) || (a.id < b.id ? -1 : 1);
        });
        var base = bary(members[0].affinity);
        var inward = Math.atan2(centroid[1] - base[1], centroid[0] - base[0]);
        members.forEach(function (p, i) {
          var r = i === 0 ? 0 : 5.4 * Math.sqrt(i);
          var ang = inward + ((i * 2.39996) % (Math.PI * 0.9)) - Math.PI * 0.45;
          pts.push({ p: p, x: base[0] + r * Math.cos(ang), y: base[1] + r * Math.sin(ang) });
        });
      });

      var inkOp = d3.scaleSqrt()
        .domain([1, d3.max(people, mass)]).range([0.45, 0.95]);

      // ---- mini-map: predicted color at full ink for the off-roster; faint actual lab for rosters
      var winnerOf = {};
      people.forEach(function (p) { winnerOf[p.id] = p.winner; });
      var seeded = function (id) {
        return shared.isList(id) && shared.communityOf(id) >= 0 && winnerOf[id] == null;
      };
      var mini = shared.minimap(right.append("div").node(), function (id) {
        if (winnerOf[id] != null) return C.community[winnerOf[id]];
        if (seeded(id)) return C.community[shared.communityOf(id)];
        return null;
      }, {
        opacityFn: function (id) {
          if (winnerOf[id] != null) return 0.95;
          if (seeded(id)) return 0.3;
          return 0.4;
        }
      });

      function tipHtml(p) {
        var rows = [0, 1, 2].filter(function (lab) { return p.affinity[lab] > 0; })
          .sort(function (a, b) { return p.affinity[b] - p.affinity[a]; })
          .map(function (lab) {
            return '<div><span style="color:' + C.community[lab] + '">●</span> ' +
              esc(labs[lab]) + " · <strong>" + shared.fmt.pct(p.affinity[lab]) + "</strong>" +
              ' <span style="color:' + C.muted + '">· ' + p.raw[lab] +
              " paper-link" + (p.raw[lab] === 1 ? "" : "s") + " to roster</span></div>";
          }).join("");
        return "<strong>" + esc(shared.labelOf(p.id)) + "</strong>" + rows;
      }

      svg.append("g").selectAll("circle").data(pts).join("circle")
        .attr("cx", function (q) { return q.x; }).attr("cy", function (q) { return q.y; })
        .attr("r", 3.6)
        .attr("fill", function (q) { return C.community[q.p.winner]; })
        .attr("fill-opacity", function (q) { return inkOp(mass(q.p)); })
        .style("cursor", "default")
        .on("mouseenter", function (evt, q) {
          d3.select(this).interrupt().attr("r", 5).attr("stroke", C.ink).attr("stroke-width", 1.2);
          shared.tooltip.show(tipHtml(q.p), evt);
          mini.highlight(q.p.id);
        })
        .on("mousemove", function (evt, q) { shared.tooltip.show(tipHtml(q.p), evt); })
        .on("mouseleave", function () {
          d3.select(this).interrupt().attr("r", 3.6).attr("stroke", null);
          shared.tooltip.hide();
          mini.highlight(null);
        });

      // ---- direct labels: strongest claim per lab, the most torn, the PI, and Priebe himself
      var callouts = pts.filter(function (q) { return q.p.callout; })
        .map(function (q) {
          var inward = Math.atan2(centroid[1] - q.y, centroid[0] - q.x);
          var nearCorner = Math.hypot(q.x - centroid[0], q.y - centroid[1]) > side * 0.33;
          var off = nearCorner ? 26 : 13;   // corner fans need clearance; edge dots don't
          return { q: q, x: q.x + off * Math.cos(inward), y: q.y + off * Math.sin(inward) + 4 };
        })
        .sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < callouts.length; i++) {       // greedy vertical de-overlap
        if (Math.abs(callouts[i].x - callouts[i - 1].x) < 110 &&
            callouts[i].y - callouts[i - 1].y < 13) callouts[i].y = callouts[i - 1].y + 13;
      }
      svg.append("g").selectAll("text").data(callouts).join("text")
        .attr("class", "m-label")
        .attr("x", function (c) { return c.x; }).attr("y", function (c) { return c.y; })
        .attr("text-anchor", function (c) {
          return c.x < centroid[0] - 30 ? "start" : (c.x > centroid[0] + 30 ? "end" : "middle");
        })
        .attr("fill", C.ink).attr("fill-opacity", 0.85)
        .text(function (c) { return shared.labelOf(c.q.p.id); });

      // ---- right column: claim tally, the unanimous, the out-of-reach ----
      var maxCount = d3.max(d.counts);
      var tally = right.append("div").style("margin-top", "10px");
      [0, 1, 2].sort(function (a, b) { return d.counts[b] - d.counts[a]; }).forEach(function (lab) {
        var row = tally.append("div")
          .style("display", "flex").style("align-items", "center")
          .style("gap", "7px").style("margin", "3px 0");
        row.append("span").attr("class", "m-axis")
          .style("color", C.community[lab]).style("width", "92px")
          .style("display", "inline-block").style("text-align", "right")
          .text(labs[lab]);
        row.append("span")
          .style("display", "inline-block").style("height", "8px")
          .style("width", Math.round(78 * d.counts[lab] / maxCount) + "px")
          .style("background", C.community[lab]).style("opacity", 0.8);
        row.append("span").attr("class", "m-axis").style("color", C.ink).text(d.counts[lab]);
      });
      tally.append("div").attr("class", "m-note").style("margin-top", "6px")
        .text("people claimed, of " + d.n_placed + " placed · " + d.n_decisive +
          " verdicts are unanimous: every paper-link points at one lab");

      var unplacedNames = d.unplaced.map(function (id) { return esc(shared.labelOf(id)); });
      right.append("div").attr("class", "m-note").style("margin-top", "12px")
        .style("border-bottom", "1px dotted " + C.muted)
        .style("display", "inline-block").style("cursor", "default")
        .text(d.unplaced.length + " of " + d.n_query + " are out of reach in one hop — " +
          "none of their co-authors here carry a lab label")
        .on("mouseenter", function (evt) {
          shared.tooltip.show("<strong>Beyond the one-hop receptive field</strong><div>" +
            unplacedNames.join(" · ") + "</div>", evt);
        })
        .on("mouseleave", function () { shared.tooltip.hide(); });

      right.append("div").attr("class", "m-note").style("margin-top", "12px")
        .text("Mini-map: off-roster people at full ink in their predicted lab’s color; " +
          "roster members faint, in their actual lab’s color.");
    }
  });
})();
