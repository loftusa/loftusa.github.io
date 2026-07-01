/* assets/js/analyses/year-everything-changed.js */
(function () {
  "use strict";

  var graphCache = null; // promise for /assets/data/coauthorship.json (positions for the minis)

  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("year-everything-changed", {
    prose: {
      intro:
        "<p>Networks usually grow the way moss grows — a little everywhere, no drama. " +
        "This one didn’t. We redraw the cumulative co-authorship network for every year " +
        "since 2015 and ask: in which year did its <em>structure</em> move farthest from " +
        "its own past?</p>",
      how:
        "<p>Each year’s graph is embedded with OMNI, which stacks all twelve yearly " +
        "matrices into one big matrix and factorizes them <em>jointly</em> — the same " +
        "trick as aligning latent spaces across model checkpoints, so a person’s movement " +
        "between years is real signal, not an artifact of each year being embedded " +
        "separately. Every person traces a trajectory through that one shared space, one " +
        "position per year.</p>" +
        "<p>Average everyone’s year-over-year displacement and you get a single curve you " +
        "read exactly like a loss curve: flat stretches mean the network is just " +
        "densifying — new papers along old lines — and a kink means it is " +
        "<em>reorganizing</em>. Change-point detection is just finding the biggest kink. " +
        "(Edges here are binarized — present or absent — so a kink reflects pure " +
        "wiring, not which handful of pairs published most.)</p>",
      method:
        "<p>This is the Euclidean-mirror construction of Athreya, Lubberts, Park &amp; " +
        "Priebe, “Euclidean Mirrors and Dynamics in Network Time Series,” <em>Journal of " +
        "the American Statistical Association</em> 120(550), 2025 (" +
        "arXiv:2205.06877), in simplified form: omnibus embedding (Levin et al. 2017) at " +
        "d&nbsp;=&nbsp;3 on the binarized graphs, plotting mean per-vertex displacement " +
        "between consecutive years instead of the full CMDS mirror ψ(t), with the " +
        "change-point taken as the largest positive jump (the iso-mirror idea). And here " +
        "is the part we can’t resist: the last author of that paper is ON this map — " +
        "Carey E. Priebe sits two hops out on the Vogelstein side, so the method that " +
        "detects this network’s reorganization was invented inside the network. Caveats: " +
        "the curve drawn here binarizes edges (present or absent), reading pure wiring; " +
        "read instead by rank-transformed collaboration volume (each paper adds " +
        "1/n_authors per pair, suggested by Stella Biderman), the biggest jump moves to " +
        "2019, the Vogelstein-lab publication burst — so 2024 is the new-ties story and " +
        "2019 the volume story. Cumulative graphs also smooth the curve — every year contains " +
        "all previous years — so the jumps shown here are conservative.</p>"
    },

    render: function (el, data, shared) {
      var d = data.data;
      var W = Math.max(el.clientWidth || 640, 360);
      var C = shared.colors;

      // ---------- the mirror curve ----------
      var H = 240, mL = 12, mR = 16, mT = 42, mB = 26;
      var years = d.years, mirror = d.mirror;
      var cp = d.changepoint, cpI = years.indexOf(cp);
      var ru = d.runnerup, ruI = years.indexOf(ru);
      var x = d3.scaleLinear().domain([years[0], years[years.length - 1]]).range([mL + 18, W - mR - 8]);
      var y = d3.scaleLinear().domain([0, d3.max(mirror) * 1.18]).range([H - mB, mT]);

      var svg = d3.select(el).append("svg").attr("width", W).attr("height", H);

      // baseline + year ticks (hairline axis, direct labels)
      svg.append("line").attr("x1", mL).attr("x2", W - mR)
        .attr("y1", y(0)).attr("y2", y(0)).attr("stroke", C.hair);
      var step = W < 560 ? 2 : 1;
      years.forEach(function (yr, i) {
        if (i % step !== 0 && yr !== cp) return;
        svg.append("text").attr("class", "m-axis")
          .attr("x", x(yr)).attr("y", y(0) + 15).attr("text-anchor", "middle")
          .attr("fill", yr === cp ? C.ink : C.muted)
          .text(yr === cp ? String(yr) : "’" + String(yr).slice(2));
      });

      // what the y-direction means — one direct label, no axis
      svg.append("text").attr("class", "m-note")
        .attr("x", mL).attr("y", 14).attr("fill", C.muted)
        .text("mean per-person movement in the joint embedding, year over year");

      // vertical hairline at the change-point
      svg.append("line")
        .attr("x1", x(cp)).attr("x2", x(cp))
        .attr("y1", y(0)).attr("y2", y(mirror[cpI]))
        .attr("stroke", C.ink).attr("stroke-width", 0.7).attr("stroke-dasharray", "2,3");

      // the curve
      var line = d3.line()
        .x(function (_, i) { return x(years[i]); })
        .y(function (m) { return y(m); });
      svg.append("path").attr("d", line(mirror))
        .attr("fill", "none").attr("stroke", C.ink).attr("stroke-width", 1.5);

      // points (+ generous invisible hover targets)
      var movers = d.top_movers.map(function (m) { return shared.labelOf(m[0]); });
      var moverIds = d.top_movers.map(function (m) { return m[0]; });
      var pts = svg.append("g");
      years.forEach(function (yr, i) {
        pts.append("circle")
          .attr("cx", x(yr)).attr("cy", y(mirror[i]))
          .attr("r", yr === cp ? 4 : 2.4)
          .attr("fill", yr === cp ? C.ink : C.bg)
          .attr("stroke", C.ink).attr("stroke-width", 1.2);
        pts.append("circle")
          .attr("cx", x(yr)).attr("cy", y(mirror[i])).attr("r", 12)
          .attr("fill", "transparent").style("cursor", "default")
          .on("mousemove", function (evt) {
            var ratio = i > 0 && mirror[i - 1] > 0 ? mirror[i] / mirror[i - 1] : null;
            var html = "<strong>" + shared.esc(yr) + "</strong><br>" +
              "movement " + shared.esc(shared.fmt.sig(mirror[i], 2)) +
              (ratio ? " (" + shared.esc(shared.fmt.sig(ratio, 2)) + "&times; the year before)" : "") +
              "<br>" + shared.esc(d.edge_counts[i]) + " ties to date";
            if (yr === cp) {
              html += "<br>fastest movers: " + movers.map(shared.esc).join(", ");
            }
            shared.tooltip.show(html, evt);
          })
          .on("mouseleave", function () { shared.tooltip.hide(); });
      });

      // annotation written AT the elbow
      var cpX = x(cp), rightish = cpX > W * 0.62;
      var anchor = rightish ? "end" : "start", ax = cpX + (rightish ? -9 : 9);
      svg.append("text").attr("class", "m-label")
        .attr("x", ax).attr("y", y(mirror[cpI]) - 14)
        .attr("text-anchor", anchor).attr("fill", C.ink)
        .text(cp + " — the super-core forms");
      svg.append("text").attr("class", "m-note")
        .attr("x", ax).attr("y", y(mirror[cpI]) - 1)
        .attr("text-anchor", anchor).attr("fill", C.muted)
        .text(d.anno_sub);

      // the runner-up jump, labeled at its own rise
      var ruY = Math.min(y(mirror[ruI]), y(mirror[Math.min(ruI + 1, mirror.length - 1)]));
      svg.append("text").attr("class", "m-note")
        .attr("x", x(ru)).attr("y", ruY - 8)
        .attr("text-anchor", "middle").attr("fill", C.muted)
        .text(ru + " — " + d.runnerup_sub);

      // who moved farthest in the change-point year (real people, direct-labeled)
      var moversNote = document.createElement("div");
      moversNote.className = "m-note";
      moversNote.style.margin = "2px 0 0";
      moversNote.innerHTML = "Fastest movers in the " + shared.esc(cp) + " reorganization: " +
        moverIds.map(function (id) { return "<strong>" + shared.seatLink(id) + "</strong>"; }).join(" · ") + ".";
      el.appendChild(moversNote);

      // ---------- small multiples: the map, redrawn with only each year's ties ----------
      var cap = document.createElement("div");
      cap.className = "m-note";
      cap.style.margin = "14px 0 6px";
      cap.textContent = "The same map, redrawn with only the ties that existed by each year. The outlined frame is the change-point.";
      el.appendChild(cap);

      var row = document.createElement("div");
      row.style.display = "flex";
      row.style.flexWrap = "wrap";
      row.style.gap = "14px";
      row.style.alignItems = "flex-end";
      el.appendChild(row);

      var MW = 110, MH = 96;
      if (!graphCache) graphCache = d3.json("/assets/data/coauthorship.json");
      graphCache.then(function (g) {
        if (!row.isConnected) return; // panel re-rendered while we were fetching
        var xe = d3.extent(g.nodes, function (n) { return n.x; });
        var ye = d3.extent(g.nodes, function (n) { return n.y; });
        var sx = d3.scaleLinear().domain(xe).range([4, MW - 4]);
        var sy = d3.scaleLinear().domain(ye).range([4, MH - 4]);
        var pos = {};
        g.nodes.forEach(function (n) { pos[n.id] = [sx(n.x), sy(n.y)]; });
        var idOf = d.vertex_order;

        d.minis.forEach(function (mini) {
          var isCp = mini.year === cp;
          var cell = document.createElement("div");
          row.appendChild(cell);
          var ms = d3.select(cell).append("svg").attr("width", MW).attr("height", MH)
            .style("border", "1px solid " + (isCp ? C.ink : "transparent"))
            .style("background", C.bg);
          var eg = ms.append("g");
          mini.edges.forEach(function (e) {
            var p = pos[idOf[e[0]]], q = pos[idOf[e[1]]];
            if (!p || !q) return;
            eg.append("line")
              .attr("x1", p[0]).attr("y1", p[1]).attr("x2", q[0]).attr("y2", q[1])
              .attr("stroke", C.muted).attr("stroke-width", 0.5).attr("stroke-opacity", 0.3);
          });
          var ng = ms.append("g");
          mini.active.forEach(function (vi) {
            var id = idOf[vi], p = pos[id];
            if (!p) return;
            ng.append("circle")
              .attr("cx", p[0]).attr("cy", p[1]).attr("r", 1.8)
              .attr("fill", shared.colorOf(id)).attr("fill-opacity", 0.8);
          });
          var lbl = document.createElement("div");
          lbl.className = "m-axis";
          lbl.style.textAlign = "center";
          lbl.style.marginTop = "3px";
          lbl.style.color = isCp ? C.ink : C.muted;
          lbl.textContent = mini.year + " · " + mini.edges.length + " ties";
          cell.appendChild(lbl);
        });
      });
    }
  });
})();
