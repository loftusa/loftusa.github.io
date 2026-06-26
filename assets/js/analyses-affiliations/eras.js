/* assets/js/analyses-affiliations/eras.js — cohort waves per shared room (Lexis swimlanes) */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("eras", {
    prose: {
      intro:
        "<p>Before this was one group chat it was several rooms, years apart — a Hopkins " +
        "cohort, a lab, a Discord, a fellowship. This panel takes every room that three or " +
        "more chat members passed through and puts all of them on one calendar: for each shared " +
        "room, when did each member walk in, and how long did they stay?</p>",
      how:
        "<p>Demographers have drawn careers this way since the 1870s: put calendar time on " +
        "one axis and let each life trace its own line, and suddenly cohorts appear as " +
        "parallel waves — people who entered together move together. It’s the same move as " +
        "aligning training runs by wall-clock instead of by step: align each run to its own " +
        "step count and every curve looks alike; align them to the shared clock and you see " +
        "which runs launched together and which started years apart.</p>" +
        "<p>Here each lane is one room, each thin line one member’s stay — a dot where they " +
        "walked in, the line running until they left, or to today if they haven’t. A diamond " +
        "is a stint inside a single calendar year (a months-long program shouldn’t draw as a " +
        "year bar). Read down the page and the chat assembles in waves: Hopkins first, then " +
        "the lab inside it, then — years later — the interpretability rooms, with MATS " +
        "admitting a fresh cohort almost every year.</p>",
      method:
        "<p>A Lexis diagram restricted to the calendar axis — life-lines on calendar time " +
        "(Lexis, 1875; see Vandeschrick, “The Lexis diagram, a misnomer,” <em>Demographic " +
        "Research</em> 2001), read as cohort analysis in the sense of Ryder, “The cohort as a " +
        "concept in the study of social change,” <em>Am. Sociol. Rev.</em> 1965. Data: the " +
        "hand-reviewed affiliation sweep; the 15 organizations with ≥3 members (the 13 " +
        "two-member rooms are omitted). Resolution is the whole year — months are flattened — " +
        "and “still there” means the sourced bio shows an open-ended range, capped at 2026. " +
        "Eleven memberships in drawn rooms carry no public dates; they are counted at the " +
        "bottom of their lane rather than guessed onto the axis. A membership whose end is " +
        "public but whose start is not draws as a line fading in from the left. Roles in the " +
        "tooltips are verbatim from the sourced entries; a line ending means only that a " +
        "dated affiliation ended.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, orgs = d.orgs, C = shared.colors, esc = shared.esc;
  // org-type palette — keep in sync across the analyses-affiliations panels + the map page
      var TYPE = { lab: "#8c510a", program: "#2166ac", company: "#7b3294",
                   community: "#35978f", university: "#98917f" };
      var X0 = d.x[0], X1 = d.x[1];

      var totalW = el.clientWidth || 680;
      var miniW = 200, wide = totalW >= 660;
      var chartW = Math.max(340, Math.min(wide ? totalW - miniW - 28 : totalW, 720));

      // ---- who stands in any drawn lane (dated or undated) ----
      var inLane = {};
      orgs.forEach(function (g) {
        g.members.forEach(function (m) { inLane[m.id] = true; });
        g.undated.forEach(function (id) { inLane[id] = true; });
      });

      // ---- mini-map, anchored top-right (static block when narrow) ----
      var mapBox = document.createElement("div");
      mapBox.className = "m-map";
      if (!wide) {
        mapBox.style.position = "static";
        mapBox.style.margin = "0 0 12px";
      }
      var mapDiv = document.createElement("div");
      mapBox.appendChild(mapDiv);
      var mapCap = document.createElement("div");
      mapCap.className = "m-map-cap";
      mapCap.style.maxWidth = miniW + "px";
      mapCap.textContent = "everyone who passed through a drawn room, in their usual seats; " +
        "hover a line or a room’s name";
      mapBox.appendChild(mapCap);
      el.appendChild(mapBox);

      var baseOp = function (id) { return inLane[id] ? 0.85 : 0.3; };
      var mini = shared.minimap(mapDiv, function (id) {
        return inLane[id] ? shared.colorOf(id) : null;
      }, { width: miniW, opacityFn: baseOp });

      // highlight an arbitrary SET on the mini-map (the built-in highlight is single-id)
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

      // ---- vertical layout: top axis, then one lane per room ----
      var ROW = 7, HEAD = 14, AXIS = 20, padL = 6, padR = 16;
      var x = d3.scaleLinear().domain([X0, X1]).range([padL, chartW - padR]);
      var xMax = x(X1);
      var laneY = {}, yCur = AXIS;
      orgs.forEach(function (g) { laneY[g.id] = yCur; yCur += HEAD + g.n * ROW; });
      var svgH = yCur + 6;

      var chart = d3.select(el).append("div").style("max-width", chartW + "px");
      var svg = chart.append("svg").attr("width", chartW).attr("height", svgH);
      var defs = svg.append("defs");
      var fades = {};                       // one fade-in gradient per org type actually needed
      function fadeId(type) {
        if (!fades[type]) {
          var grad = defs.append("linearGradient").attr("id", "eras-fade-" + type);
          grad.append("stop").attr("offset", "0").attr("stop-color", TYPE[type]).attr("stop-opacity", 0);
          grad.append("stop").attr("offset", "1").attr("stop-color", TYPE[type]).attr("stop-opacity", 0.55);
          fades[type] = true;
        }
        return "url(#eras-fade-" + type + ")";
      }

      // year axis on top + hairline verticals the full height
      for (var yr = X0 + 1; yr <= X1; yr += 2) {
        svg.append("line")
          .attr("x1", x(yr)).attr("x2", x(yr)).attr("y1", AXIS - 4).attr("y2", svgH - 2)
          .attr("stroke", "#efe9dc").attr("stroke-width", 1);
        svg.append("text").attr("class", "m-axis")
          .attr("x", x(yr)).attr("y", 11).attr("text-anchor", "middle").attr("fill", C.muted)
          .text(yr);
      }

      function tipHtml(g, m) {
        var h = "<span class='t-name'>" + esc(shared.labelOf(m.id)) + "</span>";
        if (m.role) h += "<div class='t-sub'>" + esc(m.role) + "</div>";
        h += "<div class='t-sub'>" + esc(g.label) + " · " + esc(m.years) + "</div>";
        return h;
      }

      orgs.forEach(function (g) {
        var y0 = laneY[g.id], col = TYPE[g.type] || C.other;

        // -- room header: type dot, name, muted "type · n people · k waves" --
        svg.append("circle").attr("cx", padL + 3).attr("cy", y0 + 7).attr("r", 3).attr("fill", col);
        var head = svg.append("text").attr("class", "m-label")
          .attr("x", padL + 11).attr("y", y0 + 11).text(g.label);
        head.append("tspan").attr("class", "m-axis").attr("fill", C.muted).attr("dx", 7)
          .text(g.type + " · " + g.n + " people · " + g.waves + (g.waves === 1 ? " wave" : " waves"));
        svg.append("rect")                  // header hit zone → light the whole room
          .attr("x", 0).attr("y", y0).attr("width", chartW).attr("height", HEAD)
          .attr("fill", "transparent")
          .on("mouseenter", function () { highlightSet(g.members.map(function (m) { return m.id; }).concat(g.undated)); })
          .on("mouseleave", function () { highlightSet(null); });

        // -- one thin line per member, walk-in dot at start --
        g.members.forEach(function (m, i) {
          var cy = y0 + HEAD + i * ROW + 3.5;
          var marks = [], hit0, hit1;
          if (m.start == null) {            // end-only: known exit, unrecorded entry → fade in
            var xe = x(m.end), xs = Math.max(padL, xe - 40);
            marks.push(svg.append("rect")
              .attr("x", xs).attr("y", cy - 1).attr("width", xe - xs).attr("height", 2)
              .attr("fill", fadeId(g.type)));
            hit0 = xs; hit1 = xe;
          } else if (m.spell) {             // single calendar year → diamond, never a year bar
            var cx = x(m.start), r = 3.4, op = m.ongoing ? 1 : 0.55;
            marks.push(svg.append("path")
              .attr("d", "M" + cx + " " + (cy - r) + "L" + (cx + r) + " " + cy +
                         "L" + cx + " " + (cy + r) + "L" + (cx - r) + " " + cy + "Z")
              .attr("fill", col).attr("fill-opacity", op));
            hit0 = cx - 8; hit1 = cx + 8;
          } else {
            var xa = x(m.start), xb = m.ongoing ? xMax : x(m.end);
            var op2 = m.ongoing ? 1 : 0.55;
            marks.push(svg.append("line")
              .attr("x1", xa).attr("x2", xb).attr("y1", cy).attr("y2", cy)
              .attr("stroke", col).attr("stroke-width", 2).attr("stroke-opacity", op2));
            marks.push(svg.append("circle")
              .attr("cx", xa).attr("cy", cy).attr("r", 2.4)
              .attr("fill", col).attr("fill-opacity", op2));
            hit0 = xa - 5; hit1 = xb + 5;
          }
          svg.append("rect")                // generous invisible hover target
            .attr("x", hit0).attr("y", cy - ROW / 2).attr("width", hit1 - hit0).attr("height", ROW)
            .attr("fill", "transparent")
            .on("mouseenter", function (evt) {
              marks.forEach(function (s) {
                s.attr("stroke-width", s.node().tagName === "line" ? 3 : null);
              });
              mini.highlight(m.id);
              shared.tooltip.show(tipHtml(g, m), evt);
            })
            .on("mousemove", function (evt) { shared.tooltip.show(tipHtml(g, m), evt); })
            .on("mouseleave", function () {
              marks.forEach(function (s) {
                s.attr("stroke-width", s.node().tagName === "line" ? 2 : null);
              });
              mini.highlight(null);
              shared.tooltip.hide();
            });
        });

        // -- undated members: counted at the lane bottom, never guessed onto the axis --
        if (g.undated.length) {
          svg.append("text").attr("class", "m-axis").attr("fill", C.muted)
            .attr("x", padL + 11)
            .attr("y", y0 + HEAD + g.members.length * ROW + (g.undated.length * ROW) / 2 + 4)
            .text(g.undated.length + " more, undated");
        }
      });

      // ---- two annotations, positioned from the data (skip when cramped) ----
      if (chartW >= 480) {
        var nd = orgs.filter(function (g) { return g.id === "neurodata-lab-johns-hopkins"; })[0];
        if (nd) {
          svg.append("text").attr("class", "m-anno").attr("text-anchor", "end")
            .attr("x", xMax).attr("y", laneY[nd.id] + HEAD + 2 * ROW + 5)
            .text("the long decade");
        }
        var mats = orgs.filter(function (g) { return g.id === "mats"; })[0];
        if (mats) {
          svg.append("text").attr("class", "m-anno").attr("text-anchor", "end")
            .attr("x", x(mats.first) - 10)
            .attr("y", laneY[mats.id] + HEAD + (mats.members.length * ROW) / 2 + 4)
            .text("a new wave most years");
        }
      }

      chart.append("div").attr("class", "m-note").style("border", "none").style("padding", "0")
        .style("margin-top", "10px")
        .html("Each lane is a room, each line a member: the dot marks the year they walked " +
          "in; full ink runs to today, faded ink ended. A diamond is a stint inside one " +
          "calendar year; a line fading in from the left has a public end but no public start.");
    }
  });
})();
