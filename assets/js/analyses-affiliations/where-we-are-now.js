/* assets/js/analyses-affiliations/where-we-are-now.js — cross-sectional census: who's where today */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("where-we-are-now", {
    prose: {
      intro:
        "<p>The last two panels watched careers move — cohort waves arriving, university " +
        "years leading somewhere next. This one stops the tape: one frame, 2026, who's inside " +
        "a company, who's on a campus or in a lab, who's doing both. It also doubles as the " +
        "practical panel — hover a city and find your city-mates.</p>",
      how:
        "<p>The pipeline panel was the training curve: it measured <em>transitions</em> — where " +
        "a university year tends to lead next. This panel is a single checkpoint eval: load the " +
        "2026 weights, run the benchmark once, report the state. The two views answer different " +
        "questions and can honestly disagree — a model whose eval score is flat may still be " +
        "moving fast underneath, and a network that looks half-industry today says nothing " +
        "about which way the flow is running.</p>" +
        "<p>Demographers keep the two straight with two words: the snapshot is " +
        "<em>prevalence</em> (how many people are in a company right now), the moves are " +
        "<em>incidence</em> (how many crossed over this year) — and reading one as the other is " +
        "the classic way to fool yourself. Here each person is drawn exactly once, filed under " +
        "one column each — companies first, then labs, then campuses (the straddlers file under their company); every overlapping hat stays in the tooltip.</p>",
      method:
        "<p>A cross-sectional census, as against the longitudinal panel design of the two " +
        "preceding analyses; the stock/flow distinction is <em>prevalence</em> vs " +
        "<em>incidence</em> in the sense of Rothman, Greenland &amp; Lash, <em>Modern " +
        "Epidemiology</em>, 3rd ed., 2008. “Current” means current:&nbsp;true in the " +
        "hand-reviewed affiliation sweep, every entry carrying a public source link — so this " +
        "is a snapshot of the sweep date (2026), already aging, and it trusts public bios, " +
        "which lag reality. Counts are affiliations, not employment: a community membership is " +
        "a real row but not a job. Column assignment uses the precedence company &gt; lab &gt; " +
        "university &gt; program &gt; community, so the header tallies overlap — a person " +
        "filed under companies may also sit in the universities count. Cities are as recorded " +
        "in the sweep, kept city-level only; 11 of 48 are unrecorded.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, C = shared.colors;
  // org-type palette — keep in sync across the analyses-affiliations panels + the map page
      var TYPE = { company: "#7b3294", lab: "#8c510a", university: "#98917f",
                   program: "#2166ac", community: "#35978f" };
      var PLURAL = { company: "companies", lab: "labs", university: "universities",
                     program: "programs", community: "communities" };
      var PREC = ["company", "lab", "university", "program", "community"];
      var byId = {};
      d.people.forEach(function (p) { byId[p.id] = p; });

      var totalW = el.clientWidth || 680;
      var miniW = 200, gap = 24;
      var wide = totalW >= 660;
      var W = Math.max(360, Math.min(wide ? totalW - miniW - gap : totalW, 720));

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", gap + "px").style("align-items", "flex-start");
      var left = wrap.append("div").style("flex", "0 0 " + W + "px").style("max-width", "100%");
      var right = wrap.append("div").style("flex", "0 0 " + miniW + "px");

      // ---- mini-map: everyone in their usual seat, tinted by today's primary type ----
      var mini = shared.minimap(right.node(), function (id) {
        var p = byId[id];
        return p && p.primary ? TYPE[p.primary] : null;
      }, {});
      right.append("div").attr("class", "m-map-cap").style("text-align", "left")
        .text("everyone in their usual co-authorship seat, tinted by today's primary " +
          "affiliation — hover a dot or a city label to find people here");

      function highlightSet(ids) {
        var circles = mini.svg.selectAll("circle");
        if (!circles.size()) return;
        circles.interrupt();
        if (!ids) {
          circles.attr("stroke", null).attr("fill-opacity", 0.9);
          return;
        }
        var set = {};
        ids.forEach(function (id) { set[id] = true; });
        circles
          .attr("stroke", function (n) { return set[n.id] ? C.ink : null; })
          .attr("stroke-width", 1.4)
          .attr("fill-opacity", function (n) { return set[n.id] ? 1 : 0.18; });
      }

      function tipHtml(p) {
        var rows = p.all_current.map(function (e) {
          return "<div style='margin-top:2px'><span style='display:inline-block;width:7px;" +
            "height:7px;border-radius:50%;background:" + TYPE[e.type] + ";margin-right:5px'>" +
            "</span>" + shared.esc(e.org) +
            (e.years ? " <span style='color:" + C.muted + "'>" + shared.esc(e.years) + "</span>" : "") +
            "</div>";
        }).join("");
        var place = p.city
          ? (p.metro && p.metro !== p.city ? p.city + " · " + p.metro : p.city)
          : "city unrecorded";
        return "<strong>" + shared.esc(shared.labelOf(p.id)) + "</strong>" + rows +
          "<div style='margin-top:3px;color:" + C.muted + "'>" + shared.esc(place) + "</div>";
      }

      function wireDot(sel, p) {
        sel.on("mouseenter", function (evt) {
          mini.highlight(p.id);
          shared.tooltip.show(tipHtml(p), evt);
        }).on("mousemove", function (evt) {
          shared.tooltip.show(tipHtml(p), evt);
        }).on("mouseleave", function () {
          mini.highlight(null);
          shared.tooltip.hide();
        });
      }

      // 11px monogram dot; straddlers wear a second concentric ring in university grey
      function dot(parent, p, fill, ring) {
        var s = parent.append("span")
          .style("width", "11px").style("height", "11px").style("border-radius", "50%")
          .style("background", fill)
          .style("display", "inline-flex").style("align-items", "center")
          .style("justify-content", "center")
          .style("color", "#fff").style("font-size", "5px").style("font-weight", "700")
          .style("line-height", "1").style("cursor", "default").style("flex", "0 0 auto")
          .text(p.initials);
        if (ring) s.style("box-shadow", "0 0 0 1.5px " + C.bg + ", 0 0 0 3px " + TYPE.university);
        wireDot(s, p);
      }

      // ---- block (a): the census — each person once, under the heaviest hat ----
      left.append("div").attr("class", "m-axis").style("color", C.muted)
        .style("margin-bottom", "8px")
        .text("the census — each person drawn once; straddlers file under their company");

      var cols = left.append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", "14px").style("align-items", "flex-start");
      PREC.forEach(function (t) {
        var assigned = d.people.filter(function (p) { return p.primary === t; })
          .sort(function (a, b) {
            return (b.straddler - a.straddler) ||
              d3.ascending(shared.labelOf(a.id), shared.labelOf(b.id));
          });
        var col = cols.append("div").style("flex", "1 1 0").style("min-width", "82px");
        col.append("div").attr("class", "m-label")
          .style("color", TYPE[t]).style("margin-bottom", "6px")
          .html(PLURAL[t] + " — <strong>" + d.type_counts_people[t] + "</strong> people");
        var box = col.append("div")
          .style("display", "flex").style("flex-wrap", "wrap").style("gap", "5px");
        assigned.forEach(function (p) { dot(box, p, TYPE[t], p.straddler); });
        if (!assigned.length) {
          col.append("div").attr("class", "m-axis").style("color", C.muted)
            .text("all filed under a heavier hat");
        }
      });

      var strads = d.people.filter(function (p) { return p.straddler; });
      left.append("div").attr("class", "m-anno").style("color", "#6b665d")
        .style("margin-top", "10px")
        .html("four people are running both stacks at once — " +
          strads.map(function (p) { return shared.seatLink(p.id); }).join(", ") +
          " — the grey ring marks them.");

      var noCur = d.no_current_ids.map(function (id) { return shared.seatLink(id); });
      left.append("div").attr("class", "m-note").style("margin-top", "12px")
        .html("Headers count everyone holding that kind of affiliation, so they overlap: the " +
          "four straddlers sit under their company, and all <strong>" +
          d.type_counts_people.program + "</strong> people currently in a program are filed " +
          "higher up. The sweep couldn't verify a current chapter for " + d.no_current +
          " people — " + noCur.join(", ") + " — corrections welcome.");

      // ---- block (b): the cities — old tribes mixing in new cities ----
      left.append("div").attr("class", "m-axis").style("color", C.muted)
        .style("margin", "22px 0 4px")
        .text("the cities — dots wear their co-authorship tribe color; " +
          "hover a city to light up your city-mates");

      var metros = [];
      d.cities.forEach(function (c) {
        var m = metros[metros.length - 1];
        if (!m || m.name !== c.metro) {
          m = { name: c.metro, subs: [], members: [] };
          metros.push(m);
        }
        m.subs.push(c);
        c.members.forEach(function (id) { m.members.push(id); });
      });
      var bigs = metros.filter(function (m) { return m.members.length > 1; });
      var singles = metros.filter(function (m) { return m.members.length === 1; });
      var shortName = function (s) { return s.replace(/,.*$/, ""); };

      function cityRow(first) {
        return left.append("div")
          .style("display", "flex").style("gap", "10px").style("align-items", "flex-start")
          .style("padding", "6px 0")
          .style("border-top", "1px solid " + (first ? C.hair : "#efe9dc"));
      }
      function wireLabel(sel, members) {
        sel.style("cursor", "default")
          .on("mouseenter", function () { highlightSet(members); })
          .on("mouseleave", function () { highlightSet(null); });
      }

      bigs.forEach(function (m, i) {
        var row = cityRow(i === 0);
        var lbl = row.append("div").attr("class", "m-label").style("flex", "0 0 132px")
          .html(shared.esc(m.name) + " — <strong>" + m.members.length + "</strong>");
        wireLabel(lbl, m.members);
        var box = row.append("div").style("display", "flex").style("flex-wrap", "wrap")
          .style("gap", "5px 14px").style("align-items", "center").style("padding-top", "1px");
        m.subs.forEach(function (c) {
          var g = box.append("span").style("display", "inline-flex")
            .style("align-items", "center").style("gap", "5px");
          if (m.subs.length > 1) {
            var sub = g.append("span").attr("class", "m-axis").style("color", C.muted)
              .text(shortName(c.name));
            wireLabel(sub, c.members);
          }
          c.members.forEach(function (id) { dot(g, byId[id], shared.colorOf(id), false); });
        });
      });

      if (singles.length) {
        var row = cityRow(false);
        var lbl = row.append("div").attr("class", "m-label").style("flex", "0 0 132px")
          .style("color", C.muted).text("one person each");
        wireLabel(lbl, singles.reduce(function (acc, m) { return acc.concat(m.members); }, []));
        var box = row.append("div").style("display", "flex").style("flex-wrap", "wrap")
          .style("gap", "5px 14px").style("align-items", "center").style("padding-top", "1px");
        singles.forEach(function (m) {
          var g = box.append("span").style("display", "inline-flex")
            .style("align-items", "center").style("gap", "5px");
          var sub = g.append("span").attr("class", "m-axis").style("color", C.muted)
            .text(shortName(m.name));
          wireLabel(sub, m.members);
          m.members.forEach(function (id) { dot(g, byId[id], shared.colorOf(id), false); });
        });
      }

      left.append("div").attr("class", "m-axis").style("color", C.muted)
        .style("padding", "6px 0").style("border-top", "1px solid #efe9dc")
        .text(d.n_city_unknown + " people, city unrecorded.");
    }
  });
})();
