/* assets/js/analyses/heating-up.js — vertex-centered scan statistic: where it's heating up */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("heating-up", {
    prose: {
      intro:
        "<p>Everything on this map is busier than it was three years ago — interpretability is " +
        "having a moment, and the graph shows it. The sharper question is <em>where</em>: which " +
        "corner isn’t just rising with the tide but outrunning its own past? This panel scans " +
        "every person’s neighborhood, year by year, and points at the one that ignited.</p>",
      how:
        "<p>A scan statistic slides a window over the data and asks where the count inside most " +
        "exceeds its own baseline — anomaly detection where the anomaly is a <em>place</em>, not " +
        "a point in a time series. Here the window is one person’s radius-1 neighborhood: them, " +
        "their co-authors, and every tie among those co-authors. Each year we total the " +
        "collaboration weight inside that window on the cumulative graph — every tie counting " +
        "in proportion to the papers behind it.</p>" +
        "<p>The year-over-year jump is then standardized against that same neighborhood’s " +
        "<em>own</em> history — the way you’d judge a loss spike against the run’s own curve, not " +
        "against some other model’s. A quiet corner that suddenly wakes up therefore scores far " +
        "higher than a busy corner staying busy. The hot spot is wherever recent growth " +
        "(2024–2026) most exceeds the local baseline: sliding-window detection, with the window " +
        "sliding over graph locations instead of timesteps.</p>",
      method:
        "<p>The vertex-centered scan statistic of Priebe, Conroy, Marchette &amp; Park, “Scan " +
        "Statistics on Enron Graphs,” <em>Computational &amp; Mathematical Organization Theory</em> " +
        "11(3), 229–247, 2005 — a method built to catch anomalous bursts of email traffic inside " +
        "Enron before the collapse. The easter egg writes itself: Carey E. Priebe is a node on " +
        "this very map (Vogelstein cluster), so the fraud-hunting statistic is here pointed at " +
        "its own inventor’s collaboration neighborhood. Verdict: nothing to report — his corner " +
        "ranks second-to-last of the 102 scored, large and steady, lately growing slightly slower than " +
        "its own history predicts. The mechanics: the locality statistic Ψ<sub>t</sub>(v) is the " +
        "induced edge weight of v’s closed one-hop neighborhood on the cumulative graph at year " +
        "t; edges use fractional co-authorship counting (each paper adds 1/n_authors to each of " +
        "its co-author pairs; suggested by Stella Biderman), so a pair’s tie strength is the sum " +
        "of 1/n_authors over their shared papers. That fixes an old caveat of this panel: under " +
        "the previous pair-×-paper units a single hypothetical 30-author paper would drop 435 " +
        "full-strength ties at once (one per pair on it, 30·29/2) and could light up a whole " +
        "neighborhood by itself; now each of those ties carries 1/30 weight. The " +
        "statistic is first-differenced and standardized against v’s own earlier differences " +
        "(standard deviation floored at 1), then maximized over 2024–2026. Neighborhoods with no " +
        "edges before year t are not scored — a debut has no baseline of its own. Caveats: 2026 " +
        "is only partially observed, understating the newest growth; and the source databases " +
        "keep some name variants apart (a “Samuel Marks” beside a “Samuel D. Marks,” an “E. " +
        "Lubana” beside an “Ekdeep Singh Lubana”), which can split one person’s record in two — " +
        "an artifact of the databases, never a comment on the people.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, hot = d.hot, esc = shared.esc, C = shared.colors;
      var inHot = {};
      hot.members.forEach(function (m) { inHot[m] = true; });
      var hotColor = shared.colorOf(hot.center);
      var lastYear = hot.series[hot.series.length - 1][0];
      var lastW = hot.series[hot.series.length - 1][1];
      function trendAt(yr) { return hot.trend.slope * yr + hot.trend.intercept; }
      var elW = el.clientWidth || 660;

      // ---- top row: large mini-map (hot neighborhood lit, rest dimmed) + ranked corners
      var row = document.createElement("div");
      row.style.cssText = "display:flex;flex-wrap:wrap;gap:14px 30px;align-items:flex-start";
      el.appendChild(row);

      var mapCol = document.createElement("div");
      row.appendChild(mapCol);
      var mapCap = document.createElement("div");
      mapCap.className = "m-map-cap";
      mapCap.style.cssText = "text-align:left;margin:0 0 4px";
      mapCap.textContent = "the hot neighborhood — " + hot.members.length + " people around " +
        shared.labelOf(hot.center) + " in " + hot.year + "; everyone else dimmed";
      mapCol.appendChild(mapCap);
      var mapDiv = document.createElement("div");
      mapCol.appendChild(mapDiv);

      var mapW = Math.max(260, Math.min(420, elW - 8));
      var mapH = Math.round(mapW * 320 / 420);
      var map = shared.minimap(mapDiv, function (id) { return shared.colorOf(id); }, {
        width: mapW, height: mapH, edgeOpacity: 0.45,
        radiusFn: function (id) { return id === hot.center ? 5 : (inHot[id] ? 3.6 : 2.2); },
        opacityFn: function (id) { return inHot[id] ? 0.95 : 0.12; },
        onReady: function (api) {
          var c = api.svg.selectAll("circle")
            .filter(function (n) { return !!(n && n.id === hot.center); });
          if (!c.empty()) {
            var cx = +c.attr("cx"), cy = +c.attr("cy"), flip = cx > mapW - 110;
            api.svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 8)
              .attr("fill", "none").attr("stroke", C.ink).attr("stroke-width", 1);
            api.svg.append("text").attr("class", "m-label")
              .attr("x", flip ? cx - 12 : cx + 12).attr("y", cy + 4)
              .attr("text-anchor", flip ? "end" : "start")
              .text(shared.labelOf(hot.center));
          }
          api.svg.selectAll("circle")
            .filter(function (n) { return !!(n && n.id && inHot[n.id]); })
            .on("mouseenter", function (evt, n) {
              shared.tooltip.show("<strong>" + esc(shared.labelOf(n.id)) + "</strong><br>in " +
                esc(shared.labelOf(hot.center)) + "’s " + hot.year + " neighborhood", evt);
            })
            .on("mouseleave", function () { shared.tooltip.hide(); });
        }
      });

      var side = document.createElement("div");
      side.style.cssText = "flex:1;min-width:210px;max-width:280px";
      row.appendChild(side);
      var head = document.createElement("div");
      head.className = "m-map-cap";
      head.style.cssText = "text-align:left;margin:0 0 6px";
      head.textContent = "hottest corners, ranked by scan score";
      side.appendChild(head);

      var entries = [{ id: hot.center, S: hot.S, year: hot.year, w_prev: hot.w_prev,
                       w: hot.w, win: true }].concat(d.runners);
      entries.forEach(function (r) {
        var name = esc(shared.labelOf(r.id));
        var nameLink = shared.seatLink(r.id);
        var rowEl = document.createElement("div");
        rowEl.style.cssText = "padding:4px 0;cursor:default";
        rowEl.innerHTML =
          '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' +
          shared.colorOf(r.id) + ";opacity:" + Math.max(0.45, r.S / hot.S).toFixed(2) +
          ';margin-right:7px"></span>' +
          (r.win ? "<strong>" + nameLink + "</strong>" : nameLink) +
          ' <span style="color:' + C.muted + ';font-size:12px">· ' + r.year + "</span>" +
          '<div style="margin-left:15px;color:' + C.muted + ';font-size:12px;line-height:1.4">' +
          "weight " + shared.fmt.num(r.w_prev) + " → " + shared.fmt.num(r.w) +
          " · score " + shared.fmt.num(r.S) + "</div>";
        rowEl.addEventListener("mouseenter", function (evt) {
          map.highlight(r.id);
          shared.tooltip.show("<strong>" + name + "</strong><br>neighborhood weight grew " +
            shared.fmt.num(r.w_prev) + " → " + shared.fmt.num(r.w) + " in " + r.year +
            "<br>scan score " + shared.fmt.num(r.S), evt);
        });
        rowEl.addEventListener("mouseleave", function () {
          map.highlight(null); shared.tooltip.hide();
        });
        side.appendChild(rowEl);
      });

      // ---- below: the hot neighborhood's weight-by-year curve vs its own pre-hot trend
      var cap = document.createElement("div");
      cap.className = "m-map-cap";
      cap.style.cssText = "text-align:left;margin:20px 0 2px";
      cap.textContent = "collaboration weight inside " + shared.labelOf(hot.center) +
        "’s neighborhood, cumulative by year, against its pre-" + hot.year + " trend";
      el.appendChild(cap);
      var chartDiv = document.createElement("div");
      el.appendChild(chartDiv);

      var Wc = Math.max(300, Math.min(elW, 660)), H = 215, narrow = Wc < 520;
      var m = { l: 38, r: narrow ? 96 : 172, t: 14, b: 26 };
      var svg = d3.select(chartDiv).append("svg").attr("width", Wc).attr("height", H);
      var x = d3.scaleLinear().domain([hot.series[0][0], lastYear]).range([m.l, Wc - m.r]);
      var y = d3.scaleLinear().domain([0, lastW * 1.06]).range([H - m.b, m.t]);

      y.ticks(4).forEach(function (t) {
        svg.append("line").attr("x1", m.l).attr("x2", Wc - m.r).attr("y1", y(t)).attr("y2", y(t))
          .attr("stroke", t === 0 ? C.hair : "#efe9dc").attr("stroke-width", 1);
        if (t > 0) svg.append("text").attr("class", "m-axis").attr("x", m.l - 6).attr("y", y(t) + 3)
          .attr("text-anchor", "end").text(t);
      });
      hot.series.forEach(function (s) {
        if (s[0] % 2 === 0) svg.append("text").attr("class", "m-axis")
          .attr("x", x(s[0])).attr("y", H - m.b + 15).attr("text-anchor", "middle").text(s[0]);
      });

      // dashed hairline: the pre-hot trend, extrapolated through today
      var trendPts = hot.series
        .map(function (s) { return [s[0], trendAt(s[0])]; })
        .filter(function (p) { return p[1] >= 0; });
      svg.append("path")
        .attr("d", d3.line().x(function (p) { return x(p[0]); }).y(function (p) { return y(p[1]); })(trendPts))
        .attr("fill", "none").attr("stroke", C.muted).attr("stroke-width", 1)
        .attr("stroke-dasharray", "3,3");
      svg.append("text").attr("class", "m-anno")
        .attr("x", x(lastYear - 3)).attr("y", y(Math.max(0, trendAt(lastYear))) - 7)
        .attr("text-anchor", "end").text("pre-" + hot.year + " trend →");

      // the real curve
      svg.append("path")
        .attr("d", d3.line().x(function (s) { return x(s[0]); }).y(function (s) { return y(s[1]); })(hot.series))
        .attr("fill", "none").attr("stroke", hotColor).attr("stroke-width", 1.8);
      hot.series.forEach(function (s) {
        svg.append("circle").attr("cx", x(s[0])).attr("cy", y(s[1])).attr("r", 2.4).attr("fill", hotColor);
        svg.append("circle").attr("cx", x(s[0])).attr("cy", y(s[1])).attr("r", 9)
          .attr("fill", "transparent")
          .on("mouseenter", function (evt) {
            shared.tooltip.show(s[0] + " · weight <strong>" + shared.fmt.num(s[1]) +
              "</strong> inside the neighborhood<br>its old trend predicted " +
              shared.fmt.num(Math.max(0, trendAt(s[0]))), evt);
          })
          .on("mouseleave", function () { shared.tooltip.hide(); });
      });

      // the year the scan flags
      var hotPt = hot.series.filter(function (s) { return s[0] === hot.year; })[0];
      svg.append("circle").attr("cx", x(hotPt[0])).attr("cy", y(hotPt[1])).attr("r", 5)
        .attr("fill", "none").attr("stroke", C.ink).attr("stroke-width", 1);
      svg.append("text").attr("class", "m-label").attr("x", x(hotPt[0]) - 9).attr("y", y(hotPt[1]) - 2)
        .attr("text-anchor", "end")
        .text(narrow ? "+" + shared.fmt.num(hot.D) + " weight in one year"
                     : "+" + shared.fmt.num(hot.D) + " weight — " + hot.papers_hot +
                       " papers — in one year");
      svg.append("text").attr("class", "m-anno").attr("x", x(hotPt[0]) - 9).attr("y", y(hotPt[1]) + 12)
        .attr("text-anchor", "end").text("the scan flags " + hot.year);

      // gap bracket at the latest year: where it is vs where its old trend says it would be
      var bx = x(lastYear) + 6;
      var yAct = y(lastW), yTr = y(Math.max(0, trendAt(lastYear)));
      svg.append("line").attr("x1", bx).attr("x2", bx).attr("y1", yAct).attr("y2", yTr)
        .attr("stroke", C.ink).attr("stroke-width", 1);
      [yAct, yTr].forEach(function (yy) {
        svg.append("line").attr("x1", bx - 3).attr("x2", bx + 3).attr("y1", yy).attr("y2", yy)
          .attr("stroke", C.ink).attr("stroke-width", 1);
      });
      var midY = (yAct + yTr) / 2;
      svg.append("text").attr("class", "m-label").attr("x", bx + 7).attr("y", midY - 2)
        .text("+" + shared.fmt.num(hot.gap_last) + " weight");
      svg.append("text").attr("class", "m-anno").attr("x", bx + 7).attr("y", midY + 12)
        .text(narrow ? "above trend" : "above its old trend");
    }
  });
})();
