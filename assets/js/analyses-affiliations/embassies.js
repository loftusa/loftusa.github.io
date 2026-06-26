/* assets/js/analyses-affiliations/embassies.js — the organizations exactly one person holds */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("embassies", {
    prose: {
      intro:
        "<p>The shared rooms get the attention — Hopkins with thirteen people, the Bau lab with " +
        "eight. But most of this map is the opposite: places where exactly one person has " +
        "standing. Forty-eight people hold keys to 151 institutions, and 123 of those keys " +
        "exist in a single copy. Read them not as gaps but as embassies — territory one person " +
        "holds on behalf of the whole chat. Want an intro at Astera? There&rsquo;s an embassy.</p>",
      how:
        "<p>The computation is just a tally: count how many of the group each organization contains. " +
        "Plot the counts and you get the signature heavy-tailed curve — the same shape as token " +
        "frequencies in a corpus, or as the features in a sparse autoencoder dictionary: a few " +
        "hub features fire everywhere (Hopkins, a dozen-plus people) while most encode exactly one thing, " +
        "and the dictionary&rsquo;s coverage lives in that long tail.</p>" +
        "<p>In graph terms, careers form a bipartite people×orgs network, and this tally is its " +
        "org-side degree distribution. This panel won&rsquo;t claim a power law from 151 points — that " +
        "takes orders of magnitude more data. The honest claim is simpler: the tail isn&rsquo;t " +
        "the edge case, it <em>is</em> most of the graph — 123 of 151 organizations are " +
        "one-person outposts, so the distribution is drawn literally, as a shelf, with no fitted " +
        "line and no log axes.</p>",
      method:
        "<p>Org-side degree distribution of a two-mode (bipartite) affiliation network — see " +
        "Latapy, Magnien &amp; Del Vecchio, &ldquo;Basic notions for the analysis of large " +
        "two-mode networks,&rdquo; Social Networks 2008. On heavy tails and Zipf-like size " +
        "distributions: Newman, &ldquo;Power laws, Pareto distributions and Zipf&rsquo;s " +
        "law,&rdquo; Contemporary Physics 2005. Nothing is deliberately fitted here: reliable power-law " +
        "inference needs far more than 151 observations (Clauset, Shalizi &amp; Newman, " +
        "&ldquo;Power-law distributions in empirical data,&rdquo; SIAM Review 2009), so the " +
        "claim stops at &ldquo;heavy-tailed.&rdquo; Caveats: &ldquo;solo&rdquo; means solo " +
        "among these 48 — the organizations are full of people, just not chat people; a missed " +
        "alias in the org canonicalizer could split one organization into two false embassies; " +
        "and the sweep records public statements only, so the tail undercounts quiet " +
        "memberships. Three people hold no solo org — a count reported without names.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, C = shared.colors;
  // org-type palette — keep in sync across the analyses-affiliations panels + the map page
      var TYPE_COLOR = {
        lab: "#8c510a", program: "#2166ac", company: "#7b3294",
        community: "#35978f", university: "#98917f"
      };
      var TYPE_PLURAL = {
        company: "companies", university: "universities", lab: "labs",
        program: "programs", community: "communities"
      };

      var totalW = el.clientWidth || 680;
      var miniW = 200, gap = 24;
      var wide = totalW >= 660;
      var W = Math.max(360, Math.min(wide ? totalW - miniW - gap : totalW, 720));
      var labW = 112;

      var wrap = d3.select(el).append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", gap + "px").style("align-items", "flex-start");
      var left = wrap.append("div").style("flex", "0 0 " + W + "px").style("max-width", "100%");
      var right = wrap.append("div").style("flex", "0 0 " + miniW + "px");

      // ---- header strip ----
      var head = left.append("div")
        .style("display", "flex").style("gap", "10px").style("margin-bottom", "2px");
      head.append("div").attr("class", "m-axis").style("flex", "0 0 " + labW + "px")
        .style("color", C.muted).text("held by");
      head.append("div").attr("class", "m-axis").style("flex", "1 1 auto")
        .style("color", C.muted).text("one square per organization, colored by type");

      // ---- minimap (declared early so row hovers can call highlightSet) ----
      var kmax = 0;
      Object.keys(d.holders).forEach(function (p) { kmax = Math.max(kmax, d.holders[p]); });
      function baseOp(id) {
        var k = d.holders[id] || 0;
        return k ? 0.3 + 0.7 * (k / kmax) : 0.15;
      }
      var mapBox = right.append("div");
      var ambBox = right.append("div");   // filled below; map first, list under it
      var mini = shared.minimap(mapBox.node(), function (id) {
        return d.holders[id] ? shared.colorOf(id) : null;
      }, {
        opacityFn: baseOp,
        radiusFn: function (id) { return d.holders[id] ? 2.9 : 2; }
      });
      var nZero = Math.max(0, (shared.nodes.size || 48) - Object.keys(d.holders).length);
      mapBox.append("div").attr("class", "m-map-cap")
        .text("ink = how many embassies each person keeps; grey = none (" + nZero + " people)");

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
        circles.attr("stroke", function (n) { return set[n.id] ? C.ink : null; })
          .attr("stroke-width", 1.4)
          .attr("fill-opacity", function (n) { return set[n.id] ? 1 : 0.18; });
      }

      // ---- org tooltip + square hover wiring ----
      function orgTip(o) {
        var who = o.members.map(function (id) {
          return shared.esc(shared.labelOf(id));
        }).join(" · ");
        var h = "<strong>" + shared.esc(o.label) + "</strong><br>" +
          "<span style='color:" + C.muted + "'>" + shared.esc(o.type) +
          (o.members.length > 1 ? " · " + o.members.length + " people" : " · embassy") +
          "</span><div style='margin-top:2px'>" + who + "</div>";
        if (o.members.length === 1) {
          if (o.role) h += "<div style='color:" + C.muted + ";margin-top:2px'>" +
            shared.esc(o.role) + "</div>";
          if (o.years) h += "<div style='color:" + C.muted + "'>" +
            shared.esc(o.years) + "</div>";
        }
        return h;
      }
      function wireOrg(sel, o, sq) {
        sel.style("cursor", "default")
          .on("mouseenter", function (evt) {
            sq.style("outline", "1.5px solid " + C.ink);
            highlightSet(o.members);
            shared.tooltip.show(orgTip(o), evt);
          })
          .on("mousemove", function (evt) { shared.tooltip.show(orgTip(o), evt); })
          .on("mouseleave", function () {
            sq.style("outline", null);
            highlightSet(null);
            shared.tooltip.hide();
          });
      }
      function square(parent, type, size) {
        return parent.append("span")
          .style("width", size + "px").style("height", size + "px")
          .style("background", TYPE_COLOR[type] || C.other)
          .style("flex", "0 0 auto").style("display", "inline-block");
      }

      // ---- the shelf: one row per membership count, top-to-bottom 13 → 1 ----
      d.tally.forEach(function (r, ri) {
        var row = left.append("div")
          .style("display", "flex").style("align-items", "flex-start").style("gap", "10px")
          .style("padding", "6px 0")
          .style("border-top", "1px solid " + (ri === 0 ? C.hair : "#efe9dc"));

        var lab = row.append("div").attr("class", "m-axis")
          .style("flex", "0 0 " + labW + "px").style("padding-top", "1px");
        if (r.n === 1) {
          lab.append("div").style("color", C.ink).text("exactly one person");
          lab.append("div").style("color", C.ink).style("margin-top", "1px")
            .html("<strong>" + d.n_solo + "</strong> organizations");
        } else {
          lab.style("color", C.muted).text(r.n + " people");
        }

        var body = row.append("div")
          .style("flex", "1 1 auto").style("display", "flex").style("flex-wrap", "wrap")
          .style("gap", r.n === 1 ? "2px" : "4px 14px").style("align-items", "center");

        r.orgs.forEach(function (o) {
          if (r.n === 1) {
            var sq = square(body, o.type, 10);
            wireOrg(sq, o, sq);
          } else {
            var chip = body.append("span")
              .style("display", "inline-flex").style("align-items", "center")
              .style("gap", "5px").style("white-space", "nowrap");
            var csq = square(chip, o.type, 10);
            chip.append("span").attr("class", "m-label").style("color", C.ink)
              .text(o.label);
            wireOrg(chip, o, csq);
          }
        });
      });

      // ---- type legend doubling as the solo-type mix ----
      var legend = left.append("div")
        .style("display", "flex").style("flex-wrap", "wrap").style("gap", "4px 14px")
        .style("margin-top", "8px").style("border-top", "1px solid #efe9dc")
        .style("padding-top", "6px");
      Object.keys(d.solo_types).forEach(function (t) {
        var item = legend.append("span")
          .style("display", "inline-flex").style("align-items", "center").style("gap", "5px");
        square(item, t, 8);
        item.append("span").attr("class", "m-axis").style("color", C.muted)
          .text(d.solo_types[t] + " " + (TYPE_PLURAL[t] || t));
      });
      legend.append("span").attr("class", "m-axis").style("color", C.muted)
        .text("— the " + d.n_solo + " embassies by type");

      left.append("div").attr("class", "m-note").style("margin-top", "8px")
        .html((shared.nodes.size - nZero) + " of " + shared.nodes.size + " people hold at " +
          "least one embassy; " + nZero + " hold none. &ldquo;Solo&rdquo; means solo among " +
          "these 48 — the buildings are full of people, just not chat people. Hover any " +
          "square for the org and who holds it.");

      // ---- ambassadors: everyone keeping ≥5 embassies ----
      ambBox.append("div").attr("class", "m-axis")
        .style("color", C.muted).style("margin", "10px 0 2px")
        .text("the ambassadors — 5 or more embassies");
      d.ambassadors.forEach(function (a) {
        var row = ambBox.append("div")
          .style("display", "flex").style("align-items", "center").style("gap", "6px")
          .style("padding", "3px 0").style("cursor", "default");
        row.append("span")
          .style("width", "7px").style("height", "7px").style("border-radius", "50%")
          .style("background", shared.colorOf(a.id)).style("flex", "0 0 auto");
        row.append("span").attr("class", "m-label").style("color", C.ink)
          .html(shared.seatLink(a.id) +
            " <span style='color:" + C.muted + "'>— " + a.k + " embassies</span>");
        function ambTip(evt) {
          var rows = a.orgs.map(function (lbl) {
            return "<div style='margin-top:2px'>" + shared.esc(lbl) + "</div>";
          }).join("");
          shared.tooltip.show("<strong>" + shared.esc(shared.labelOf(a.id)) + "</strong><br>" +
            "<span style='color:" + C.muted + "'>" + a.k +
            " orgs where they are the only member</span>" + rows, evt);
        }
        row.on("mouseenter", function (evt) { mini.highlight(a.id); ambTip(evt); })
          .on("mousemove", ambTip)
          .on("mouseleave", function () { mini.highlight(null); shared.tooltip.hide(); });
      });
    }
  });
})();
