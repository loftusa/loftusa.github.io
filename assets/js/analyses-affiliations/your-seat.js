/* assets/js/analyses-affiliations/your-seat.js — the People capstone: the network read from one chair */
(function () {
  "use strict";
  (window.AnalysesRegistry = window.AnalysesRegistry || {
    _q: [], register: function (s, d) { this._q.push([s, d]); }
  }).register("your-seat", {
    prose: {
      intro:
        "<p>Every panel so far reads the whole network at once — all the rooms, all the waves, " +
        "the full transition table. This one turns the map around and reads it from one chair: " +
        "pick any name (or find yours) and the six analyses above are re-asked about that single " +
        "person — their strongest threads, their open doors, their chapters, what history " +
        "suggests comes next. Seats are deep-linkable: <code>?p=</code> in the URL carries a " +
        "seat, so anyone can be handed theirs.</p>",
      how:
        "<p>Nothing new is computed here — this panel is a gather operation, not a model. Each " +
        "of the six analyses already published its results as data: the projection&rsquo;s " +
        "threads, the shared-rooms-without-a-paper pairs, the cohort spans, the transition " +
        "table, the range scores, the solo organizations. This panel indexes all six by one " +
        "person id and lays the rows side by side — like reading a single row of every " +
        "attention map instead of staring at six full matrices.</p>" +
        "<p>The only judgment calls are editorial: which five threads to surface, and reading " +
        "&ldquo;what&rsquo;s next&rdquo; as the two most likely moves out of the person&rsquo;s " +
        "current kind of room, taken straight from the pipeline panel&rsquo;s transition row. " +
        "Because it is a view and not a recomputation, every number here can be checked against " +
        "the panel it came from.</p>",
      method:
        "<p>Ego-network (ego-centric) analysis: fix a focal node, then describe its alters, " +
        "ties, and attributes — the standard treatment is Wasserman &amp; Faust, <em>Social " +
        "Network Analysis: Methods and Applications</em>, Cambridge University Press, 1994. " +
        "All numbers are read verbatim from the six parent panels&rsquo; published data; " +
        "nothing is re-derived. Tie weights come from the career map&rsquo;s " +
        "nesting-discounted bipartite projection (a shared lab displaces its university in the " +
        "weight), so threads here match the map page exactly. Caveats inherit from the " +
        "parents: the sweep records public bios only; &ldquo;open doors&rdquo; count rooms " +
        "both people verifiably shared without a joint paper; and three people&rsquo;s " +
        "chapters carry no dates, so their pipeline reading is omitted rather than guessed.</p>"
    },
    render: function (el, data, shared) {
      var d = data.data, C = shared.colors;
      // org-type palette — keep in sync across the analyses-affiliations panels + the map page
      var TYPE = { lab: "#8c510a", program: "#2166ac", company: "#7b3294",
                   community: "#35978f", university: "#98917f" };

      var totalW = el.clientWidth || 680;
      var miniW = 200, gap = 24;
      var wide = totalW >= 660;
      var W = Math.max(360, Math.min(wide ? totalW - miniW - gap : totalW, 720));
      var twoCol = W >= 540;

      var root = d3.select(el);
      var header = root.append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("justify-content", "space-between").style("align-items", "flex-end")
        .style("gap", "10px 16px").style("margin-bottom", "14px")
        .style("max-width", (wide ? W + gap + miniW : W) + "px");
      var hLeft = header.append("div");
      var hRight = header.append("div").style("flex", "0 0 auto");

      var body = root.append("div")
        .style("display", "flex").style("flex-wrap", "wrap")
        .style("gap", gap + "px").style("align-items", "flex-start");
      var left = body.append("div").style("flex", "0 0 " + W + "px").style("max-width", "100%");
      var grid = left.append("div")
        .style("display", "grid")
        .style("grid-template-columns", twoCol ? "repeat(2, minmax(0, 1fr))" : "1fr")
        .style("column-gap", "26px").style("align-items", "start");
      var right = body.append("div").style("flex", "0 0 " + miniW + "px");

      // ---- resolve the seat: ?p= → localStorage identity → the host's chair ----
      var qp = new URLSearchParams(location.search).get("p");
      var cur = null, unknown = null;
      if (qp) {
        if (d.people[qp]) cur = qp; else unknown = qp;
      } else {
        try {
          var idn = JSON.parse(localStorage.getItem("network_identity") || "null");
          if (idn && idn.id && d.people[idn.id]) cur = idn.id;
        } catch (e) { /* malformed identity — fall through */ }
        if (!cur) cur = "alex loftus";
      }

      // ---- person picker (chrome — the one place this panel says "you") ----
      var byLabel = {};
      d.order.forEach(function (id) { byLabel[shared.labelOf(id).toLowerCase()] = id; });
      var input = hRight.append("input")
        .attr("list", "your-seat-names")
        .attr("placeholder", "pick a name — or find yours")
        .style("font-size", "12.5px").style("color", C.ink)
        .style("background", "#fff").style("border", "1px solid " + C.hair)
        .style("border-radius", "4px").style("padding", "5px 9px").style("width", "190px");
      var dl = hRight.append("datalist").attr("id", "your-seat-names");
      d.order.forEach(function (id) {
        dl.append("option").attr("value", shared.labelOf(id));
      });
      input.on("input change", function () {
        var id = byLabel[this.value.trim().toLowerCase()];
        if (!id || id === cur) return;
        cur = id; unknown = null;
        var u = new URL(location.href);
        u.searchParams.set("p", id);
        history.replaceState(null, "", u.toString());   // keeps the #your-seat hash
        this.value = "";
        drawPerson(id);
      });

      // ---- mini-map: rebuilt per seat so the ego's threads wear their colors ----
      var mini = { highlight: function () {} };
      function drawMap(pid, capText) {
        right.html("");
        var tieSet = {};
        if (pid) {
          d.people[pid].tie_ids.forEach(function (id) { tieSet[id] = true; });
          tieSet[pid] = true;
        }
        mini = shared.minimap(right.node(), function (id) {
          return tieSet[id] ? shared.colorOf(id) : null;
        }, pid ? { onReady: function (api) { api.highlight(pid); } } : {});
        right.append("div").attr("class", "m-map-cap").style("text-align", "left")
          .text(capText);
      }

      // ---- shared row plumbing ----
      function hoverPerson(row, otherId, tipFn) {
        row.style("cursor", "default")
          .on("mouseenter", function (evt) {
            d3.select(this).style("background", "#f4f1e9");
            mini.highlight(otherId);
            shared.tooltip.show(tipFn(), evt);
          })
          .on("mousemove", function (evt) { shared.tooltip.show(tipFn(), evt); })
          .on("mouseleave", function () {
            d3.select(this).style("background", null);
            mini.highlight(cur);              // restore the seat's own highlight
            shared.tooltip.hide();
          });
      }
      function orgHtml(names) {            // org names in their (muted) type colors
        return names.map(function (o) {
          return "<span style='color:" + (TYPE[d.org_types[o]] || C.muted) + "'>" +
            shared.esc(o) + "</span>";
        }).join("<span style='color:" + C.muted + "'> · </span>");
      }
      function card(title) {
        var c = grid.append("div").style("min-width", "0")
          .style("border-top", "1px solid " + C.hair).style("padding", "10px 0 16px");
        c.append("div").attr("class", "m-axis").style("color", C.muted)
          .style("font-weight", "600").style("letter-spacing", ".07em")
          .style("text-transform", "uppercase").style("margin-bottom", "7px")
          .text(title);
        return c;
      }
      function note(c, txt) {
        c.append("div").style("font-size", "12.5px").style("color", C.muted)
          .style("line-height", "1.45").text(txt);
      }
      function spanText(ch) {
        if (ch.start == null && ch.end == null) return "undated";
        if (ch.spell) return String(ch.start);
        if (ch.ongoing) return ch.start + "–";
        if (ch.start == null) return "–" + ch.end;
        return ch.start + "–" + ch.end;
      }

      // ---- the six cards ----
      function threadsCard(p, label, title) {
        var c = card(title);
        if (!p.ties.length) {
          note(c, "no threads in the projection yet — " + label +
            "’s rooms don’t overlap the chat’s shared rooms.");
          return;
        }
        var maxW = d3.max(p.ties, function (t) { return t.w; });
        p.ties.forEach(function (t) {
          var row = c.append("div").style("padding", "4px 2px");
          row.append("div").style("font-size", "12.5px").style("color", C.ink)
            .html(shared.seatLink(t.other));
          row.append("div").style("height", "3px")
            .style("width", Math.max(6, Math.round(100 * t.w / maxW)) + "%")
            .style("background", shared.colorOf(t.other)).style("opacity", 0.85)
            .style("margin", "3px 0 2px");
          row.append("div").style("font-size", "11px").style("line-height", "1.5")
            .html(orgHtml(t.orgs));
          hoverPerson(row, t.other, function () {
            return "<strong>" + shared.esc(shared.labelOf(t.other)) + "</strong>" +
              "<div style='color:" + C.muted + "'>thread weight " + shared.fmt.num(t.w) +
              "</div><div style='margin-top:2px'>" + orgHtml(t.orgs) + "</div>";
          });
        });
        if (p.n_ties > p.ties.length) {
          c.append("div").style("font-size", "11px").style("color", C.muted)
            .style("margin-top", "4px")
            .text("the 5 strongest of " + p.n_ties + " threads — all of them lit on the map");
        }
      }

      function invitationsCard(p, label, promoted) {
        var c = card("Open invitations");
        if (!p.invitations.length) {
          note(c, "every shared room on " + label + "’s record has already produced a paper.");
          return;
        }
        if (promoted) {
          c.append("div").style("font-size", "12.5px").style("color", "#4a463f")
            .style("line-height", "1.45").style("margin-bottom", "6px")
            .html("the network is holding <strong>" + p.n_invitations +
              "</strong> open door" + (p.n_invitations === 1 ? "" : "s") + " for " +
              shared.esc(label) + ".");
        }
        p.invitations.forEach(function (v) {
          var row = c.append("div").style("font-size", "12.5px").style("padding", "3px 2px")
            .style("line-height", "1.45");
          row.html("<span style='color:" + C.ink + "'>" +
            shared.seatLink(v.other) + "</span><span style='color:" + C.muted +
            "'> — shared " + v.orgs.map(shared.esc).join(", ") + "</span>");
          hoverPerson(row, v.other, function () {
            return "<strong>" + shared.esc(shared.labelOf(v.other)) + "</strong>" +
              "<div style='color:" + C.muted + "'>" + v.overlap + " shared room" +
              (v.overlap === 1 ? "" : "s") + ", no joint paper yet</div>" +
              "<div style='margin-top:2px'>" + orgHtml(v.orgs) + "</div>";
          });
        });
        if (p.n_invitations > p.invitations.length) {
          c.append("div").style("font-size", "11px").style("color", C.muted)
            .style("margin-top", "4px")
            .text("and " + (p.n_invitations - p.invitations.length) + " more open doors");
        }
      }

      function chaptersCard(p, label) {
        var c = card("Chapters");
        if (!p.chapters.length) {
          note(c, "no chapters on record for " + label + " yet.");
          return;
        }
        p.chapters.forEach(function (ch) {
          var row = c.append("div").style("display", "flex").style("align-items", "baseline")
            .style("gap", "7px").style("padding", "3px 2px").style("font-size", "12.5px");
          row.append("span").style("width", "9px").style("height", "9px")
            .style("background", TYPE[ch.type] || C.other)
            .style("flex", "0 0 auto").style("align-self", "center");
          row.append("span").style("color", C.ink).style("flex", "1 1 auto")
            .style("min-width", "0").text(ch.org);
          row.append("span").attr("class", "m-axis").style("color", C.muted)
            .style("flex", "0 0 auto").text(spanText(ch));
          var chTip = function () {
            return "<strong>" + shared.esc(ch.org) + "</strong>" +
              "<div style='color:" + C.muted + "'>" + shared.esc(ch.type) + " · " +
              shared.esc(spanText(ch)) + "</div>" +
              (ch.role ? "<div style='margin-top:2px'>" + shared.esc(ch.role) + "</div>" : "");
          };
          row.style("cursor", "default")
            .on("mouseenter", function (evt) {
              d3.select(this).style("background", "#f4f1e9");
              shared.tooltip.show(chTip(), evt);
            })
            .on("mousemove", function (evt) { shared.tooltip.show(chTip(), evt); })
            .on("mouseleave", function () {
              d3.select(this).style("background", null);
              shared.tooltip.hide();
            });
        });
      }

      function pipelineCard(p, label) {
        var c = card("The pipeline says");
        if (!p.pipeline) {
          note(c, label + "’s chapters carry no dates yet, so the pipeline has nothing to read.");
          return;
        }
        var tape = c.append("div").style("display", "flex").style("flex-wrap", "wrap")
          .style("gap", "3px").style("align-items", "center");
        p.pipeline.seq.forEach(function (t, i) {
          var tkTip = function () {
            return "dated chapter " + (i + 1) + " of " + p.pipeline.seq.length +
              " — <span style='color:" + TYPE[t] + "'>" + shared.esc(t) + "</span>";
          };
          tape.append("span").style("width", "10px").style("height", "10px")
            .style("background", TYPE[t]).style("display", "inline-block")
            .style("flex", "0 0 auto").style("cursor", "default")
            .on("mouseenter", function (evt) { shared.tooltip.show(tkTip(), evt); })
            .on("mousemove", function (evt) { shared.tooltip.show(tkTip(), evt); })
            .on("mouseleave", function () { shared.tooltip.hide(); });
        });
        var curT = p.pipeline.current, nx = p.pipeline.next;
        c.append("div").style("font-size", "12.5px").style("margin-top", "7px")
          .html("<span style='color:" + C.muted + "'>now in a</span> <span style='color:" +
            TYPE[curT] + "'>" + shared.esc(curT) + "</span>");
        c.append("div").style("font-size", "12.5px").style("color", C.muted)
          .style("margin-top", "3px").style("line-height", "1.45")
          .html("history suggests next: <span style='color:" + TYPE[nx[0].type] + "'>" +
            shared.esc(nx[0].type) + "</span> (" + shared.fmt.pct(nx[0].p) +
            "), then <span style='color:" + TYPE[nx[1].type] + "'>" + shared.esc(nx[1].type) +
            "</span> (" + shared.fmt.pct(nx[1].p) + ")");
      }

      function rangeCard(p, label) {
        var c = card("Range");
        var r = p.range;
        var line = c.append("div").style("font-size", "12.5px").style("color", "#4a463f")
          .style("line-height", "1.5");
        if (r.low) {
          line.html(shared.esc(label) + "’s range is still being written — <strong>" + r.n +
            "</strong> room" + (r.n === 1 ? "" : "s") + " on record so far.");
        } else {
          line.html(shared.esc(label) + "’s record covers <strong>" + r.n +
            "</strong> rooms — effectively <strong>" + shared.fmt.num(r.n1) +
            "</strong> different kinds of institution.");
        }
        if (r.quadrant) {
          c.append("div").style("font-size", "12.5px").style("font-style", "italic")
            .style("color", "#6b665d").style("margin-top", "4px")
            .text("a " + r.quadrant + ".");
        }
      }

      function embassiesCard(p, label) {
        var c = card("Embassies");
        if (!p.embassies.n) {
          note(c, "every organization on " + label + "’s record is shared with someone here — " +
            "no solo outposts, all company.");
          return;
        }
        c.append("div").style("font-size", "12.5px").style("color", "#4a463f")
          .style("line-height", "1.45")
          .html("<strong>" + p.embassies.n + "</strong> room" +
            (p.embassies.n === 1 ? "" : "s") + " where " + shared.esc(label) +
            " is the chat’s only member:");
        c.append("div").style("font-size", "11.5px").style("margin-top", "5px")
          .style("line-height", "1.7").html(orgHtml(p.embassies.orgs));
        if (p.embassies.n > p.embassies.orgs.length) {
          c.append("div").style("font-size", "11px").style("color", C.muted)
            .text("and " + (p.embassies.n - p.embassies.orgs.length) + " more.");
        }
      }

      // ---- per-seat assembly ----
      function drawHeader(title, metaHtml) {
        hLeft.html("");
        hLeft.append("h3").attr("class", "m-label")
          .style("margin", "0").style("font-size", "17px").style("font-weight", "600")
          .style("color", C.ink).text(title);
        hLeft.append("div").style("font-size", "12.5px").style("color", C.muted)
          .style("margin-top", "2px").html(metaHtml);
      }

      function drawPerson(pid) {
        var p = d.people[pid], label = shared.labelOf(pid);
        var bits = [];
        if (p.now.city) bits.push(shared.esc(p.now.city));
        bits.push(p.now.no_current ? "between rooms" : shared.esc(p.now.org || ""));
        drawHeader(label + "’s seat", bits.filter(Boolean).join(" · "));
        drawMap(pid, label + "’s corner of the map — their threads wear community colors; " +
          "hover a thread or an invitation to place its person");

        grid.html("");
        if (p.sparse) {            // no ties or invitations yet — invite, but show what IS here
          grid.append("div").style("grid-column", "1 / -1")
            .style("border-top", "1px solid " + C.hair).style("padding", "10px 2px 4px")
            .style("font-size", "12.5px").style("color", "#6b665d").style("line-height", "1.5")
            .html("<strong>The map is young in this corner</strong> — no shared rooms on record " +
              "for " + shared.esc(label) + " yet. " +
              '<a href="/networks/affiliations/?edit=1&p=' + encodeURIComponent(pid) +
              '" style="color:#6b665d">add a chapter on the career map →</a>');
        } else if (p.n_ties + p.n_invitations <= 1 && p.chapters.length <= 1) {
          var nroom = p.chapters.length;                 // thin-but-not-empty: a new arrival
          grid.append("div").style("grid-column", "1 / -1")
            .style("border-top", "1px solid " + C.hair).style("padding", "10px 2px 4px")
            .style("font-size", "12.5px").style("color", "#6b665d").style("line-height", "1.5")
            .html("<strong>The map is young in this corner</strong> — " + shared.esc(label) +
              " has " + (nroom ? "just one room" : "no rooms") + " on record so far, so the " +
              "threads here are still thin. This seat fills in as more of their affiliations are " +
              "added. " +
              '<a href="/networks/affiliations/?edit=1&p=' + encodeURIComponent(pid) +
              '" style="color:#6b665d">add a chapter on the career map →</a>');
        }
        if (p.n_ties < 3) {            // a young corner: lead with the open doors
          invitationsCard(p, label, true);
          threadsCard(p, label, "Newest threads");
        } else {
          threadsCard(p, label, "Strongest threads");
          invitationsCard(p, label, false);
        }
        chaptersCard(p, label);
        pipelineCard(p, label);
        rangeCard(p, label);
        embassiesCard(p, label);
      }

      function drawUnknown(raw) {
        drawHeader("This seat is being set",
          "“" + shared.esc(raw) + "” isn’t on the map yet");
        drawMap(null, "the map, holding the room until the seat is set");
        grid.html("");
        grid.append("div").style("grid-column", "1 / -1")
          .style("border-top", "1px solid " + C.hair).style("padding", "18px 2px")
          .style("font-size", "13.5px").style("color", "#4a463f")
          .style("line-height", "1.55").style("max-width", "56ch")
          .html("This seat is being set — new members get their seat after the next data refresh — usually a day or two. " +
            'Until then, the <a href="/networks/affiliations/" style="color:#6b665d">' +
            "career map</a> holds everyone whose chapters are already on record.");
      }

      if (unknown != null) drawUnknown(unknown); else drawPerson(cur);
    }
  });
})();
