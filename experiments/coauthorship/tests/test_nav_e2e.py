# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.55.0"]
# ///
"""Headless E2E of the two-level network/view nav + both analyses pages.

Prereq: `bundle exec jekyll build` from the repo root (serves _site/ locally).
Run:    cd experiments/coauthorship && uv run tests/test_nav_e2e.py

Covers: all four pages render two .tab-row rows with the right .on pills; switching network
preserves the view; each analyses page builds its nav from its own ANALYSES_CONFIG (item count
matches) and renders its first panel's .m-viz svg with zero page errors.
"""
import functools
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[3]
SITE = REPO / "_site"
PORT = 4127
assert (SITE / "networks" / "affiliations" / "analyses" / "index.html").exists(), \
    "run `bundle exec jekyll build` first"

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

PAGES = {  # path -> (network pill, view pill)
    "/networks/": ("papers", "map"),
    "/networks/analyses/": ("papers", "analyses"),
    "/networks/affiliations/": ("careers", "map"),
    "/networks/affiliations/analyses/": ("careers", "analyses"),
}

with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    # the identity picker greets first visits; dismiss it for nav assertions (it has its own e2e)
    page.add_init_script("localStorage.setItem('network_identity_dismissed', '1')")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    for path, (net, view) in PAGES.items():
        page.goto(f"http://127.0.0.1:{PORT}{path}")
        page.wait_for_selector(".page-tabs", timeout=15000)
        rows = page.locator(".page-tabs .tab-row")
        assert rows.count() == 2, (path, rows.count())
        on = [page.locator(".page-tabs .tab.on").nth(i).inner_text() for i in range(2)]
        assert on == [net, view], (path, on)
        # switching network preserves the view
        other = page.locator(".tab-row").first.locator("a.tab").first
        href = other.get_attribute("href")
        assert href.endswith("analyses/") == (view == "analyses"), (path, href)
    print("nav OK on all four pages (active pills + view-preserving links)")

    for path in ("/networks/analyses/", "/networks/affiliations/analyses/"):
        page.goto(f"http://127.0.0.1:{PORT}{path}")
        page.wait_for_selector(".nav-item", timeout=15000)
        n_cfg = page.evaluate("window.ANALYSES_CONFIG.methods.length")
        n_nav = page.locator("#nav .nav-item").count()
        assert n_nav == n_cfg, (path, n_nav, n_cfg)
        # activate EVERY panel: catches a methods entry whose script tag or JSON is missing
        # (that failure renders an .m-err only on activation, with no pageerror)
        for i in range(n_nav):
            item = page.locator("#nav .nav-item").nth(i)
            slug = item.get_attribute("data-slug")
            item.click()
            page.wait_for_selector(f"#panel-{slug}.active .m-viz svg", timeout=20000)
            assert page.locator(f"#panel-{slug} .m-err").count() == 0, (path, slug, "error panel")
        print(f"{path}: all {n_nav} panels activate and render")

    # your-seat: person deep link, picker switch, ?p= survives panel navigation
    page.goto(f"http://127.0.0.1:{PORT}/networks/affiliations/analyses/?p=can%20rager#your-seat")
    page.wait_for_selector("#panel-your-seat.active .m-viz svg", timeout=20000)
    assert "Can Rager" in page.locator("#panel-your-seat .m-viz h3").inner_text()
    page.locator("#panel-your-seat input[list]").fill("Kayo Yin")
    page.locator("#panel-your-seat input[list]").dispatch_event("change")
    page.wait_for_timeout(400)
    assert "Kayo Yin" in page.locator("#panel-your-seat .m-viz h3").inner_text()
    assert "p=kayo" in page.evaluate("location.search")
    page.locator(".nav-item[data-slug='eras']").click()
    page.wait_for_selector("#panel-eras.active .m-viz svg", timeout=20000)
    assert "p=kayo" in page.evaluate("location.search")   # ?p= survives panel nav
    print("your-seat deep link + person switch OK")

    # adaptive reach slider on the careers map: meaning follows view state
    page.goto(f"http://127.0.0.1:{PORT}/networks/affiliations/")
    page.wait_for_selector("#people li", timeout=15000)
    page.wait_for_timeout(800)
    assert "org size" in page.locator("#reach-name").inner_text()
    all_orgs = page.locator("#graph text.olabel").count()
    page.locator("#reach").fill(page.locator("#reach").get_attribute("max"))
    page.locator("#reach").dispatch_event("input")
    page.wait_for_timeout(300)
    big_orgs = page.locator("#graph text.olabel").count()
    assert 0 < big_orgs < all_orgs, (all_orgs, big_orgs)
    page.locator("#viewctl .mode[data-mode='people']").click()
    page.wait_for_timeout(300)
    assert "tie strength" in page.locator("#reach-name").inner_text()
    page.locator("#people li", has_text="Loftus").first.click()
    page.wait_for_timeout(300)
    assert "steps from Alexander" in page.locator("#reach-name").inner_text()
    lit_1hop = page.evaluate(
        "[...document.querySelectorAll('#graph g.person')].filter(g => g.getAttribute('opacity') === '1').length")
    page.locator("#reach").fill("3")
    page.locator("#reach").dispatch_event("input")
    page.wait_for_timeout(300)
    lit_3hop = page.evaluate(
        "[...document.querySelectorAll('#graph g.person')].filter(g => g.getAttribute('opacity') === '1').length")
    assert lit_3hop > lit_1hop, (lit_1hop, lit_3hop)
    # off-map hop reveal: hollow nodes (build_hops.py layer) appear as the reach widens
    hop_3 = page.locator("#graph g.hop").count()
    assert hop_3 > 0, "no off-map people revealed at 3 steps"
    page.locator("#reach").fill("1")
    page.locator("#reach").dispatch_event("input")
    page.wait_for_timeout(300)
    hop_1 = page.locator("#graph g.hop").count()
    assert hop_1 < hop_3, (hop_1, hop_3)
    print(f"reach slider OK (orgs {all_orgs}→{big_orgs} at max size; lit people {lit_1hop}→{lit_3hop}, "
          f"off-map {hop_1}→{hop_3} at 3 steps)")

    # hop person detail + the add-to-map funnel (reveal stays anchored to Loftus's reach)
    page.locator("#reach").fill("3")
    page.locator("#reach").dispatch_event("input")
    page.wait_for_timeout(600)
    page.locator("#graph g.hop").first.click(force=True)
    page.wait_for_timeout(300)
    assert page.locator("#graph g.hop").count() == hop_3   # clicking didn't collapse the reveal
    hop_name = page.locator("#detail .d-title").inner_text()
    assert "off the map" in page.locator("#detail .d-sub").inner_text()
    page.locator("#d-add-hop").click()
    page.wait_for_timeout(200)
    assert page.locator("#ep-jname").input_value() == hop_name      # join form prefilled
    assert page.locator("#edit-panel .ep-papers li").count() > 0    # rooms drafted as chapters
    print(f"hop detail + add-to-map CTA OK ({hop_name})")

    # guest finder: search anyone -> temporary OpenAlex-backed node wired into the graph
    import json as _json
    AUTHOR = {"id": "https://openalex.org/A999", "display_name": "Testa Guestperson",
              "affiliations": [
                  {"institution": {"display_name": "Massachusetts Institute of Technology",
                                   "type": "education"}, "years": [2021, 2022]},
                  {"institution": {"display_name": "Totally New Startup (United States)",
                                   "type": "company"}, "years": [2024]}]}
    cors = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}
    page.route("**/autocomplete/authors*", lambda r: r.fulfill(headers=cors, body=_json.dumps(
        {"results": [{"id": AUTHOR["id"], "display_name": AUTHOR["display_name"],
                      "hint": "Test University", "works_count": 5}]})))
    page.route("**/api.openalex.org/authors/A999*",
               lambda r: r.fulfill(headers=cors, body=_json.dumps(AUTHOR)))
    page.goto(f"http://127.0.0.1:{PORT}/networks/affiliations/")
    page.wait_for_selector("#people li", timeout=15000)
    page.wait_for_timeout(500)
    n_before = page.locator("#graph g.person").count()
    page.locator("#finder-q").fill("testa guest")
    page.wait_for_selector("#finder-results .fr", timeout=8000)
    assert "OpenAlex" in page.locator("#finder-results .fr").first.inner_text()
    page.locator("#finder-results .fr").first.click()
    page.wait_for_timeout(800)
    assert page.locator("#graph g.person").count() == n_before + 1
    assert "guest preview" in page.locator("#detail").inner_text()
    assert "MIT" in page.locator("#detail").inner_text()            # mapped onto the existing org
    assert "Totally New Startup" in page.locator("#detail").inner_text()  # ghost org listed
    page.locator("#d-keep-guest").click()                           # add-for-real prefills the join
    page.wait_for_timeout(300)
    assert page.locator("#ep-jname").input_value() == "Testa Guestperson"
    assert page.locator("#edit-panel .ep-papers li").count() == 2
    assert page.locator("#graph g.person").count() == n_before      # guest node handed off
    print("guest finder OK (search → temp node → add-for-real prefill)")
    page.unroute("**/autocomplete/authors*")
    page.unroute("**/api.openalex.org/authors/A999*")

    # extended-graph index: a known outside coauthor places instantly (live author mocked)
    CONMY = {"id": "https://openalex.org/A5071070679", "display_name": "Arthur Conmy",
             "affiliations": [{"institution": {"display_name": "University of Cambridge",
                                               "type": "education"}, "years": [2022]}]}
    page.route("**/autocomplete/authors*", lambda r: r.fulfill(headers=cors, body='{"results": []}'))
    page.route("**/api.openalex.org/authors/A5071070679*",
               lambda r: r.fulfill(headers=cors, body=_json.dumps(CONMY)))
    page.goto(f"http://127.0.0.1:{PORT}/networks/affiliations/")
    page.wait_for_selector("#people li", timeout=15000)
    page.wait_for_timeout(500)
    page.locator("#finder-q").fill("arthur conmy")
    page.wait_for_selector("#finder-results .fr", timeout=8000)
    row = page.locator("#finder-results .fr", has_text="w/ the group").first
    assert "Arthur Conmy" in row.inner_text()
    row.click()
    page.wait_for_timeout(800)
    assert "guest preview" in page.locator("#detail").inner_text()
    assert "University of Cambridge" in page.locator("#detail").inner_text()
    print("extended-graph index OK (Conmy placed from index + live record)")
    page.unroute("**/autocomplete/authors*")
    page.unroute("**/api.openalex.org/authors/A5071070679*")

    # papers map finder: outside coauthors already drawn are findable; click reveals + selects
    page.goto(f"http://127.0.0.1:{PORT}/networks/")
    page.wait_for_selector("#people li", timeout=15000)
    page.wait_for_timeout(800)
    page.locator("#finder-q").fill("arthur conmy")
    page.wait_for_selector("#finder-results .fr", timeout=8000)
    prow = page.locator("#finder-results .fr", has_text="coauthor").first
    assert "Arthur Conmy" in prow.inner_text()
    prow.click()
    page.wait_for_timeout(600)
    assert int(page.locator("#hops").input_value()) >= 2   # slider bumped to reveal him
    print("papers-map finder OK (Conmy revealed via hop bump)")

    # clean person URL: static stub forwards to the seat; old URLs redirect
    page.goto(f"http://127.0.0.1:{PORT}/networks/can-rager/")
    page.wait_for_selector("#panel-your-seat.active .m-viz svg", timeout=20000)
    assert "Can Rager" in page.locator("#panel-your-seat .m-viz h3").inner_text()
    legacy = (SITE / "coauthorship" / "affiliations" / "index.html").read_text()
    assert "/networks/affiliations/" in legacy   # jekyll-redirect-from page points at the new home
    print("person URL + legacy redirect OK")

    assert not errors, errors
    print("no JS errors — nav e2e PASSED")
    browser.close()
