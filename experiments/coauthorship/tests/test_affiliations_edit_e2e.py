# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.55.0"]
# ///
"""Headless E2E of the affiliation self-service frontend: overlay applied at load, edit/join
flows POST the right events with optimistic re-render, identity picker + info check.

The API is MOCKED with page.route (no live Fly endpoints needed): overlay requests are
fulfilled with a fixture, POSTs are captured and acknowledged.

Prereq: `bundle exec jekyll build`.  Run: cd experiments/coauthorship && uv run tests/test_affiliations_edit_e2e.py
"""
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[3]
SITE = REPO / "_site"
PORT = 4129
assert (SITE / "networks" / "affiliations" / "index.html").exists(), "run jekyll build first"

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

OVERLAY = {  # one pending joiner + one pending edit, as the live API would fold them
    "version": 1,
    "join": {"pat tester": {"name": "Pat Tester", "city": "Lisbon", "scholar_url": None,
                            "homepage": None, "ts": "2026-06-11T00:00:00"}},
    "entry_set": {"pat tester": {"eleutherai": {"org": "EleutherAI", "type": "community",
                                                "role": "contributor", "years": "2025–",
                                                "current": True, "source": "",
                                                "ts": "2026-06-11T00:00:00"}}},
    "entry_remove": {}, "city": {}, "confirmed": {},
}

with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    posted = []
    page.route("**/affiliations/overlay",
               lambda r: r.fulfill(json=OVERLAY, headers={"Access-Control-Allow-Origin": "*"}))
    page.route("**/affiliations/corrections",
               lambda r: (posted.append(json.loads(r.request.post_data)),
                          r.fulfill(json={"ok": True},
                                    headers={"Access-Control-Allow-Origin": "*"})))
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # ---- overlay applied at load: joiner node present (grey), org membership bumped ----
    page.add_init_script("localStorage.setItem('network_identity_dismissed', '1')")
    page.goto(f"http://127.0.0.1:{PORT}/networks/affiliations/")
    page.wait_for_selector("#graph svg g.person", timeout=15000)
    page.wait_for_timeout(1500)
    assert page.locator("#graph g.person").count() == 53, "joiner not minted"
    # 12 no-paper people are community -1 (beige by design — colorOf guard); +1 = the joiner
    grey = page.eval_on_selector_all("#graph g.person circle",
        "els => els.filter(e => e.getAttribute('fill') === '#b3a98f').length")
    assert grey == 13, ("12 community:-1 people + the joiner", grey)
    black = page.eval_on_selector_all("#graph g.person circle",
        "els => els.filter(e => !e.getAttribute('fill')).length")
    assert black == 0, "undefined fills are back (PALETTE[-1] regression)"
    page.locator("#people li", has_text="Pat Tester").click()
    detail = page.locator("#detail").inner_text()
    assert "EleutherAI" in detail and "Lisbon" in detail
    print("overlay at load OK (53 people, joiner grey, joiner entry visible)")

    # ---- edit flow: add an entry -> correct POST + optimistic render, no reload ----
    page.locator("#edit-toggle").click()
    page.locator("#ep-who").fill("Ted Kyi")
    page.locator("#ep-who-go").click()
    page.wait_for_selector("#ep-org", timeout=5000)
    page.locator("#ep-org").fill("Test Startup XYZ")
    page.locator("#ep-type").select_option("company")
    page.locator("#ep-role").fill("founder")
    page.locator("#ep-years").fill("2026–")
    page.locator("#ep-current").check()
    page.locator("#ep-add").click()
    page.wait_for_timeout(600)
    assert posted and posted[-1]["type"] == "aff_entry_set"
    assert posted[-1]["payload"]["person"] == "ted kyi"
    assert posted[-1]["payload"]["org"] == "Test Startup XYZ"
    assert posted[-1]["payload"]["current"] is True
    # optimistic: the new org node is in the SVG without any page reload
    page.locator("#singles").check()
    page.wait_for_timeout(800)
    labels = page.eval_on_selector_all("#graph text.olabel", "els => els.map(e => e.textContent)")
    assert "Test Startup XYZ" in labels, "optimistic org node missing"
    print("edit flow OK (aff_entry_set posted, node re-wired live)")

    # ---- remove flow ----
    page.on("dialog", lambda d: d.accept())
    page.locator("#edit-panel [data-remove='Test Startup XYZ']").click()
    page.wait_for_timeout(500)
    assert posted[-1]["type"] == "aff_entry_remove"
    print("remove flow OK (aff_entry_remove posted)")

    # ---- join flow ----
    page.locator("#ep-back").click()
    page.locator("#ep-join").click()
    page.locator("#ep-jname").fill("Totally New Member")
    page.locator("#ep-jcity").fill("Porto")
    page.locator("#epj-org").fill("MATS")
    page.locator("#epj-add").click()
    page.locator("#ep-join-save").click()
    page.wait_for_timeout(600)
    join = posted[-1]
    assert join["type"] == "aff_join" and join["payload"]["name"] == "Totally New Member"
    assert join["payload"]["entries"][0]["org"] == "MATS"
    assert join["payload"]["entries"][0]["type"] == "program"   # datalist match locked the type
    assert page.locator("#graph g.person").count() == 54
    print("join flow OK (aff_join posted, temp node minted)")

    # ---- papers map: pending joiner minted instantly as a hollow no-papers node ----
    page.goto(f"http://127.0.0.1:{PORT}/networks/")
    page.wait_for_selector("#people li", timeout=15000)
    page.wait_for_timeout(1500)
    assert page.locator("#people li", has_text="Pat Tester").count() == 1, \
        "pending joiner missing from the papers map"
    print("papers map mints pending joiner instantly")

    # ---- identity picker: first visit -> pick -> info check -> confirm ----
    ctx2 = browser.new_context(viewport={"width": 1440, "height": 900})
    page2 = ctx2.new_page()
    posted2 = []
    page2.route("**/affiliations/overlay",
                lambda r: r.fulfill(json=OVERLAY, headers={"Access-Control-Allow-Origin": "*"}))
    page2.route("**/affiliations/corrections",
                lambda r: (posted2.append(json.loads(r.request.post_data)),
                           r.fulfill(json={"ok": True},
                                     headers={"Access-Control-Allow-Origin": "*"})))
    page2.on("pageerror", lambda e: errors.append(str(e)))
    page2.goto(f"http://127.0.0.1:{PORT}/networks/")
    page2.wait_for_selector(".nid-card", timeout=15000)
    page2.locator(".nid-card input").fill("kayo")
    page2.locator(".nid-card li", has_text="Kayo Yin").click()
    page2.wait_for_timeout(1000)
    card = page2.locator(".nid-card").inner_text()
    assert "Kayo Yin" in card and ("Berkeley" in card or "UC Berkeley" in card), card[:200]
    page2.get_by_text("looks right").click()
    page2.wait_for_timeout(500)
    assert posted2 and posted2[-1]["type"] == "aff_confirm"
    assert posted2[-1]["payload"]["person"] == "kayo yin"
    ident = page2.evaluate("localStorage.getItem('network_identity')")
    assert json.loads(ident)["id"] == "kayo yin"
    assert "you are" in page2.locator("#foot").inner_text().lower()
    # personalized default: affiliations analyses with no hash -> your-seat?p=
    page2.goto(f"http://127.0.0.1:{PORT}/networks/affiliations/analyses/")
    page2.wait_for_selector("#panel-your-seat.active .m-viz svg", timeout=20000)
    assert "Kayo Yin" in page2.locator("#panel-your-seat .m-viz h3").inner_text()
    print("identity flow OK (picker -> info check -> aff_confirm -> personalized default)")

    # dismissed path: fresh context, "just browsing" -> no picker on next load
    ctx3 = browser.new_context()
    page3 = ctx3.new_page()
    page3.goto(f"http://127.0.0.1:{PORT}/networks/")
    page3.wait_for_selector(".nid-card", timeout=15000)
    page3.get_by_text("just browsing").click()
    page3.goto(f"http://127.0.0.1:{PORT}/networks/")
    page3.wait_for_timeout(1200)
    assert page3.locator(".nid-card").count() == 0
    print("dismissed path OK")

    assert not errors, errors
    print("ALL EDIT/IDENTITY E2E PASSED")
    browser.close()
