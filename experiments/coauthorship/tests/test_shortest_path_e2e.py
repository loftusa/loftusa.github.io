# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.55.0"]
# ///
"""Headless E2E test of the shortest-path finder panel on /networks/.

Prereq: `bundle exec jekyll build` from the repo root (tests serve _site/ locally).
Run:    uv run test_shortest_path_e2e.py

Covers: slot arming/filling (click and type-ahead), 1-hop pair, 6-hop pair (connectors
force-revealed past the Reach slider), warm-ink chain, isolated-person pair via bridge
edges (must match what clicking that person shows), disconnected pair message, ✕ clear,
hop-slider persistence, and the dim-vs-fade-in race regression (highlightRoute must
interrupt applyVisibility's 400ms fade or every node tweens back to opacity 1).
"""
import functools
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[3]
SITE = REPO / "_site"
PORT = 4123
INK = "rgb(90, 83, 70)"  # #5a5346, the pair-chain stroke
OPACITY_HISTO = """els => {
  const h = {};
  els.filter(e => e.style.display !== 'none').forEach(e => {
    const o = (+getComputedStyle(e).opacity).toFixed(2);
    h[o] = (h[o] || 0) + 1;
  });
  return h;
}"""

assert (SITE / "networks").exists(), "run `bundle exec jekyll build` first"

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()


def launch(pw):
    try:
        return pw.chromium.launch()
    except Exception:  # version-mismatched cache: use whatever chromium build is present
        exe = next(Path.home().glob(
            "Library/Caches/ms-playwright/chromium-*/chrome-mac*/"
            "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"))
        return pw.chromium.launch(executable_path=str(exe))


def inked_count(page):
    return page.eval_on_selector_all(
        "#graph line:not(.hit)", f"els => els.filter(e => e.style.stroke === '{INK}').length")


with sync_playwright() as pw:
    browser = launch(pw)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    # the identity picker greets first visits; its scrim would swallow every click here
    page.add_init_script("localStorage.setItem('network_identity_dismissed', '1')")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{PORT}/networks/", wait_until="networkidle")
    page.wait_for_selector("#people li", timeout=15000)
    person = lambda name: page.locator("#people li", has_text=name).first

    # panel renders with two empty type-ahead slots, clear hidden
    slots = page.locator("#pathctl .pp-slot")
    assert slots.count() == 2, "expected two slots"
    assert slots.nth(0).locator("input.pp-input").count() == 1, "empty slot missing its input"
    assert page.locator("#pathctl .pp-clear").evaluate("el => getComputedStyle(el).visibility") == "hidden"

    # 1-hop pair: arm 'from' (focus its input), fill both by clicking people in the list
    slots.nth(0).click()
    assert "armed" in (slots.nth(0).get_attribute("class") or ""), "from slot did not arm"
    person("Loftus").click()
    assert "Loftus" in slots.nth(0).inner_text(), "from slot not filled"
    assert "armed" in (slots.nth(1).get_attribute("class") or ""), "to slot should auto-arm"
    person("Can Rager").click()
    page.wait_for_timeout(800)
    detail = page.locator("#path-detail").inner_text()
    assert "hop" in detail and "Loftus" in detail and "Can Rager" in detail, f"bad detail: {detail!r}"
    assert inked_count(page) >= 1, "no path edges painted in warm ink"

    # 6-hop pair: connectors hidden at hop 1 must be force-revealed
    slots.nth(0).click()
    person("Ronak Mehta").click()
    slots.nth(1).click()
    person("Alice Rigg").click()
    page.wait_for_timeout(800)
    detail = page.locator("#path-detail").inner_text()
    assert "6 hops" in detail, f"expected 6 hops, got: {detail!r}"
    assert inked_count(page) == 6, f"expected 6 inked edges, got {inked_count(page)}"

    # hop slider change must keep the pair path alive
    page.locator("#hops").fill("2")
    page.locator("#hops").dispatch_event("input")
    page.wait_for_timeout(800)
    assert "6 hops" in page.locator("#path-detail").inner_text(), "path lost after hop change"
    assert inked_count(page) == 6, "ink lost after hop change"

    # regression: dim must persist past applyVisibility's 400ms fade-in
    page.wait_for_timeout(600)
    histo = page.eval_on_selector_all("#graph g.node", OPACITY_HISTO)
    assert histo.get("1.00") == 7, f"chain nodes not lit: {histo}"
    assert histo.get("0.12", 0) >= 40, f"dim lost to fade-in transition: {histo}"

    # type-ahead: re-open 'from' (click clears a filled slot), type, pick from the menu
    slots.nth(0).click()
    box = slots.nth(0).locator("input.pp-input")
    box.fill("ryan chesler")
    opt = slots.nth(0).locator(".pp-menu .pp-opt")
    assert "Chesler" in opt.first.inner_text(), "type-ahead suggestion missing"
    opt.first.dispatch_event("mousedown")
    assert "Chesler" in slots.nth(0).inner_text(), "type-ahead did not fill the slot"
    slots.nth(1).click()
    slots.nth(1).locator("input.pp-input").fill("biderman")
    slots.nth(1).locator("input.pp-input").press("Enter")
    page.wait_for_timeout(800)
    pair_detail = page.locator("#path-detail").inner_text()
    assert "6 hops" in pair_detail, f"isolated-pair route missing: {pair_detail!r}"

    # …and it must agree with what clicking the isolated person shows (the bridge route)
    page.locator("#pathctl .pp-clear").click()
    person("Ryan Chesler").click()
    page.wait_for_timeout(800)
    click_detail = page.locator("#path-detail").inner_text()
    assert "6 hops to Stella Biderman" in click_detail, f"bridge route: {click_detail!r}"
    assert pair_detail.splitlines()[1:] == click_detail.splitlines()[1:], (
        f"pair chain != click chain:\n  {pair_detail!r}\n  {click_detail!r}")
    person("Ryan Chesler").click()  # deselect

    # disconnected pair: a no-indexed-papers person honestly reports no path (typed —
    # such people aren't clickable in the list, but they are valid typed endpoints)
    slots.nth(0).locator("input.pp-input").fill("daniel brown")
    slots.nth(0).locator("input.pp-input").press("Enter")
    slots.nth(1).locator("input.pp-input").fill("can rager")
    slots.nth(1).locator("input.pp-input").press("Enter")
    page.wait_for_timeout(400)
    assert "No co-authorship path" in page.locator("#path-detail").inner_text()

    # clear restores everything; normal selection still works
    page.locator("#pathctl .pp-clear").click()
    page.wait_for_timeout(400)
    assert slots.nth(0).locator("input.pp-input").input_value() == "", "clear did not reset slots"
    assert page.locator("#path-detail").inner_text().strip() == "", "detail box not cleared"
    assert inked_count(page) == 0, "warm ink not wiped after clear"
    person("Can Rager").click()
    page.wait_for_timeout(300)
    assert page.locator("#people li.sel").count() == 1, "normal selection broken"

    assert not errors, f"JS page errors: {errors}"
    browser.close()

srv.shutdown()
print("E2E-OK")
