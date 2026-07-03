"""rate.py pure parts: audience grouping + gio listing text.

rate.py imports anthropic + PIL at module level; skip cleanly where absent.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("PIL")

_RATE = (
    Path(__file__).resolve().parents[2] / "public" / "houses" / "refresh" / "rate.py"
)
_spec = importlib.util.spec_from_file_location("houses_rate", _RATE)
rate = importlib.util.module_from_spec(_spec)
sys.modules["houses_rate"] = rate
_spec.loader.exec_module(rate)


def _row(lid, aud):
    return {
        "id": lid,
        "aud": aud,
        "price": 2000,
        "hood": "mission bay",
        "region": "SF",
        "bucket": "room",
        "title": "t",
        "body": "",
        "walk_min": 9,
    }


def test_group_batches_never_mixes_audiences():
    sel = [_row(f"L{i:02d}", "alex") for i in range(1, 8)]
    sel += [_row(f"G{i:02d}", "gio") for i in range(1, 4)]
    batches = rate.group_batches(sel, 5)
    assert [(aud, len(b)) for aud, b in batches] == [
        ("alex", 5),
        ("alex", 2),
        ("gio", 3),
    ]
    for aud, b in batches:
        assert all(r.get("aud", "alex") == aud for r in b)
    got = [r["id"] for _, b in batches for r in b]
    assert got == [r["id"] for r in sel]  # nothing dropped, stable order


def test_rubrics_exist_and_differ():
    assert set(rate.RUBRICS) == {"alex", "gio"}
    assert "OpenAI" in rate.RUBRICS["gio"] and "FAR Labs" not in rate.RUBRICS["gio"]
    assert "networking" in rate.RUBRICS["alex"]


def test_listing_text_gio_walk_line():
    txt = rate.listing_text(_row("G01", "gio"))
    assert "9 min walk to OpenAI HQ" in txt
    txt_alex = rate.listing_text(_row("L01", "alex"))
    assert "walk to OpenAI" not in txt_alex and "(SF)" in txt_alex
