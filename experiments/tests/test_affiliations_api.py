"""Endpoint tests for the affiliation self-service API (experiments/chat_api.py).

Run:  uv run --no-sync pytest experiments/tests/test_affiliations_api.py -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `experiments` imports

# point the log volume at a temp dir BEFORE importing the app (paths bind at import)
_TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("LOG_PATH", str(_TMP / "chat_logs.jsonl"))
os.environ.setdefault("LOG_ACCESS_TOKEN", "test-token")

from fastapi.testclient import TestClient  # noqa: E402
import experiments.chat_api as api  # noqa: E402

client = TestClient(api.app)
AUTH = {"Authorization": f"Bearer {os.environ['LOG_ACCESS_TOKEN']}"}


@pytest.fixture(autouse=True)
def clean_log():
    if api.AFF_EVENTS_PATH.exists():
        api.AFF_EVENTS_PATH.unlink()
    api._correction_hits.clear()
    yield


def post(body):
    return client.post("/affiliations/corrections", json=body)


def set_entry(person="can rager", org="Goodfire", typ="company", **kw):
    return post({"type": "aff_entry_set",
                 "payload": {"person": person, "org": org, "type": typ, **kw}})


def test_valid_entry_set_appended():
    r = set_entry(role="Research Engineer", years="2026–", current=True)
    assert r.status_code == 200 and r.json()["ok"]
    lines = api.AFF_EVENTS_PATH.read_text().splitlines()
    assert len(lines) == 1 and '"aff_entry_set"' in lines[0] and '"ts"' in lines[0]


def test_unknown_type_rejected():
    assert post({"type": "aff_drop_table", "payload": {}}).status_code == 422


def test_missing_keys_rejected():
    r = post({"type": "aff_entry_set", "payload": {"person": "can rager"}})
    assert r.status_code == 422 and "org" in r.json()["detail"]


def test_bad_entry_type_rejected():
    r = set_entry(typ="cult")
    assert r.status_code == 422 and "type" in r.json()["detail"]


def test_field_caps_rejected():
    assert set_entry(org="x" * 201).status_code == 422
    assert post({"type": "aff_city", "payload": {"person": "can rager", "city": "y" * 81}}).status_code == 422


def test_join_entry_count_capped():
    entries = [{"org": f"Org {i}", "type": "company"} for i in range(11)]
    r = post({"type": "aff_join", "payload": {"name": "New Person", "entries": entries}})
    assert r.status_code == 422


def test_join_collision_with_roster_409():
    r = post({"type": "aff_join", "payload": {"name": "Can RAGER", "entries": []}})
    assert r.status_code == 409


def test_join_collision_with_pending_join_409():
    assert post({"type": "aff_join", "payload": {"name": "Brand New"}}).status_code == 200
    assert post({"type": "aff_join", "payload": {"name": "brand  new"}}).status_code == 409


def test_confirm_and_city_roundtrip_via_overlay():
    assert post({"type": "aff_confirm", "payload": {"person": "can rager"}}).status_code == 200
    assert post({"type": "aff_city", "payload": {"person": "can rager", "city": "Boston"}}).status_code == 200
    ov = client.get("/affiliations/overlay").json()
    assert "can rager" in ov["confirmed"] and ov["city"]["can rager"] == "Boston"


def test_overlay_folds_set_then_remove():
    set_entry(org="Goodfire")
    post({"type": "aff_entry_remove", "payload": {"person": "can rager", "org": "goodfire"}})
    ov = client.get("/affiliations/overlay").json()
    assert "can rager" not in ov["entry_set"]
    assert ov["entry_remove"]["can rager"] == ["goodfire"]


def test_oversized_event_413():
    # valid capped fields can't exceed the byte cap; it guards against junk payload keys
    r = post({"type": "aff_city",
              "payload": {"person": "can rager", "city": "Boston", "junk": "A" * 8000}})
    assert r.status_code == 413


def test_export_requires_bearer():
    assert client.get("/affiliations/corrections").status_code == 401
    set_entry()
    r = client.get("/affiliations/corrections", headers=AUTH)
    assert r.status_code == 200 and '"aff_entry_set"' in r.text


def test_delete_reverts_durably():
    set_entry()
    ts = json.loads(api.AFF_EVENTS_PATH.read_text())["ts"]
    assert client.delete(f"/affiliations/corrections?ts={ts}").status_code == 401  # no bearer
    r = client.delete(f"/affiliations/corrections?ts={ts}", headers=AUTH)
    assert r.status_code == 200 and r.json()["removed"] == 1
    assert client.get("/affiliations/overlay").json()["entry_set"] == {}
    assert client.delete(f"/affiliations/corrections?ts={ts}", headers=AUTH).status_code == 404


def test_rate_limit_429():
    api.CORRECTION_RATE_SAVED = api.CORRECTION_RATE
    try:
        api.CORRECTION_RATE = (2, 600)
        assert set_entry().status_code == 200
        assert set_entry(org="Other Org").status_code == 200
        assert set_entry(org="Third Org").status_code == 429
    finally:
        api.CORRECTION_RATE = api.CORRECTION_RATE_SAVED
