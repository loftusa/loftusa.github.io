"""Endpoint tests for the co-authorship correction API (experiments/chat_api.py).

Run:  uv run --no-sync pytest experiments/tests/test_corrections_api.py -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `experiments` imports

# point the log/corrections volume at a temp dir BEFORE importing the app (paths bind at import)
_TMP = Path(tempfile.mkdtemp())
os.environ["LOG_PATH"] = str(_TMP / "chat_logs.jsonl")
os.environ["LOG_ACCESS_TOKEN"] = "test-token"

from fastapi.testclient import TestClient  # noqa: E402
import experiments.chat_api as api  # noqa: E402

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def clean_log():
    if api.CORRECTIONS_PATH.exists():
        api.CORRECTIONS_PATH.unlink()
    api._correction_hits.clear()
    yield


def post(body):
    return client.post("/coauthorship/corrections", json=body)


def test_valid_correction_is_appended():
    r = post({"type": "node_label", "payload": {"id": "bob", "label": "Bob X"}})
    assert r.status_code == 200 and r.json()["ok"]
    lines = api.CORRECTIONS_PATH.read_text().splitlines()
    assert len(lines) == 1 and '"node_label"' in lines[0] and '"ts"' in lines[0]


def test_unknown_type_rejected():
    r = post({"type": "drop_database", "payload": {}})
    assert r.status_code == 422  # not in the Literal set


def test_missing_payload_key_rejected():
    r = post({"type": "node_label", "payload": {"id": "bob"}})  # no "label"
    assert r.status_code == 422
    assert "label" in r.json()["detail"]


def test_bad_between_rejected():
    r = post({"type": "remove_edge", "payload": {"between": ["only-one"]}})
    assert r.status_code == 422


def test_overlay_reflects_posted_events():
    post({"type": "node_label", "payload": {"id": "Bob Smith", "label": "Bob Q. Smith"}})
    post({"type": "remove_edge", "payload": {"between": ["alice", "bob"]}})
    overlay = client.get("/coauthorship/overlay").json()
    assert overlay["node_label"]["bob smith"] == "Bob Q. Smith"  # id normalized
    assert overlay["remove_edges"] == [["alice", "bob"]]


def test_rate_limit_returns_429():
    limit, _ = api.CORRECTION_RATE
    for _ in range(limit):
        assert post({"type": "remove_node", "payload": {"id": "x"}}).status_code == 200
    assert post({"type": "remove_node", "payload": {"id": "x"}}).status_code == 429


def test_export_requires_bearer():
    post({"type": "node_label", "payload": {"id": "bob", "label": "B"}})
    assert client.get("/coauthorship/corrections").status_code == 401
    assert client.get("/coauthorship/corrections",
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/coauthorship/corrections",
                    headers={"Authorization": "Bearer test-token"})
    assert ok.status_code == 200 and '"node_label"' in ok.text


def test_overlay_is_open():
    assert client.get("/coauthorship/overlay").status_code == 200


AUTH = {"Authorization": "Bearer test-token"}


def test_delete_requires_bearer():
    post({"type": "node_label", "payload": {"id": "bob", "label": "B"}})
    ts = json.loads(api.CORRECTIONS_PATH.read_text().splitlines()[0])["ts"]
    assert client.delete(f"/coauthorship/corrections?ts={ts}").status_code == 401
    assert client.delete(f"/coauthorship/corrections?ts={ts}",
                         headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_delete_reverts_event_durably():
    post({"type": "node_label", "payload": {"id": "bob", "label": "Vandal Edit"}})
    post({"type": "remove_edge", "payload": {"between": ["alice", "bob"]}})
    lines = [json.loads(l) for l in api.CORRECTIONS_PATH.read_text().splitlines()]
    vandal_ts = next(e["ts"] for e in lines if e["type"] == "node_label")

    r = client.delete(f"/coauthorship/corrections?ts={vandal_ts}", headers=AUTH)
    assert r.status_code == 200 and r.json()["removed"] == 1 and r.json()["remaining"] == 1

    overlay = client.get("/coauthorship/overlay").json()
    assert overlay["node_label"] == {}                       # vandal edit gone from the live overlay
    assert overlay["remove_edges"] == [["alice", "bob"]]     # unrelated event untouched


def test_delete_unknown_ts_404():
    post({"type": "node_label", "payload": {"id": "bob", "label": "B"}})
    r = client.delete("/coauthorship/corrections?ts=1999-01-01T00:00:00", headers=AUTH)
    assert r.status_code == 404
