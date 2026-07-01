"""Endpoint tests for the co-authorship correction API — ported from the pre-migration
experiments/tests/test_corrections_api.py. Every behavioral assertion is preserved; storage
introspection now reads the bearer raw-export endpoint instead of the JSONL file.
"""
import json

from backend.app import config
from backend.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-token"}


def post(body):
    return client.post("/coauthorship/corrections", json=body)


def raw_rows():
    r = client.get("/coauthorship/corrections", headers=AUTH)
    return [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]


def test_valid_correction_is_appended():
    r = post({"type": "node_label", "payload": {"id": "bob", "label": "Bob X"}})
    assert r.status_code == 200 and r.json()["ok"]
    rows = raw_rows()
    assert len(rows) == 1 and rows[0]["type"] == "node_label" and "ts" in rows[0]


def test_unknown_type_rejected():
    assert post({"type": "drop_database", "payload": {}}).status_code == 422


def test_missing_payload_key_rejected():
    r = post({"type": "node_label", "payload": {"id": "bob"}})
    assert r.status_code == 422 and "label" in r.json()["detail"]


def test_bad_between_rejected():
    assert (
        post({"type": "remove_edge", "payload": {"between": ["only-one"]}}).status_code
        == 422
    )


def test_overlay_reflects_posted_events():
    post(
        {"type": "node_label", "payload": {"id": "Bob Smith", "label": "Bob Q. Smith"}}
    )
    post({"type": "remove_edge", "payload": {"between": ["alice", "bob"]}})
    overlay = client.get("/coauthorship/overlay").json()
    assert overlay["node_label"]["bob smith"] == "Bob Q. Smith"
    assert overlay["remove_edges"] == [["alice", "bob"]]


def test_rate_limit_returns_429():
    limit, _ = config.CORRECTION_RATE
    for _ in range(limit):
        assert post({"type": "remove_node", "payload": {"id": "x"}}).status_code == 200
    assert post({"type": "remove_node", "payload": {"id": "x"}}).status_code == 429


def test_export_requires_bearer():
    post({"type": "node_label", "payload": {"id": "bob", "label": "B"}})
    assert client.get("/coauthorship/corrections").status_code == 401
    assert (
        client.get(
            "/coauthorship/corrections", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    ok = client.get("/coauthorship/corrections", headers=AUTH)
    assert ok.status_code == 200 and '"node_label"' in ok.text


def test_overlay_is_open():
    assert client.get("/coauthorship/overlay").status_code == 200


def test_delete_requires_bearer():
    post({"type": "node_label", "payload": {"id": "bob", "label": "B"}})
    ts = raw_rows()[0]["ts"]
    assert client.delete(f"/coauthorship/corrections?ts={ts}").status_code == 401
    assert (
        client.delete(
            f"/coauthorship/corrections?ts={ts}",
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 401
    )


def test_delete_reverts_event_durably():
    post({"type": "node_label", "payload": {"id": "bob", "label": "Vandal Edit"}})
    post({"type": "remove_edge", "payload": {"between": ["alice", "bob"]}})
    vandal_ts = next(e["ts"] for e in raw_rows() if e["type"] == "node_label")

    r = client.delete(f"/coauthorship/corrections?ts={vandal_ts}", headers=AUTH)
    assert (
        r.status_code == 200 and r.json()["removed"] == 1 and r.json()["remaining"] == 1
    )

    overlay = client.get("/coauthorship/overlay").json()
    assert overlay["node_label"] == {}
    assert overlay["remove_edges"] == [["alice", "bob"]]


def test_delete_unknown_ts_404():
    post({"type": "node_label", "payload": {"id": "bob", "label": "B"}})
    r = client.delete("/coauthorship/corrections?ts=1999-01-01T00:00:00", headers=AUTH)
    assert r.status_code == 404
