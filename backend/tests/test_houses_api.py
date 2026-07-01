from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

URL = "https://sfbay.craigslist.org/nby/apa/d/sausalito-nice/7890123456.html"


def test_empty_list():
    assert client.get("/houses/reached-out").json() == []


def test_upsert_and_list():
    r = client.post(
        "/houses/reached-out",
        json={
            "url": URL,
            "title": "Sausalito 1BR",
            "message": "Hi!",
            "channel": "email",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["url"] == URL and data["title"] == "Sausalito 1BR"
    assert data["channel"] == "email"

    rows = client.get("/houses/reached-out").json()
    assert len(rows) == 1 and rows[0]["url"] == URL


def test_upsert_updates_fields():
    client.post("/houses/reached-out", json={"url": URL, "message": "first"})
    client.post("/houses/reached-out", json={"url": URL, "message": "second"})
    rows = client.get("/houses/reached-out").json()
    assert len(rows) == 1 and rows[0]["message"] == "second"


def test_delete():
    client.post("/houses/reached-out", json={"url": URL})
    r = client.delete("/houses/reached-out", params={"url": URL})
    assert r.status_code == 200 and r.json()["ok"]
    assert client.get("/houses/reached-out").json() == []


def test_delete_not_found():
    r = client.delete("/houses/reached-out", params={"url": "https://nope"})
    assert r.status_code == 404


def test_missing_url_rejected():
    r = client.post("/houses/reached-out", json={"message": "no url"})
    assert r.status_code == 422
