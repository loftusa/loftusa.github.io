"""client_ip must honor Fly-Client-IP: on Fly the TCP peer is the edge proxy, so without
the header every visitor would share one rate-limit bucket."""
from __future__ import annotations

from backend.app.deps import client_ip
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

_app = FastAPI()


@_app.get("/ip")
def whoami(request: Request) -> dict:
    return {"ip": client_ip(request)}


client = TestClient(_app)


def test_fly_client_ip_header_wins():
    r = client.get("/ip", headers={"Fly-Client-IP": "203.0.113.9"})
    assert r.json()["ip"] == "203.0.113.9"


def test_falls_back_to_socket_peer_without_header():
    assert client.get("/ip").json()["ip"] == "testclient"
