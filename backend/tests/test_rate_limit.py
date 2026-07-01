"""DB-backed sliding-window rate limiter. The whole point of moving off the in-memory
`_correction_hits` dict is that limits survive a Fly restart/deploy — so durability across a
fresh session is a first-class test here. `now` is injectable for deterministic window tests.
"""
from backend.app.db import SessionLocal
from backend.app.services import rate_limit


def test_blocks_after_limit(session):
    for _ in range(3):
        assert rate_limit.rate_ok(session, "1.1.1.1", limit=3, window=600, now=1000.0)
    assert not rate_limit.rate_ok(session, "1.1.1.1", limit=3, window=600, now=1000.0)


def test_rejection_does_not_consume_a_slot(session):
    # mirror the old limiter: a rejected request must not itself count as a hit
    for _ in range(3):
        rate_limit.rate_ok(session, "ip", limit=3, window=600, now=1000.0)
    rate_limit.rate_ok(session, "ip", limit=3, window=600, now=1000.0)  # rejected
    # exactly 3 hits stored, not 4
    assert rate_limit.hit_count(session, "ip", window=600, now=1000.0) == 3


def test_per_ip_isolation(session):
    for _ in range(3):
        rate_limit.rate_ok(session, "a", limit=3, window=600, now=1000.0)
    assert rate_limit.rate_ok(session, "b", limit=3, window=600, now=1000.0)


def test_window_expiry_frees_slots(session):
    assert rate_limit.rate_ok(session, "ip", limit=1, window=100, now=1000.0)
    assert not rate_limit.rate_ok(
        session, "ip", limit=1, window=100, now=1050.0
    )  # still in window
    assert rate_limit.rate_ok(
        session, "ip", limit=1, window=100, now=1200.0
    )  # window passed


def test_survives_a_fresh_session(session):
    # durability: hits written in one session still count after a "restart" (new session, same DB)
    s1 = SessionLocal()
    for _ in range(3):
        rate_limit.rate_ok(s1, "ip", limit=3, window=600, now=1000.0)
    s1.close()
    s2 = SessionLocal()
    try:
        assert not rate_limit.rate_ok(s2, "ip", limit=3, window=600, now=1000.0)
    finally:
        s2.close()
