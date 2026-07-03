# /// script
# dependencies = ["cryptography>=42.0"]
# ///
"""Build a password-gated personal job board at alex-loftus.com/jobs/u/<slug>/.

One customer = one slug = one directory public/jobs/u/<slug>/ containing:
  data.enc.js   window.JOBS_ENC = {...}  — PBKDF2-SHA256(210k) + AES-256-GCM
                blob of a payload shaped exactly like the main data.js payload
                ({meta, jobs, bank?}), scored against THEIR profile.
  index.html    user_template.html with __SLUG__ filled in — the same board UI
                as /jobs but behind a client-side password gate, noindexed.

Usage:
  uv run make_user.py --slug jane --profile users/jane/profile.md \\
      --password SECRET [--prep-bank users/jane/prep_bank.json]
  # test/re-emit path, no API calls:
  uv run make_user.py --slug jane --profile users/jane/profile.md \\
      --password SECRET --skip-rate --ratings users/jane.ratings.json

WHERE THE FULL JDs COME FROM (design choice, verified 2026-07-03):
  The daily pipeline's artifacts do NOT contain the full board with JD text:
    - live_index.json  = every live role's metadata, but NO jd_text.
    - shortlist.json   = jd_text included, but only for roles that needed
                         scoring on the last daily pull (new/changed roles) —
                         on a normal day this is NOT the whole board.
    - data.js payload  = no jd_text either.
  A NEW user needs the ENTIRE current board scored, so stitching those files
  together can't work in general. Instead make_user.py calls
  refresh.pull_boards() itself: one fresh, internally consistent snapshot of
  all boards WITH JDs. It writes that snapshot as a temp shortlist and points
  rate.py at it (rate.py's --shortlist/--profile/--prep-bank/--out flags exist
  for exactly this). Caveat: pull_boards() applies refresh.py's BOARDS list
  and PREFILTER keyword lane to non-Anthropic boards — v1 customers get the
  same universe of roles as the main board.

  With --skip-rate no network pull happens: role metadata comes from
  live_index.json and scores from --ratings. This is the test path and the
  cheap re-emit path (e.g. password rotation).

The crypto helpers are copied verbatim from the b013610 gated refresh.py so
this script is self-contained; the page decrypts with WebCrypto
(PBKDF2-SHA256, 210k iterations, AES-256-GCM).
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refresh  # noqa: E402  — pull_boards, fit_score, FIT_W, MIN_FIT_NON_ANTHROPIC

TODAY = date.today().isoformat()
U_DIR = os.path.normpath(os.path.join(HERE, "..", "u"))
TEMPLATE = os.path.join(HERE, "user_template.html")
USERS_DIR = os.path.join(HERE, "users")
SHORTLIST = os.path.join(HERE, "shortlist.json")
LIVE_INDEX = os.path.join(HERE, "live_index.json")
PULL_STATS = os.path.join(HERE, "pull_stats.json")
RATE_PY = os.path.join(HERE, "rate.py")

# meta fields carried from a pulled role into the payload (same set as
# refresh.py's live_index)
META_KEYS = (
    "company",
    "group",
    "title",
    "url",
    "locations",
    "remote",
    "comp",
    "published",
)


# ---------------------------------------------------------------- crypto ----
# Copied from b013610:public/jobs/refresh/refresh.py — keep in sync with the
# WebCrypto decrypt in user_template.html.
def _derive(password: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    return PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=210_000
    ).derive(password.encode())


def encrypt_blob(obj, password: str) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt, iv = os.urandom(16), os.urandom(12)
    ct = AESGCM(_derive(password, salt)).encrypt(
        iv, json.dumps(obj, separators=(",", ":")).encode(), None
    )
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return {
        "v": 1,
        "kdf": "PBKDF2-SHA256",
        "iter": 210_000,
        "salt": b64(salt),
        "iv": b64(iv),
        "ct": b64(ct),
    }


def decrypt_blob(blob: dict, password: str):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    b64 = base64.b64decode
    pt = AESGCM(_derive(password, b64(blob["salt"]))).decrypt(
        b64(blob["iv"]), b64(blob["ct"]), None
    )
    return json.loads(pt)


# ---------------------------------------------------------------- helpers ---
def ensure_users_gitignore():
    """users/ holds customer profiles + prep banks — private, never committed."""
    os.makedirs(USERS_DIR, exist_ok=True)
    gi = os.path.join(USERS_DIR, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w") as f:
            f.write("*\n")


def assert_fresh_pull():
    for p in (SHORTLIST, LIVE_INDEX):
        if not os.path.exists(p):
            sys.exit(
                f"missing {p}\n"
                "Run a fresh pull first:\n"
                f"  cd {HERE} && python3 refresh.py --pull"
            )
    if os.path.exists(PULL_STATS):
        stats = json.load(open(PULL_STATS))
        if stats.get("_date") != TODAY:
            print(
                f"WARNING: last pull was {stats.get('_date')} (today is {TODAY}) — "
                "consider re-running refresh.py --pull for a current board"
            )


def run_rate(profile_path, prep_bank_path, tmpdir):
    """Pull the full board (with JDs), score it against the user's profile.

    Returns (ratings list, live meta dict id -> {META_KEYS...}).
    """
    print("pulling all boards for a full-JD snapshot...", flush=True)
    all_roles, stats = refresh.pull_boards()
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("_")}))
    failed = refresh.failed_companies(stats)
    assert not failed, (
        f"boards failed this pull: {sorted(failed)} — a paid board must not "
        "silently miss a company; retry when they recover"
    )
    # pull_boards returns EVERY role (the public layer); only lane roles get
    # LLM-scored — same cost envelope as the main board's daily refresh.
    roles = [r for r in all_roles if r.get("lane")]
    assert roles, "pull_boards returned no lane roles"

    tmp_shortlist = os.path.join(tmpdir, "user_shortlist.json")
    tmp_ratings = os.path.join(tmpdir, "user_ratings.json")
    with open(tmp_shortlist, "w") as f:
        json.dump(roles, f)

    # rate.py treats a missing prep-bank file as "run without bank context";
    # never fall back to rate.py's default (that is Alex's personal bank).
    bank_arg = (
        os.path.abspath(prep_bank_path)
        if prep_bank_path
        else os.path.join(tmpdir, "no_prep_bank.json")  # intentionally absent
    )

    runner = ["uv", "run"] if shutil.which("uv") else [sys.executable]
    cmd = runner + [
        RATE_PY,
        "--shortlist",
        tmp_shortlist,
        "--profile",
        os.path.abspath(profile_path),
        "--prep-bank",
        bank_arg,
        "--out",
        tmp_ratings,
    ]
    print("scoring", len(roles), "roles:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"rate.py failed (exit {r.returncode}) — no files written")

    ratings = json.load(open(tmp_ratings))["ratings"]
    live = {r_["id"]: {k: r_[k] for k in META_KEYS} for r_ in roles}
    return ratings, live


def build_payload(slug, live, ratings, bank):
    """Same merge/rank logic as refresh.cmd_build, for a first-run board."""
    by_id = {r["id"]: r for r in ratings}
    jobs = []
    for rid, meta in live.items():
        rated = by_id.get(rid)
        if not rated or "scores" not in rated:
            continue  # never show an unscored role
        rec = {k: v for k, v in rated.items() if k != "id"}
        rec.update(id=rid, first_seen=TODAY, last_seen=TODAY, closed=False, **meta)
        rec["fit"] = refresh.fit_score(rec["scores"])
        rec["hidden"] = (
            rec.get("group", "anthropic") != "anthropic"
            and rec["fit"] < refresh.MIN_FIT_NON_ANTHROPIC
        )
        jobs.append(rec)

    assert jobs, "no roles carried both live metadata and a score — nothing to publish"
    jobs.sort(key=lambda j: (-j["fit"], -(j.get("prob") or 0)))
    for i, j in enumerate(jobs):
        j["rank"] = i + 1
    n_shown = sum(1 for j in jobs if not j["hidden"])

    payload = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="minutes"),
            "date": TODAY,
            "user": slug,
            "total": len(jobs),
            "shown": n_shown,
            "new_today": 0,
            "new_ids": [],
            "fit_weights": refresh.FIT_W,
            "companies": sorted({j["company"] for j in jobs}),
        },
        "jobs": jobs,
    }
    if bank:
        payload["bank"] = bank
    return payload


def emit(slug, payload, password):
    outdir = os.path.join(U_DIR, slug)
    os.makedirs(outdir, exist_ok=True)

    blob = encrypt_blob(payload, password)
    assert decrypt_blob(blob, password) == payload, "encrypt/decrypt round-trip failed"

    enc_path = os.path.join(outdir, "data.enc.js")
    with open(enc_path, "w") as f:
        f.write(
            f"// Encrypted personal job board for /jobs/u/{slug}/ — "
            "decrypted client-side.\n"
            "window.JOBS_ENC = " + json.dumps(blob) + ";\n"
        )

    template = open(TEMPLATE).read()
    assert "__SLUG__" in template, f"{TEMPLATE} has no __SLUG__ placeholder"
    html_path = os.path.join(outdir, "index.html")
    with open(html_path, "w") as f:
        f.write(template.replace("__SLUG__", slug))

    print(f"wrote {enc_path} ({os.path.getsize(enc_path) // 1024} KB)")
    print(f"wrote {html_path}")
    return outdir


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(
        description="Build a password-gated personal board at /jobs/u/<slug>/"
    )
    ap.add_argument("--slug", required=True, help="url-safe id: [a-z0-9_-]")
    ap.add_argument("--profile", required=True, help="the customer's profile.md")
    ap.add_argument(
        "--password",
        default=os.environ.get("JOBS_USER_PASSWORD"),
        help="page password (or set JOBS_USER_PASSWORD env — avoids shell "
        "history/ps exposure); share out-of-band",
    )
    ap.add_argument(
        "--prep-bank",
        dest="prep_bank",
        default=None,
        help="the customer's prep_bank.json (optional)",
    )
    ap.add_argument(
        "--skip-rate",
        action="store_true",
        help="no pull/API: reuse --ratings + live_index.json",
    )
    ap.add_argument(
        "--ratings",
        default=None,
        help="ratings file, REQUIRED with --skip-rate (use the customer's "
        "users/<slug>.ratings.json — never Alex's refresh/ratings.json)",
    )
    args = ap.parse_args()

    slug = args.slug
    assert re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{0,40}", slug
    ), f"bad slug {slug!r} — use lowercase letters, digits, - and _"
    if args.skip_rate:
        assert args.ratings, "--skip-rate requires --ratings <file> (the customer's saved ratings)"
    assert args.password, "no password: pass --password or set JOBS_USER_PASSWORD"
    if len(args.password) < 8:
        print("WARNING: password under 8 chars — PBKDF2 only slows brute force")
    assert os.path.exists(args.profile), f"profile not found: {args.profile}"
    if args.prep_bank:
        assert os.path.exists(args.prep_bank), f"prep bank not found: {args.prep_bank}"

    ensure_users_gitignore()
    assert_fresh_pull()

    bank = json.load(open(args.prep_bank)) if args.prep_bank else None

    if args.skip_rate:
        assert os.path.exists(
            args.ratings
        ), f"--skip-rate needs an existing ratings file, none at {args.ratings}"
        ratings = json.load(open(args.ratings))["ratings"]
        live = json.load(open(LIVE_INDEX))
        print(
            f"skip-rate: {len(ratings)} ratings from {args.ratings}, "
            f"{len(live)} live roles from live_index.json"
        )
    else:
        assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY required"
        tmpdir = tempfile.mkdtemp(prefix=f"make_user_{slug}_")
        ratings, live = run_rate(args.profile, args.prep_bank, tmpdir)
        # keep the user's ratings for cheap re-emits (--skip-rate --ratings ...)
        kept = os.path.join(USERS_DIR, f"{slug}.ratings.json")
        with open(kept, "w") as f:
            json.dump({"ratings": ratings}, f, indent=1)
        print(f"saved user ratings to {kept} (gitignored)")

    payload = build_payload(slug, live, ratings, bank)
    outdir = emit(slug, payload, args.password)

    m = payload["meta"]
    n_prep = sum(1 for j in payload["jobs"] if "prep" in j)
    print(
        f"\nboard: {m['total']} roles scored, {m['shown']} above bar, "
        f"{n_prep} with prep decks"
    )
    print(
        f"""
DONE — personal board for '{slug}'
  URL       https://alex-loftus.com/jobs/u/{slug}/
  password  {args.password}   (send URL and password via separate channels)

Next steps:
  cd {os.path.normpath(os.path.join(HERE, '..', '..', '..'))}
  git add {os.path.relpath(outdir, os.path.normpath(os.path.join(HERE, '..', '..', '..')))}
  git commit -m "jobs: personal board for {slug}"
  git push
"""
    )


if __name__ == "__main__":
    main()
