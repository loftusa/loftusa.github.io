# /// script
# dependencies = ["anthropic>=0.92.0", "cryptography>=42.0"]
# ///
"""Score shortlisted roles with the Claude API and generate per-role prep.

Two passes (both run by default):
  score — every role in shortlist.json gets the 5 role-shape dimensions,
          an offer probability, and why/gap one-liners. Cheap, batched.
  prep  — roles clearing the fit bar additionally get a prep block: resume
          angle, interview drill questions (mapped to STAR stories from the
          encrypted prep bank), a 30s pitch opener, and a negotiation line.

Writes ratings.json in the exact shape refresh.py --build expects. Fails loud:
exits non-zero unless EVERY shortlisted id gets a valid score, so a partial
rating run can never silently publish a degraded board.

Env:  ANTHROPIC_API_KEY   (required)
      JOBS_PAGE_PASSWORD  (required — decrypts prep_bank.enc.json)
      RATE_MODEL          (default claude-sonnet-4-6)
Usage: python3 rate.py [--limit N] [--score-only]
"""

import argparse
import json
import os
import sys

import anthropic
from refresh import PREP_BANK_ENC, decrypt_blob, fit_score

HERE = os.path.dirname(os.path.abspath(__file__))
SHORTLIST = os.path.join(HERE, "shortlist.json")
RATINGS = os.path.join(HERE, "ratings.json")
PROFILE = open(os.path.join(HERE, "profile.md")).read()

MODEL = os.environ.get("RATE_MODEL", "claude-sonnet-4-6")
PREP_BAR = 6.0  # fit >= this earns a prep block
PREP_CAP = 45  # per run, best-first (day 1 protection)
SCORE_BATCH = 10
PREP_BATCH = 4

DIMS = ("people", "stage", "social", "technical", "domain")
SCORE = {"type": "integer"}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    **{d: SCORE for d in DIMS},
                    "prob": {"type": "integer"},
                    "why": {"type": "string"},
                    "gap": {"type": "string"},
                },
                "required": ["id", *DIMS, "prob", "why", "gap"],
            },
        }
    },
    "required": ["ratings"],
}

PREP_SCHEMA = {
    "type": "object",
    "properties": {
        "preps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "resume_angle": {"type": "array", "items": {"type": "string"}},
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "q": {"type": "string"},
                                "story": {"type": "string"},
                                "hint": {"type": "string"},
                            },
                            "required": ["q", "story", "hint"],
                        },
                    },
                    "pitch": {"type": "string"},
                    "negotiation": {"type": "string"},
                },
                "required": ["id", "resume_angle", "questions", "pitch", "negotiation"],
            },
        }
    },
    "required": ["preps"],
}


def call_json(client, system, user, schema, tool_name):
    """One structured-output call; the tool schema forces valid JSON."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": tool_name,
                "description": "Return the result.",
                "input_schema": schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return block.input


def role_snippet(r, jd_chars):
    comp = f" | comp ${r['comp'][0]:,}-${r['comp'][1]:,}" if r.get("comp") else ""
    return (
        f"### id={r['id']}\n{r['company']} — {r['title']}\n"
        f"Locations: {', '.join(r['locations']) or 'unlisted'}"
        f"{' | REMOTE' if r['remote'] else ''}{comp}\n"
        f"JD: {r['jd_text'][:jd_chars]}\n"
    )


def pass_score(client, roles):
    system = (
        "You are scoring job openings for one specific candidate, for his "
        "private job board. Score each role EXACTLY per the rubric in the "
        "profile below — dimensions describe the ROLE-SHAPE and his realistic "
        "clearance of its bar, not generic prestige.\n\n" + PROFILE
    )
    out = {}
    for i in range(0, len(roles), SCORE_BATCH):
        batch = roles[i : i + SCORE_BATCH]
        user = (
            "Score EVERY role below (all of: "
            + ", ".join(r["id"] for r in batch)
            + ").\n\n"
            + "\n".join(role_snippet(r, 3500) for r in batch)
        )
        got = call_json(client, system, user, SCORE_SCHEMA, "submit_ratings")
        for r in got["ratings"]:
            ok = (
                all(isinstance(r.get(d), int) and 1 <= r[d] <= 10 for d in DIMS)
                and 0 <= r.get("prob", -1) <= 60
            )
            if ok:
                out[r["id"]] = r
        done = min(i + SCORE_BATCH, len(roles))
        print(f"score: {done}/{len(roles)} ({len(out)} valid)", flush=True)
    return out


def pass_prep(client, roles, scored, bank):
    # deterministic fit decides who earns prep
    def fit_of(rid):
        return fit_score({d: scored[rid][d] for d in DIMS})

    eligible = sorted(
        (r for r in roles if r["id"] in scored and fit_of(r["id"]) >= PREP_BAR),
        key=lambda r: -fit_of(r["id"]),
    )[:PREP_CAP]
    if not eligible:
        return {}
    system = (
        "You are building interview-prep blocks for one candidate's private "
        "job board. Use his profile and his PERSONAL PREP BANK below.\n\n"
        + PROFILE
        + "\n\nPERSONAL PREP BANK (STAR stories, hard-question rebuttals, pitch "
        "components — reference stories by their exact `key`):\n"
        + json.dumps(bank, indent=1)
        + "\n\nFor EACH role produce:\n"
        "- resume_angle: exactly 3 bullets — which of his career highlights to "
        "LEAD with for THIS role and why, concrete (name the highlight, tie it "
        "to a JD must-have). Ultralearning 'directness': prep maps straight to "
        "what this JD tests.\n"
        "- questions: exactly 6 likely interview questions FOR THIS JD (mix: "
        "2 behavioral, 2 role-specific technical/program-design, 1 hard/skeptical "
        "about his background, 1 values/safety). For each: `story` = the prep-bank "
        "key of the best STAR story or rebuttal to use ('' if none fits), `hint` = "
        "one line on how to angle the answer. These become retrieval-practice "
        "flashcards — make them the questions an interviewer would actually ask.\n"
        "- pitch: a 30-second spoken opener tailored to THIS role, Winston-style: "
        "open with a promise (what he'll do for them), one cycled key idea, "
        "concrete evidence, no throat-clearing. First person, natural speech.\n"
        "- negotiation: one line — realistic comp anchor for this role/level and "
        "his leverage (BATNA: current OpenAI contract + PhD; competing processes)."
    )
    out = {}
    for i in range(0, len(eligible), PREP_BATCH):
        batch = eligible[i : i + PREP_BATCH]
        user = (
            "Build prep blocks for EVERY role below (all of: "
            + ", ".join(r["id"] for r in batch)
            + ").\n\n"
            + "\n".join(role_snippet(r, 5000) for r in batch)
        )
        got = call_json(client, system, user, PREP_SCHEMA, "submit_preps")
        for p in got["preps"]:
            if len(p.get("resume_angle", [])) == 3 and len(p.get("questions", [])) == 6:
                out[p["id"]] = p
        print(f"prep: {min(i + PREP_BATCH, len(eligible))}/{len(eligible)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only first N (testing)")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY required"
    password = os.environ.get("JOBS_PAGE_PASSWORD", "")
    assert password, "JOBS_PAGE_PASSWORD required"

    roles = json.load(open(SHORTLIST))
    if args.limit:
        roles = roles[: args.limit]
    if not roles:
        json.dump({"ratings": []}, open(RATINGS, "w"))
        print("shortlist empty — nothing to score")
        return

    bank = decrypt_blob(json.load(open(PREP_BANK_ENC)), password)
    client = anthropic.Anthropic()

    scored = pass_score(client, roles)
    missing = [r["id"] for r in roles if r["id"] not in scored]
    if missing:
        sys.exit(
            f"FAIL: {len(missing)} roles got no valid score "
            f"(e.g. {missing[:5]}) — refusing to write partial ratings"
        )

    preps = {} if args.score_only else pass_prep(client, roles, scored, bank)

    ratings = []
    for r in roles:
        s = scored[r["id"]]
        rec = {
            "id": r["id"],
            "group": r["group"],
            "scores": {d: s[d] for d in DIMS},
            "prob": s["prob"],
            "why": s["why"],
            "gap": s["gap"],
        }
        if r["id"] in preps:
            rec["prep"] = {
                k: preps[r["id"]][k]
                for k in ("resume_angle", "questions", "pitch", "negotiation")
            }
        ratings.append(rec)
    json.dump({"ratings": ratings}, open(RATINGS, "w"), indent=1)
    print(f"wrote ratings.json: {len(ratings)} scored, {len(preps)} with prep")


if __name__ == "__main__":
    main()
