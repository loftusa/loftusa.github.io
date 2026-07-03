# /// script
# dependencies = ["anthropic>=0.92.0", "pillow>=10.0"]
# ///
"""Rate the shortlisted listings with the Claude API — photos + body text.

For each listing in shortlist.json this script downloads the full photo
gallery, tiles it into one contact-sheet montage (Pillow), and sends batches
of listings (montage image + posting text) to the Messages API with a strict
JSON schema. Writes ratings.json in the exact shape refresh.py --build expects.

Fails loud: exits non-zero unless EVERY listing id gets a valid rating, so a
partial/garbled rating run can never produce a silently-degraded board.

Env:  ANTHROPIC_API_KEY  (required)
      RATE_MODEL         (default claude-sonnet-4-6; claude-haiku-4-5 = cheaper)
Usage: python3 rate.py [--limit N] [--batch-size K]
"""

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import anthropic
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SHORTLIST = os.path.join(HERE, "shortlist.json")
RATINGS = os.path.join(HERE, "ratings.json")

MODEL = os.environ.get("RATE_MODEL", "claude-sonnet-4-6")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

RUBRIC = """You are rating SF Bay Area rental listings for Alex, for the live board at alex-loftus.com/houses.

WHO IT'S FOR — Alex: Moving to the SF Bay Area soon. Budget up to $3000/mo, ideally ~$2000. Open to housemates SPECIFICALLY for networking with tech founders/CEOs, AI-lab people (Anthropic/OpenAI/DeepMind and similar), and elite-university types (MIT/Stanford/Harvard/Berkeley — grad students, researchers, engineers) — NOT artists or creative-collective scenes. OR a quiet studio in a nice area. Has a car; needs easy access to BOTH downtown SF and downtown Berkeley (FAR Labs), <=35 min drive each ideal. Loves nature/trees; loves Mill Valley / Sausalito / Marin; also fine in the city.

Each listing below has a photo MONTAGE (a grid of ALL its photos) and its posting text. LOOK AT EVERY PHOTO in each montage before scoring that listing.

Score each dimension as an integer 1-10, literally as defined:
- nature — proximity to trees/parks/hills/water/greenery (text + what the photos show). 10 = in/next to nature; 1 = dense concrete.
- quiet — residential calm; low traffic/noise. 10 = quiet side street; 1 = above a bar / loud arterial.
- nice — how desirable/safe/well-kept the area & unit are, per the photos. 10 = clearly nice & good shape; 1 = rough/run-down.
- social — networking potential SPECIFICALLY with Alex's target crowd (tech founders/CEOs, AI-lab, MIT/Stanford/Harvard/Berkeley people). HIGH (8-10) ONLY when the body shows a shared house whose housemates/vibe skew that way (engineers, founders, startup/tech pros, grad students, researchers). MID (4-6) generic shared place, no signal. LOW (1-3) for a solo studio/1BR with no housemates, AND for shared houses that are a DIFFERENT scene — artists, musicians, record labels, "creative collective/commune", party houses. Do NOT treat artist/creative housemates as a positive. Low social does not by itself sink a listing.
- value — price vs. what you get (space/condition/location per the photos) against ~$2000 target / $3000 ceiling. 10 = underpriced; 1 = overpriced.
- commute — ease of reaching BOTH downtown SF and downtown Berkeley by car, <=35 min each ideal. 10 = easy to both; 1 = far/painful to at least one.
- aesthetic — HOW GOOD THE PLACE LOOKS IN ITS PHOTOS, judged ONLY from the montage. Attractive, well-lit, tasteful, clean, good finishes/light/views? 10 = genuinely beautiful, magazine-quality photos of an attractive space. 5 = ordinary/plain but fine. 1-3 = ugly, dark, cluttered, grimy, low-effort/blurry photos, or a clearly unappealing space (or no usable photos). This is weighted into the ranking — do not give high aesthetic to a place whose pictures are bad.
Then:
- fit — overall 1-10 holistic fit for Alex (a deterministic model recomputes the final ranking; give your honest overall anyway). A quiet studio in a nice green area can be high fit even with low social. NEVER boost fit for an artsy/creative house just because it's social.
- why — one line in Alex's terms, referencing what the photos/listing show. If you cite housemates as a plus, it must be because they fit the tech/founder/AI/elite-academic profile.
- live — false if the post looks dead/expired/duplicate or like a scam (too cheap for the area, generic copy, off-platform payment).
- commercial — true if it's an office/retail/parking/commercial space, not a home.

Rate EVERY listing id you are given, once each."""

GIO_RUBRIC = """You are rating SF rental listings for GIO, for the "For Gio" section of the live board at alex-loftus.com/houses.

WHO IT'S FOR — Gio: works AT OpenAI's headquarters (1455 Third St, Mission Bay, San Francisco). His #1 criterion is a SHORT WALK to that office — each listing below states its computed walk time; treat it as ground truth and sanity-check any location claims in the text against it. Budget is flexible (rooms up to ~$3,500, apartments/studios up to ~$6,000) but he still wants his money's worth. He wants a place that looks genuinely nice; friendly housemates are a plus when it's a shared place — young-professional vibe welcome; he has NO networking agenda (he already works at OpenAI).

Each listing below has a photo MONTAGE (a grid of ALL its photos) and its posting text. LOOK AT EVERY PHOTO in each montage before scoring that listing.

Score each dimension as an integer 1-10, literally as defined:
- nature — greenery/water nearby: bay waterfront, Mission Bay parks, street trees (text + what the photos show). 10 = right on a park/waterfront; 1 = bare concrete canyon.
- quiet — residential calm; low traffic/noise; active construction (common in Mission Bay) counts against. 10 = quiet; 1 = above a bar / loud arterial / construction next door.
- nice — how desirable/safe/well-kept the area & unit are, per the photos. 10 = clearly nice & good shape; 1 = rough/run-down.
- social — housemate/vibe signal for a shared place: friendly, sociable, young-professional households HIGH (7-10); generic shared place with no signal MID (4-6); a solo studio/1BR with no housemates LOW (1-3) on this dimension — but that must NOT drag your overall read down (the final ranking handles it).
- value — price vs. what you get (space/condition/location per the photos) for near-office San Francisco. 10 = underpriced for what it is; 1 = overpriced.
- commute — your judgment of the walk to OpenAI HQ given the stated walk time and the text (a deterministic model recomputes this downstream; score it honestly anyway). 10 = a few minutes' stroll; 1 = not realistically walkable.
- aesthetic — HOW GOOD THE PLACE LOOKS IN ITS PHOTOS, judged ONLY from the montage. Attractive, well-lit, tasteful, clean, good finishes/light/views? 10 = genuinely beautiful, magazine-quality photos of an attractive space. 5 = ordinary/plain but fine. 1-3 = ugly, dark, cluttered, grimy, low-effort/blurry photos, or a clearly unappealing space (or no usable photos). This is weighted into the ranking — do not give high aesthetic to a place whose pictures are bad.
Then:
- fit — overall 1-10 holistic fit for Gio (a deterministic model recomputes the final ranking; give your honest overall anyway). Short walk + nice-looking place = high fit; a gorgeous place 30+ minutes away is NOT high fit.
- why — one line in Gio's terms, referencing what the photos/listing show and the walk time.
- live — false if the post looks dead/expired/duplicate or like a scam (too cheap for the area, generic copy, off-platform payment).
- commercial — true if it's an office/retail/parking/commercial space, not a home.

Rate EVERY listing id you are given, once each."""

RUBRICS = {"alex": RUBRIC, "gio": GIO_RUBRIC}

SCORE = {"type": "integer"}  # 1-10 validated client-side (API schema rejects min/max)
RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "nature": SCORE,
                    "quiet": SCORE,
                    "nice": SCORE,
                    "social": SCORE,
                    "value": SCORE,
                    "commute": SCORE,
                    "aesthetic": SCORE,
                    "fit": SCORE,
                    "why": {"type": "string"},
                    "live": {"type": "boolean"},
                    "commercial": {"type": "boolean"},
                },
                "required": [
                    "id",
                    "nature",
                    "quiet",
                    "nice",
                    "social",
                    "value",
                    "commute",
                    "aesthetic",
                    "fit",
                    "why",
                    "live",
                    "commercial",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ratings"],
    "additionalProperties": False,
}

SCORE_KEYS = (
    "nature",
    "quiet",
    "nice",
    "social",
    "value",
    "commute",
    "aesthetic",
    "fit",
)


def fetch_bytes(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def build_montage(listing, thumb=280, cols=4, max_imgs=24):
    """Tile all of a listing's photos into one JPEG contact sheet. None if no photos."""
    urls = (listing.get("imgs") or [])[:max_imgs]
    thumbs = []
    for u in urls:
        try:
            im = Image.open(io.BytesIO(fetch_bytes(u))).convert("RGB")
            im.thumbnail((thumb, thumb * 3 // 4))
            thumbs.append(im)
        except Exception:
            continue
    if not thumbs:
        return None
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = thumb * 3 // 4
    sheet = Image.new("RGB", (cols * thumb, rows * cell_h), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * cell_h))
    # keep the long edge modest so each montage stays ~2K image tokens
    sheet.thumbnail((1400, 1400))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode()


def listing_text(r):
    body = (r.get("body") or "")[:1500]
    if r.get("aud") == "gio":
        place = f"~{r.get('walk_min', '?')} min walk to OpenAI HQ (1455 Third St) — {r.get('hood')}"
    else:
        place = f"{r.get('hood')} ({r.get('region')})"
    return (
        f"### Listing {r['id']} — ${r['price']:,}/mo — {place} — "
        f"{'room in shared home' if r.get('bucket') == 'room' else 'apartment/studio'}\n"
        f"Title: {r.get('title')}\nPosting text: {body or '(no body scraped)'}"
    )


def group_batches(sel, batch_size):
    """[(aud, rows)] — batches never mix audiences; stable order within each."""
    out = []
    for aud in ("alex", "gio"):
        rows = [r for r in sel if r.get("aud", "alex") == aud]
        out += [
            (aud, rows[i : i + batch_size]) for i in range(0, len(rows), batch_size)
        ]
    return out


def validate(batch_ids, obj):
    """Return {id: rating} for valid entries; raise ValueError on malformed scores."""
    out = {}
    for rr in obj.get("ratings", []):
        if rr["id"] not in batch_ids:
            continue
        for k in SCORE_KEYS:
            v = rr[k]
            if not isinstance(v, int) or not 1 <= v <= 10:
                raise ValueError(f"{rr['id']}.{k}={v!r} not an int in 1..10")
        out[rr["id"]] = rr
    return out


def rate_batch(client, batch, montages, rubric):
    content = [{"type": "text", "text": rubric}]
    for r in batch:
        content.append({"type": "text", "text": listing_text(r)})
        b64 = montages.get(r["id"])
        if b64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                }
            )
        else:
            content.append(
                {
                    "type": "text",
                    "text": "(no photos could be downloaded for this listing — score aesthetic 2-3)",
                }
            )
    content.append(
        {
            "type": "text",
            "text": f"Now rate ALL {len(batch)} listings above ({', '.join(r['id'] for r in batch)}).",
        }
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "disabled"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": RATING_SCHEMA},
        },
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("response truncated (max_tokens) — reduce batch size")
    if response.stop_reason == "refusal":
        raise RuntimeError("model refused the batch")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=0, help="only rate first N (smoke test)"
    )
    ap.add_argument("--batch-size", type=int, default=5)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "FATAL: ANTHROPIC_API_KEY not set — cannot rate. (Sweep mode needs no key.)"
        )
    sel = json.load(open(SHORTLIST))
    if args.limit:
        sel = sel[: args.limit]
    assert sel, "shortlist.json is empty"
    ids = [r["id"] for r in sel]
    n_gio = sum(1 for r in sel if r.get("aud") == "gio")
    print(
        f"rating {len(sel)} listings ({len(sel) - n_gio} alex + {n_gio} gio) "
        f"with {MODEL} (batch={args.batch_size})"
    )

    print("building montages...")
    with ThreadPoolExecutor(max_workers=6) as ex:
        montages = dict(ex.map(lambda r: (r["id"], build_montage(r)), sel))
    n_m = sum(1 for v in montages.values() if v)
    print(f"montages: {n_m}/{len(sel)}")

    client = anthropic.Anthropic(max_retries=4)
    ratings, in_tok, out_tok = {}, 0, 0
    batches = group_batches(sel, args.batch_size)
    for bi, (aud, batch) in enumerate(batches):
        batch_ids = {r["id"] for r in batch}
        got = {}
        for attempt in (1, 2):
            todo = [r for r in batch if r["id"] not in got]
            if not todo:
                break
            try:
                obj, usage = rate_batch(client, todo, montages, RUBRICS[aud])
                got.update(validate({r["id"] for r in todo}, obj))
                in_tok += usage.input_tokens
                out_tok += usage.output_tokens
            except (json.JSONDecodeError, ValueError, RuntimeError, KeyError) as e:
                print(f"  batch {bi+1} attempt {attempt} problem: {e}", file=sys.stderr)
        missing = batch_ids - set(got)
        if missing:
            sys.exit(
                f"FATAL: could not get valid ratings for {sorted(missing)} after retries."
            )
        ratings.update(got)
        print(f"  batch {bi+1}/{len(batches)}: {len(got)}/{len(batch)} rated")

    assert set(ids) == set(ratings), "coverage check failed"
    json.dump([ratings[i] for i in ids], open(RATINGS, "w"), indent=1)
    flagged = [i for i in ids if not ratings[i]["live"] or ratings[i]["commercial"]]
    print(
        f"wrote {RATINGS}: {len(ratings)} ratings | dead/commercial flagged: {flagged or 'none'}\n"
        f"tokens: {in_tok:,} in / {out_tok:,} out"
    )


if __name__ == "__main__":
    main()
