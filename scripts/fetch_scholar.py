"""Fetch Google Scholar publications and update the website's publications tab.

Two jobs:
1. Add any new publications not already on the site.
2. Update citation counts for all Scholar-sourced publications.
"""
import re
from pathlib import Path

from scholarly import scholarly

SCHOLAR_ID = "_Njcmm8AAAAJ"
ABOUT_MD = Path(__file__).parent.parent / "_pages" / "about.md"

# Marker that identifies the publications section
PUB_SECTION_START = "## Talks & Publications"
PUB_SECTION_END = "</div>"


def fetch_scholar_pubs() -> list[dict]:
    """Fetch all publications from Google Scholar profile."""
    author = scholarly.search_author_id(SCHOLAR_ID)
    author = scholarly.fill(author, sections=["publications"])
    pubs = []
    for pub in author.get("publications", []):
        filled = scholarly.fill(pub)
        bib = filled.get("bib", {})
        pubs.append(
            {
                "title": bib.get("title", ""),
                "authors": bib.get("author", ""),
                "venue": bib.get(
                    "venue", bib.get("journal", bib.get("booktitle", ""))
                ),
                "year": bib.get("pub_year", ""),
                "url": filled.get("pub_url", filled.get("eprint_url", "")),
                "citation": bib.get("citation", ""),
                "num_citations": filled.get("num_citations", 0),
            }
        )
    return pubs


def get_existing_titles(text: str) -> set[str]:
    """Extract publication titles already on the website (lowercased for fuzzy match)."""
    titles = re.findall(r"\[([^\]]+)\]\([^\)]*\)", text)
    return {t.lower().strip() for t in titles}


def title_matches(a: str, b: str) -> bool:
    """Check if two titles match (case-insensitive, substring)."""
    a, b = a.lower().strip(), b.lower().strip()
    if a in b or b in a:
        return True
    if len(a) > 40 and a[:40] == b[:40]:
        return True
    return False


def is_duplicate(new_title: str, existing: set[str]) -> bool:
    """Check if a title (or close variant) already exists."""
    return any(title_matches(new_title, t) for t in existing)


def format_pub(pub: dict) -> str:
    """Format a publication as a markdown list item with citation count."""
    title = pub["title"]
    url = pub["url"] or ""
    year = str(pub["year"])
    venue = pub["venue"]
    citations = pub.get("num_citations", 0)

    parts = []
    if venue:
        parts.append(venue)
    if year:
        parts.append(year)
    desc = ", ".join(parts)
    if desc:
        desc = f": {desc}"

    cite_badge = ""
    if citations and citations > 0:
        cite_badge = f" ({citations} citations)"

    return f"- [{title}]({url}){desc}{cite_badge}"


def update_citations(text: str, pubs: list[dict]) -> tuple[str, int]:
    """Update citation counts on existing publication lines. Returns (new_text, count_updated)."""
    updated = 0
    for pub in pubs:
        citations = pub.get("num_citations", 0)
        if not citations:
            continue

        # Find lines whose link text matches this pub's title
        for existing_title in re.findall(r"\[([^\]]+)\]\([^\)]*\)", text):
            if not title_matches(pub["title"], existing_title):
                continue

            # Find the full line containing this title
            escaped = re.escape(existing_title)
            line_pattern = re.compile(
                rf"^(- \[{escaped}\]\([^\)]*\)[^\n]*?)"
                rf"(?:\s*\(\d+ citations?\))?"
                rf"\s*$",
                re.MULTILINE,
            )
            match = line_pattern.search(text)
            if match:
                old_line = match.group(0).rstrip()
                # Strip any existing citation badge from the base
                base = match.group(1).rstrip()
                base = re.sub(r"\s*\(\d+ citations?\)\s*$", "", base)
                new_line = f"{base} ({citations} citation{'s' if citations != 1 else ''})"
                if old_line != new_line:
                    text = text.replace(old_line, new_line)
                    print(f"  Updated citations: {existing_title} -> {citations}")
                    updated += 1
            break

    return text, updated


def update_about_md(pubs: list[dict]) -> tuple[int, int]:
    """Insert new publications and update citations. Returns (added, citations_updated)."""
    text = ABOUT_MD.read_text()
    existing = get_existing_titles(text)

    # 1. Add new publications
    to_add = [p for p in pubs if not is_duplicate(p["title"], existing)]
    added = 0
    if to_add:
        to_add.sort(key=lambda p: str(p.get("year", "0")), reverse=True)
        new_lines = "\n".join(format_pub(p) for p in to_add)

        marker = f"{PUB_SECTION_START}\n"
        idx = text.find(marker)
        assert idx != -1, f"Could not find '{PUB_SECTION_START}' in about.md"
        insert_pos = idx + len(marker) + 1
        text = text[:insert_pos] + new_lines + "\n" + text[insert_pos:]
        added = len(to_add)
        for p in to_add:
            print(f"  Added: {p['title']} ({p['year']})")

    # 2. Update citation counts
    text, cite_updated = update_citations(text, pubs)

    if added or cite_updated:
        ABOUT_MD.write_text(text)

    return added, cite_updated


def main():
    print("Fetching Google Scholar publications...")
    pubs = fetch_scholar_pubs()
    print(f"Found {len(pubs)} publications on Google Scholar.")
    added, cite_updated = update_about_md(pubs)
    if added:
        print(f"\nAdded {added} new publication(s).")
    if cite_updated:
        print(f"Updated {cite_updated} citation count(s).")
    if not added and not cite_updated:
        print("Website is up to date.")


if __name__ == "__main__":
    main()
