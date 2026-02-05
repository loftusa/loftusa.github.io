"""
extract URLS from the resume.txt file

Goal: 
    1. go through the file, extract all URLS
    2. classify (youtube, arxiv, github, scholar, pdf, html) based on the path)
    3. capture the line the URL appears on as context
    4. deduplicate YouTube URLs with separate timestamps
    5. print results if run as main
"""

import re
from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

resume_filepath = Path(__file__).parent.parent / 'resume.txt'
url_pattern = re.compile(r'https?://[^\s\)]+')
MatchPossibility = Literal['youtube', 'arxiv', 'github', 'scholar', 'pdf', 'html']


@dataclass
class Url:
    url: str
    classification: str
    line_number: int
    context: str


def classify_match(url: str) -> MatchPossibility:
    """Classify a URL based on its domain and path."""
    url_lower = url.lower()
    
    if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif 'arxiv.org' in url_lower:
        return 'arxiv'
    elif 'github.com' in url_lower:
        return 'github'
    elif 'scholar.google' in url_lower:
        return 'scholar'
    elif url_lower.endswith('.pdf'):
        return 'pdf'
    else:
        return 'html'

def youtube_video_id(url: str) -> str:
    """Extract the video ID from any YouTube URL format, for dedup."""
    parsed = urlparse(url)
    if 'youtu.be' in parsed.netloc:
        return parsed.path.lstrip('/')
    # youtube.com/watch?v=<id> or youtube.com/playlist?list=<id>
    params = parse_qs(parsed.query)
    if 'v' in params:
        return params['v'][0]
    if 'list' in params:
        return 'playlist:' + params['list'][0]
    return url  # fallback: use full URL as key


def parse_urls(filepath: Path) -> list[Url]:
    urls = []
    seen: set[str] = set()
    with filepath.open('r') as f:
        for i, line in enumerate(f):
            matches = re.findall(url_pattern, line)
            for m in matches:
                m = m.rstrip(".,;:")
                classification = classify_match(m)

                # Dedup key: video ID for YouTube, raw URL for everything else
                key = youtube_video_id(m) if classification == 'youtube' else m
                if key in seen:
                    continue
                seen.add(key)

                url_obj = Url(url=m, classification=classification, line_number=i, context=line.strip())
                urls.append(url_obj)
    return urls


if __name__ == '__main__':
    urls = parse_urls(resume_filepath)
    print(f"Found {len(urls)} URLs\n")
    for u in urls:
        print(f"[{u.classification:>8}] {u.url}")
    print(f"\nBreakdown:")
    from collections import Counter
    counts = Counter(u.classification for u in urls)
    for cls, count in counts.most_common():
        print(f"  {cls}: {count}")

