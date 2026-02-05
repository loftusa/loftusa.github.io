"""
Take a url and its classification, return text + metadata
- html loader
- arxiv loader
- youtube loader
- pdf loader
- github loader

Common return format for all loaders:
{'text': str, 'url': str, 'title': str, 'doc_type': str}

dispatcher function takes a Url object and returns this dict.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

import fitz  # pymupdf
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

from extract_urls import Url, youtube_video_id

logger = logging.getLogger(__name__)

Document = dict[str, str]


def load_document(url: Url) -> Optional[Document]:
    """Dispatch a Url to the right loader. Returns None on skip/failure."""
    loader_mapping = {
        'html': html_loader,
        'youtube': youtube_loader,
        'arxiv': arxiv_loader,
        'pdf': pdf_loader,
        'github': github_loader,
        'scholar': scholar_loader,
    }
    loader = loader_mapping.get(url.classification)
    if loader is None:
        logger.warning(f"No loader for classification: {url.classification}")
        return None
    try:
        return loader(url)
    except Exception as e:
        logger.warning(f"Failed to load {url.url}: {e}")
        return None


def html_loader(url: Url) -> Document:
    """Generic web scraper: requests + beautifulsoup."""
    resp = requests.get(url.url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Strip non-content tags
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url.url
    text = soup.get_text(separator='\n', strip=True)

    return {'text': text, 'url': url.url, 'title': title, 'doc_type': 'html'}


def arxiv_loader(url: Url) -> Document:
    """
    Fetch arXiv abstract page and extract title + abstract.

    TODO: extract entire text.
    """
    # Ensure we hit the abstract page, not the PDF
    abstract_url = url.url.replace('/pdf/', '/abs/')
    resp = requests.get(abstract_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    title_tag = soup.find('h1', class_='title')
    title = title_tag.get_text(strip=True).removeprefix('Title:') if title_tag else url.url

    abstract_tag = soup.find('blockquote', class_='abstract')
    abstract = abstract_tag.get_text(strip=True).removeprefix('Abstract:') if abstract_tag else ''

    # Also grab author list
    authors_tag = soup.find('div', class_='authors')
    authors = authors_tag.get_text(strip=True).removeprefix('Authors:') if authors_tag else ''

    text = f"{title}\n\n{authors}\n\n{abstract}"
    return {'text': text, 'url': url.url, 'title': title, 'doc_type': 'arxiv'}


def youtube_loader(url: Url) -> Document:
    """Fetch YouTube transcript via youtube-transcript-api."""
    video_id = youtube_video_id(url.url)

    # Playlists don't have transcripts
    if video_id.startswith('playlist:'):
        logger.info(f"Skipping playlist URL: {url.url}")
        return None

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    text = ' '.join(snippet.text for snippet in transcript)

    # Try to get video title from oEmbed (no API key needed)
    title = url.url
    try:
        oembed = requests.get(
            f'https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json',
            timeout=10,
        ).json()
        title = oembed.get('title', url.url)
    except Exception:
        pass

    return {'text': text, 'url': url.url, 'title': title, 'doc_type': 'youtube'}


def pdf_loader(url: Url) -> Document:
    """Download PDF and extract text with pymupdf."""
    resp = requests.get(url.url, timeout=30)
    resp.raise_for_status()

    doc = fitz.open(stream=resp.content, filetype='pdf')
    pages = [page.get_text() for page in doc]
    text = '\n'.join(pages)
    doc.close()

    # Use filename as title fallback
    path = urlparse(url.url).path
    title = path.split('/')[-1] if '/' in path else url.url

    return {'text': text, 'url': url.url, 'title': title, 'doc_type': 'pdf'}


def github_loader(url: Url) -> Document:
    """Fetch raw README.md from a GitHub repo."""
    parsed = urlparse(url.url)
    parts = parsed.path.strip('/').split('/')
    assert len(parts) >= 2, f"Expected github.com/<owner>/<repo>, got {url.url}"
    owner, repo = parts[0], parts[1]

    raw_url = f'https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md'
    resp = requests.get(raw_url, timeout=15)
    resp.raise_for_status()

    title = f'{owner}/{repo}'
    return {'text': resp.text, 'url': url.url, 'title': title, 'doc_type': 'github'}


def scholar_loader(url: Url) -> Optional[Document]:
    """Scholar URLs are skipped (anti-scraping)."""
    logger.info(f"Skipping scholar URL: {url.url}")
    return None


if __name__ == '__main__':
    from extract_urls import parse_urls, resume_filepath

    urls = parse_urls(resume_filepath)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Test a few URLs of different types
    test_types = {'html', 'arxiv', 'youtube', 'pdf', 'github'}
    tested = set()
    for u in urls:
        if u.classification in tested or u.classification not in test_types:
            continue
        print(f"\n{'='*60}")
        print(f"[{u.classification}] {u.url}")
        print(f"{'='*60}")
        doc = load_document(u)
        if doc:
            print(f"Title: {doc['title']}")
            print(f"Text length: {len(doc['text'])} chars")
            print(f"First 300 chars:\n{doc['text'][:300]}")
        else:
            print("  -> Skipped/failed")
        tested.add(u.classification)
        if tested == test_types:
            break
