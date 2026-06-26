import re

from pydantic import BaseModel

Document = dict[str, str]

# Patterns that indicate a semantic boundary (ordered by strength)
_SECTION_BREAK = re.compile(r"\n\s*\n")  # blank line (paragraph break)
_HEADER_BREAK = re.compile(r"\n#{1,4}\s")  # markdown headers
_SENTENCE_END = re.compile(r"[.!?]\s")  # end of sentence


class Chunk(BaseModel):
    text: str
    source_url: str
    title: str
    doc_type: str
    chunk_index: int


def _find_semantic_break(text: str, target: int, window: int = 200) -> int:
    """Find the best semantic break point near `target` within ±window.

    Prefers (in order): paragraph breaks, markdown headers, sentence ends.
    Falls back to target if nothing found.
    """
    lo = max(0, target - window)
    hi = min(len(text), target + window)
    region = text[lo:hi]

    for pattern in (_SECTION_BREAK, _HEADER_BREAK, _SENTENCE_END):
        # Find all matches in the region, pick the one closest to target
        matches = list(pattern.finditer(region))
        if matches:
            best = min(matches, key=lambda m: abs((lo + m.start()) - target))
            return lo + best.start()

    return target


def chunk_document(
    doc: Document,
    chunk_size: int = 800,
    overlap: int = 200,
    semantic_breaks: bool = True,
) -> list[Chunk]:
    """Split a document into overlapping chunks with semantic boundary awareness.

    input: {'text': text, 'url': url.url, 'title': title, 'doc_type': 'html'}
    output: list of Chunk objects

    When semantic_breaks=True, chunk boundaries snap to nearby paragraph breaks,
    headers, or sentence endings instead of cutting mid-sentence.
    """
    if chunk_size < overlap:
        raise ValueError("chunk size must be greater than overlap")
    text = doc["text"]
    if not text.strip():
        return []

    chunk_list: list[Chunk] = []
    chunk_index = 0
    start = 0

    while start < len(text):
        raw_end = min(len(text), start + chunk_size)

        # Snap end to a semantic boundary (unless we're at the very end)
        if semantic_breaks and raw_end < len(text):
            end = _find_semantic_break(text, raw_end)
            # Don't let the break search shrink the chunk below half size
            if end <= start + chunk_size // 2:
                end = raw_end
        else:
            end = raw_end

        text_block = text[start:end].strip()
        if text_block:
            chunk_list.append(
                Chunk(
                    text=text_block,
                    source_url=doc["url"],
                    title=doc["title"],
                    doc_type=doc["doc_type"],
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        # Advance by (end - overlap), always move forward by at least chunk_size//2
        step = max(chunk_size // 2, end - overlap - start)
        start = start + step

    return chunk_list

if __name__ == "__main__":
    # Basic test with uniform text (no semantic boundaries)
    test_doc = {"text": "A" * 1000, "url": "test", "title": "test", "doc_type": "test"}
    chunks = chunk_document(test_doc, chunk_size=500, overlap=100)
    print(f"Uniform text — Chunks: {len(chunks)}")
    for c in chunks:
        print(f"  [{c.chunk_index}] len={len(c.text)}")

    # Test with semantic boundaries
    paragraphs = "\n\n".join([f"Paragraph {i}. " + "x " * 80 for i in range(10)])
    test_doc2 = {"text": paragraphs, "url": "test", "title": "test", "doc_type": "test"}
    chunks2 = chunk_document(test_doc2, chunk_size=500, overlap=100)
    print(f"\nParagraph text — Chunks: {len(chunks2)}")
    for c in chunks2:
        print(f"  [{c.chunk_index}] len={len(c.text):>4}  starts: {c.text[:50]}...")

    # Test empty doc
    empty_doc = {"text": "   ", "url": "test", "title": "test", "doc_type": "test"}
    assert chunk_document(empty_doc) == []
    print("\nEmpty doc: OK")
  