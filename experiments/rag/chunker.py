
from dataclasses import dataclass

Document = dict[str, str]

@dataclass
class Chunk:
    text: str
    source_url: str
    title: str
    doc_type: str
    chunk_index: int

def chunk_document(doc: Document, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """
    input: {'text': text, 'url': url.url, 'title': title, 'doc_type': 'html'}
    output: list of chunks of the document, each overlapping by `overlap` characters
    """
    if chunk_size < overlap:
        raise ValueError("chunk size must be greater than overlap")
    text = doc['text']
    chunk_list = []
    chunk_index = 0
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        text_block = text[start:end]
        chunk = Chunk(
            text=text_block,
            source_url=doc['url'],
            title=doc['title'],
            doc_type=doc['doc_type'],
            chunk_index=chunk_index,
        )
        chunk_list.append(chunk)
        start += chunk_size - overlap
        chunk_index += 1
    return chunk_list

if __name__ == "__main__":
    test_doc = {'text': 'A' * 1000, 'url': 'test', 'title': 'test', 'doc_type': 'test'}
    chunks = chunk_document(test_doc, chunk_size=500, overlap=100)
    print(f"Chunks: {len(chunks)}")
    for c in chunks:
        print(f"  [{c.chunk_index}] start={c.chunk_index * 400}, len={len(c.text)}")
  