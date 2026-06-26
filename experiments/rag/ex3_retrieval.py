"""
Exercise 3: End-to-End Retrieval

CONCEPT: Ingest real documents → chunk → store in Chroma → search

BACKGROUND:
This connects YOUR existing code to the vector store:

    extract_urls.py  →  list of Url objects from resume.txt
    loaders.py       →  load_document(url) → {"text", "url", "title", "doc_type"}
    chunker.py       →  chunk_document(doc) → list of Chunk objects

The pipeline:
    resume.txt → extract URLs → fetch each → chunk → store in Chroma
                                                         ↑
                                           query text ───┘ → top-K chunks

YOUR GOAL:
1. Parse all URLs from resume.txt (extract_urls.parse_urls)
2. Load each URL's content (loaders.load_document)
3. Chunk each document (chunker.chunk_document)
4. Store all chunks in a persistent Chroma collection, with metadata
   (source_url, title, doc_type, chunk_index)
5. Support a --query flag to search the index
6. Support a --rebuild flag to re-ingest from scratch

It should print how many chunks came from each URL during ingestion,
and show the top-K results with source info for queries.

IMPORTS YOU'LL NEED:
    from extract_urls import parse_urls, resume_filepath
    from loaders import load_document
    from chunker import chunk_document

TIPS:
- load_document() returns None for URLs it can't/won't load (scholar, etc.)
- Give each chunk a unique ID like f"chunk_{i}"
- Store metadata so you can see WHERE a result came from
- First run will be slow (fetching URLs). Second run should load instantly.

SUCCESS CRITERIA:
- Ingestion loads 20+ URLs and produces hundreds of chunks
- "what is the thesis about?" returns chunks from the thesis PDF
- "NNsight" returns chunks from the NNsight paper
- "m2g pipeline" returns chunks from the m2g GitHub README
- Second run says "Loaded existing index" and skips ingestion

RUN:
    uv run experiments/rag/ex3_retrieval.py
    uv run experiments/rag/ex3_retrieval.py --query "connectome analysis"
    uv run experiments/rag/ex3_retrieval.py --rebuild
"""
import shutil

import chromadb
import click
from pathlib import Path
from tqdm import tqdm

from extract_urls import parse_urls, Url
from loaders import load_document
from chunker import chunk_document, Chunk

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_rag"
RESUME_FILEPATH = Path(__file__).parent.parent / "resume.txt"


def ingest(collection: chromadb.Collection) -> None:
    """Ingest all resume URLs into the Chroma collection."""
    urls = parse_urls(RESUME_FILEPATH)
    chunk_id = 0

    for url in tqdm(urls, desc="Ingesting URLs"):
        doc: dict[str, str] | None = load_document(url)
        if doc is None:
            continue
        chunks: list[Chunk] = chunk_document(doc)
        if not chunks:
            continue

        chunk_docs = [c.text for c in chunks]
        chunk_ids = [f"chunk_{chunk_id + i}" for i in range(len(chunks))]
        chunk_metadatas = [c.model_dump(exclude={"text"}) for c in chunks]

        collection.add(
            ids=chunk_ids,
            documents=chunk_docs,
            metadatas=chunk_metadatas,
        )
        chunk_id += len(chunks)
        print(
            f"  [{url.classification:>8}] {len(chunks):>3} chunks  "
            f"{doc['title'][:60]}"
        )

    print(f"\nTotal: {collection.count()} chunks")


def search(collection: chromadb.Collection, query: str, n_results: int = 5) -> None:
    """Search the collection and print results."""
    results = collection.query(query_texts=[query], n_results=n_results)

    print(f'\nQuery: "{query}"\n')
    for i, (doc, dist, meta) in enumerate(
        zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ):
        sim = 1 / (1 + dist)
        print(
            f"{i + 1}. (sim={sim:.3f}) [{meta.get('doc_type')}] "
            f"{meta.get('title', '?')}"
        )
        print(f"   {doc[:150]}…\n")


@click.command()
@click.option("--query", default=None, help="Search query to run against the index.")
@click.option("--rebuild", is_flag=True, help="Delete existing index and re-ingest.")
def main(query: str | None, rebuild: bool) -> None:
    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    DATA_DIR.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="resume_rag")

    if collection.count() == 0:
        print("Ingesting all resume URLs …\n")
        ingest(collection)
    else:
        print(f"Loaded existing index: {collection.count()} chunks")

    # Run query (or default demo queries)
    queries = (
        [query]
        if query
        else [
            "what is the thesis about?",
            "NNsight",
            "m2g pipeline",
        ]
    )
    for q in queries:
        search(collection, q)


if __name__ == "__main__":
    main()
