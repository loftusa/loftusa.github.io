"""
Exercise 5: End-to-End Retrieval (put it all together)

CONCEPT: Full RAG pipeline - ingest, embed, index, search

BACKGROUND:
Now we combine everything:
1. Ingest content (PDFs, GitHub READMEs)
2. Chunk the content
3. Embed all chunks
4. Build FAISS index
5. Search with a query

This is the complete offline + online pipeline!

YOUR GOAL:
1. Build a RAGIndex class that handles the full pipeline
2. Ingest your thesis + one other document
3. Search and verify results are relevant

RUN WHEN COMPLETE:
    uv run experiments/rag_exercises/ex5_retrieval.py

    # With custom query:
    uv run experiments/rag_exercises/ex5_retrieval.py --query "what is interpretability"

SUCCESS CRITERIA:
- Ingestion works for multiple documents
- Search returns relevant chunks
- Index saves/loads correctly

WHAT YOU'RE LEARNING: The complete RAG retrieval system
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Import from previous exercises (or use solutions)
try:
    from ex1_embeddings import load_embedding_model, embed_texts
    from ex2_faiss_index import create_index, add_vectors, search_index, save_index, load_index
    from ex3_chunking import Chunk, chunk_text
    from ex4_pdf_ingestion import download_pdf, extract_text_from_pdf, clean_text
except ImportError:
    print("Warning: Could not import from previous exercises.")
    print("Make sure ex1-ex4 are implemented, or use solutions.py")
    raise

DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_DIM = 384


@dataclass
class IndexedChunk:
    """A chunk with its position in the FAISS index."""

    text: str
    source: str
    chunk_index: int
    faiss_id: int  # Position in FAISS index


class RAGIndex:
    """Complete RAG indexing and retrieval system."""

    def __init__(self, index_path: Path = None, chunks_path: Path = None):
        """Initialize the RAG index.

        TODO: Initialize components

        HINT:
            self.model = load_embedding_model()
            self.index = create_index(EMBEDDING_DIM)
            self.chunks: list[IndexedChunk] = []
            self.index_path = index_path or DATA_DIR / "faiss.index"
            self.chunks_path = chunks_path or DATA_DIR / "chunks.json"

        Args:
            index_path: Where to save/load FAISS index
            chunks_path: Where to save/load chunk metadata
        """
        # TODO: Your code here
        pass

    def add_document(self, text: str, source: str):
        """Add a document to the index.

        TODO: Chunk, embed, and add to index

        ALGORITHM:
        1. Chunk the text
        2. Embed all chunks
        3. Add embeddings to FAISS index
        4. Store chunk metadata

        HINT:
            chunks = chunk_text(text, source)
            texts = [c.text for c in chunks]
            embeddings = embed_texts(self.model, texts)

            start_id = len(self.chunks)
            add_vectors(self.index, embeddings)

            for i, chunk in enumerate(chunks):
                self.chunks.append(IndexedChunk(
                    text=chunk.text,
                    source=source,
                    chunk_index=chunk.chunk_index,
                    faiss_id=start_id + i
                ))

        Args:
            text: Full document text
            source: Source identifier (URL)
        """
        # TODO: Your code here
        pass

    def search(self, query: str, k: int = 3) -> list[tuple[IndexedChunk, float]]:
        """Search for relevant chunks.

        TODO: Embed query and search FAISS

        HINT:
            query_embedding = embed_texts(self.model, [query])
            distances, indices = search_index(self.index, query_embedding[0], k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunks):
                    results.append((self.chunks[idx], float(dist)))
            return results

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (chunk, distance) tuples, sorted by relevance
        """
        # TODO: Your code here
        pass

    def save(self):
        """Save index and chunks to disk.

        TODO: Save FAISS index and chunk metadata

        HINT:
            DATA_DIR.mkdir(exist_ok=True)
            save_index(self.index, self.index_path)

            chunks_data = [asdict(c) for c in self.chunks]
            with open(self.chunks_path, 'w') as f:
                json.dump(chunks_data, f)
        """
        # TODO: Your code here
        pass

    def load(self):
        """Load index and chunks from disk.

        TODO: Load FAISS index and chunk metadata

        HINT:
            self.index = load_index(self.index_path)
            with open(self.chunks_path, 'r') as f:
                chunks_data = json.load(f)
            self.chunks = [IndexedChunk(**c) for c in chunks_data]
        """
        # TODO: Your code here
        pass


def fetch_github_readme(repo_url: str) -> str:
    """Fetch README from a GitHub repo.

    TODO: Convert repo URL to raw README URL and fetch

    EXAMPLE:
        https://github.com/neurodata/m2g
        -> https://raw.githubusercontent.com/neurodata/m2g/main/README.md

    HINT:
        import requests
        # Parse repo URL
        parts = repo_url.rstrip('/').split('/')
        owner, repo = parts[-2], parts[-1]

        # Try main branch first, then master
        for branch in ['main', 'master']:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
            response = requests.get(raw_url)
            if response.status_code == 200:
                return response.text

        raise ValueError(f"Could not fetch README from {repo_url}")

    Args:
        repo_url: GitHub repository URL

    Returns:
        README content as string
    """
    # TODO: Your code here
    pass


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Demo")
    parser.add_argument("--query", type=str, default="what is interpretability research about?")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    args = parser.parse_args()

    print("Exercise 5: End-to-End Retrieval\n")

    # Initialize index
    rag = RAGIndex()
    if rag.model is None:
        print("ERROR: RAGIndex.__init__() not implemented correctly!")
        return

    # Check if we have a saved index
    index_exists = (DATA_DIR / "faiss.index").exists() and (DATA_DIR / "chunks.json").exists()

    if index_exists and not args.rebuild:
        print("Loading existing index...")
        rag.load()
        print(f"Loaded {len(rag.chunks)} chunks from disk")
    else:
        print("Building new index...")

        # Document sources
        sources = [
            ("pdf", "https://alex-loftus.com/files/submitted_thesis.pdf"),
            ("github", "https://github.com/neurodata/m2g"),
        ]

        for doc_type, url in sources:
            print(f"\nIngesting {doc_type}: {url}")

            try:
                if doc_type == "pdf":
                    pdf_bytes = download_pdf(url)
                    text = extract_text_from_pdf(pdf_bytes)
                    text = clean_text(text)
                elif doc_type == "github":
                    text = fetch_github_readme(url)
                else:
                    print(f"Unknown type: {doc_type}")
                    continue

                print(f"  Extracted {len(text):,} characters")
                rag.add_document(text, url)
                print(f"  Index now has {len(rag.chunks)} chunks")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        # Save the index
        print("\nSaving index...")
        rag.save()
        print(f"Saved to {DATA_DIR}")

    # Search demo
    print("\n" + "=" * 60)
    print(f"SEARCH QUERY: '{args.query}'")
    print("=" * 60)

    results = rag.search(args.query, k=5)

    if results is None:
        print("ERROR: search() returned None. Implement it!")
        return

    print(f"\nTop {len(results)} results:\n")

    for i, (chunk, distance) in enumerate(results):
        # Lower distance = more similar for L2
        similarity = 1 / (1 + distance)  # Convert to similarity score
        print(f"--- Result {i + 1} (similarity: {similarity:.3f}) ---")
        print(f"Source: {chunk.source}")
        print(f"Text preview:")
        print(f"  {chunk.text[:300]}...")
        print()

    print("=" * 60)
    print("Try different queries:")
    print("  uv run experiments/rag_exercises/ex5_retrieval.py --query 'connectome'")
    print("  uv run experiments/rag_exercises/ex5_retrieval.py --query 'graph neural network'")
    print("  uv run experiments/rag_exercises/ex5_retrieval.py --query 'm2g pipeline'")


if __name__ == "__main__":
    main()
