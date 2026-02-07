"""
Solutions for RAG Learning Exercises

Reference implementations - use to check your work or if you get stuck.

IMPORTANT: Try to implement each exercise yourself first!
Learning happens through struggle, not copy-paste.
"""

import io
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import faiss
import numpy as np
import requests
from pypdf2 import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_DIM = 384


# =============================================================================
# Exercise 1: Embeddings
# =============================================================================


def load_embedding_model():
    """Load the sentence-transformers model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(model, text: str) -> np.ndarray:
    """Convert text to a vector."""
    return model.encode(text)


def embed_texts(model, texts: list[str]) -> np.ndarray:
    """Convert multiple texts to vectors."""
    return model.encode(texts)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


# =============================================================================
# Exercise 2: FAISS Index
# =============================================================================


def create_index(dimension: int):
    """Create a FAISS index."""
    return faiss.IndexFlatL2(dimension)


def add_vectors(index, vectors: np.ndarray):
    """Add vectors to the index."""
    vectors = vectors.astype(np.float32)
    index.add(vectors)


def search_index(index, query_vector: np.ndarray, k: int = 3):
    """Search for k nearest neighbors."""
    query = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query, k)
    return distances, indices


def save_index(index, path: Path):
    """Save index to disk."""
    faiss.write_index(index, str(path))


def load_index(path: Path):
    """Load index from disk."""
    return faiss.read_index(str(path))


# =============================================================================
# Exercise 3: Chunking
# =============================================================================


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_content = text[start:end]

        chunks.append(
            Chunk(
                text=chunk_text_content,
                source=source,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
            )
        )

        # Move forward, but not past the end
        start += chunk_size - chunk_overlap
        chunk_index += 1

        # Prevent infinite loop on tiny remaining text
        if start >= len(text):
            break

    return chunks


def chunk_text_by_sentences(
    text: str,
    source: str,
    sentences_per_chunk: int = 5,
    sentence_overlap: int = 1,
) -> list[Chunk]:
    """Split text into chunks by sentence boundaries."""
    # Split on sentence endings
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    chunk_index = 0
    i = 0

    while i < len(sentences):
        # Take sentences_per_chunk sentences
        end_i = min(i + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[i:end_i]
        chunk_text_content = " ".join(chunk_sentences)

        # Calculate character positions (approximate)
        start_char = len(" ".join(sentences[:i])) + (i if i > 0 else 0)
        end_char = start_char + len(chunk_text_content)

        chunks.append(
            Chunk(
                text=chunk_text_content,
                source=source,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
            )
        )

        # Move forward with overlap
        i += sentences_per_chunk - sentence_overlap
        chunk_index += 1

    return chunks


# =============================================================================
# Exercise 4: PDF Ingestion
# =============================================================================


def download_pdf(url: str) -> bytes:
    """Download PDF from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def clean_text(text: str) -> str:
    """Clean up extracted PDF text."""
    # Replace multiple whitespace with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip lines and remove very short ones
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    # Keep lines that have actual content
    lines = [line for line in lines if len(line) > 10 or line == ""]
    return "\n".join(lines)


def ingest_pdf(url: str) -> list[Chunk]:
    """Full pipeline: download, extract, clean, chunk."""
    pdf_bytes = download_pdf(url)
    text = extract_text_from_pdf(pdf_bytes)
    text = clean_text(text)
    return chunk_text(text, url)


# =============================================================================
# Exercise 5: RAG Index
# =============================================================================


@dataclass
class IndexedChunk:
    """A chunk with its position in the FAISS index."""

    text: str
    source: str
    chunk_index: int
    faiss_id: int


class RAGIndex:
    """Complete RAG indexing and retrieval system."""

    def __init__(self, index_path: Path = None, chunks_path: Path = None):
        """Initialize the RAG index."""
        self.model = load_embedding_model()
        self.index = create_index(EMBEDDING_DIM)
        self.chunks: list[IndexedChunk] = []
        self.index_path = index_path or DATA_DIR / "faiss.index"
        self.chunks_path = chunks_path or DATA_DIR / "chunks.json"

    def add_document(self, text: str, source: str):
        """Add a document to the index."""
        # Chunk the text
        chunks = chunk_text(text, source)

        # Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = embed_texts(self.model, texts)

        # Add to FAISS index
        start_id = len(self.chunks)
        add_vectors(self.index, embeddings)

        # Store chunk metadata
        for i, chunk in enumerate(chunks):
            self.chunks.append(
                IndexedChunk(
                    text=chunk.text,
                    source=source,
                    chunk_index=chunk.chunk_index,
                    faiss_id=start_id + i,
                )
            )

    def search(self, query: str, k: int = 3) -> list[tuple[IndexedChunk, float]]:
        """Search for relevant chunks."""
        # Embed query
        query_embedding = embed_texts(self.model, [query])

        # Search FAISS
        distances, indices = search_index(self.index, query_embedding[0], k)

        # Map indices back to chunks
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        return results

    def save(self):
        """Save index and chunks to disk."""
        DATA_DIR.mkdir(exist_ok=True)
        save_index(self.index, self.index_path)

        chunks_data = [asdict(c) for c in self.chunks]
        with open(self.chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2)

    def load(self):
        """Load index and chunks from disk."""
        self.index = load_index(self.index_path)

        with open(self.chunks_path, "r") as f:
            chunks_data = json.load(f)

        self.chunks = [IndexedChunk(**c) for c in chunks_data]


def fetch_github_readme(repo_url: str) -> str:
    """Fetch README from a GitHub repo."""
    # Parse repo URL
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]

    # Try main branch first, then master
    for branch in ["main", "master"]:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
        response = requests.get(raw_url, timeout=10)
        if response.status_code == 200:
            return response.text

    raise ValueError(f"Could not fetch README from {repo_url}")


# =============================================================================
# Exercise 6: Chat Integration
# =============================================================================


def build_rag_prompt(
    rag_index: RAGIndex,
    user_message: str,
    system_prompt: str,
    resume: str,
    k: int = 3,
    min_similarity: float = 0.3,
) -> list[dict[str, str]]:
    """Build a prompt with RAG context."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": resume},
    ]

    # Search for relevant context
    results = rag_index.search(user_message, k=k)

    # Format retrieved context
    if results:
        context_parts = []
        for chunk, distance in results:
            # Convert L2 distance to similarity score
            similarity = 1 / (1 + distance)
            if similarity >= min_similarity:
                context_parts.append(f"[Source: {chunk.source}]\n{chunk.text}")

        if context_parts:
            rag_context = "\n\n---\n\n".join(context_parts)
            messages.append(
                {
                    "role": "system",
                    "content": f"Additional context from Alex's work:\n\n{rag_context}",
                }
            )

    messages.append({"role": "user", "content": user_message})
    return messages
