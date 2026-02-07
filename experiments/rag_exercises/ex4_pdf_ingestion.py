"""
Exercise 4: PDF Ingestion (extract content from real documents)

CONCEPT: Fetching URLs, PDF text extraction

BACKGROUND:
RAG needs text to embed. Your resume links to PDFs, GitHub repos, etc.
This exercise focuses on PDFs since they're common in academic contexts.

The pipeline:
1. Download PDF from URL
2. Extract text using pypdf2
3. Clean up the text (remove weird whitespace, etc.)

We'll use your actual thesis as the test document!

YOUR GOAL:
1. Download a PDF from a URL
2. Extract text from it
3. Clean up the extracted text
4. Chunk it using your ex3 code

RUN WHEN COMPLETE:
    uv run experiments/rag_exercises/ex4_pdf_ingestion.py

SUCCESS CRITERIA:
- PDF downloads successfully
- Text extraction produces readable content
- Ready to embed and index

WHAT YOU'RE LEARNING: How to get real content into your RAG system
"""

import tempfile
from pathlib import Path

# Import your chunking code from ex3
# (If you haven't completed ex3, you can use the solution)
try:
    from ex3_chunking import Chunk, chunk_text
except ImportError:
    # Fallback: simple chunking if ex3 not done
    from dataclasses import dataclass

    @dataclass
    class Chunk:
        text: str
        source: str
        chunk_index: int
        start_char: int
        end_char: int

    def chunk_text(text, source, chunk_size=500, chunk_overlap=100):
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(Chunk(text[start:end], source, idx, start, end))
            start += chunk_size - chunk_overlap
            idx += 1
        return chunks


def download_pdf(url: str) -> bytes:
    """Download PDF from URL.

    TODO: Use requests to download the PDF

    HINT:
        import requests
        response = requests.get(url)
        response.raise_for_status()  # Raise error if download failed
        return response.content

    Args:
        url: URL of the PDF

    Returns:
        PDF content as bytes
    """
    # TODO: Your code here
    pass


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes.

    TODO: Use pypdf2 to extract text

    HINT:
        import io
        from pypdf2 import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    Args:
        pdf_bytes: Raw PDF content

    Returns:
        Extracted text
    """
    # TODO: Your code here
    pass


def clean_text(text: str) -> str:
    """Clean up extracted PDF text.

    TODO: Remove extra whitespace and normalize

    PDF extraction often produces:
    - Multiple spaces
    - Weird line breaks
    - Headers/footers repeated

    HINT:
        import re
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove very short lines (often headers/page numbers)
        lines = text.split('\n')
        lines = [l for l in lines if len(l.strip()) > 20]
        return '\n'.join(lines)

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # TODO: Your code here
    pass


def ingest_pdf(url: str) -> list[Chunk]:
    """Full pipeline: download, extract, clean, chunk.

    TODO: Combine the functions above

    Args:
        url: URL of the PDF

    Returns:
        List of Chunk objects ready for embedding
    """
    # TODO: Your code here
    # 1. Download PDF
    # 2. Extract text
    # 3. Clean text
    # 4. Chunk text
    pass


def main():
    print("Exercise 4: PDF Ingestion\n")

    # Your thesis PDF
    thesis_url = "https://alex-loftus.com/files/submitted_thesis.pdf"

    # Alternative: NNsight paper (smaller)
    # paper_url = "https://arxiv.org/pdf/2407.14561.pdf"

    print(f"Downloading: {thesis_url}")

    # Step 1: Download
    pdf_bytes = download_pdf(thesis_url)
    if pdf_bytes is None:
        print("ERROR: download_pdf() returned None. Implement it!")
        return

    print(f"Downloaded {len(pdf_bytes):,} bytes")

    # Step 2: Extract text
    print("\nExtracting text...")
    raw_text = extract_text_from_pdf(pdf_bytes)
    if raw_text is None:
        print("ERROR: extract_text_from_pdf() returned None. Implement it!")
        return

    print(f"Extracted {len(raw_text):,} characters")
    print(f"\nFirst 500 characters (raw):")
    print("-" * 40)
    print(raw_text[:500])
    print("-" * 40)

    # Step 3: Clean text
    print("\nCleaning text...")
    clean = clean_text(raw_text)
    if clean is None:
        print("ERROR: clean_text() returned None. Implement it!")
        return

    print(f"Cleaned text: {len(clean):,} characters")
    print(f"\nFirst 500 characters (cleaned):")
    print("-" * 40)
    print(clean[:500])
    print("-" * 40)

    # Step 4: Chunk
    print("\nChunking text...")
    chunks = chunk_text(clean, thesis_url, chunk_size=500, chunk_overlap=100)

    print(f"Created {len(chunks)} chunks")
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(chunk.text[:200] + "...")

    # Step 5: Full pipeline test
    print("\n" + "=" * 60)
    print("Testing full pipeline with ingest_pdf()...")

    chunks2 = ingest_pdf(thesis_url)
    if chunks2 is None:
        print("ERROR: ingest_pdf() returned None. Implement it!")
        return

    print(f"Full pipeline produced {len(chunks2)} chunks")

    # Save a sample for inspection
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    sample_file = data_dir / "thesis_sample_chunks.txt"

    with open(sample_file, "w") as f:
        for i, chunk in enumerate(chunks2[:10]):
            f.write(f"=== CHUNK {i} ===\n")
            f.write(f"Source: {chunk.source}\n")
            f.write(f"Chars {chunk.start_char}-{chunk.end_char}\n")
            f.write(chunk.text)
            f.write("\n\n")

    print(f"\nSaved first 10 chunks to: {sample_file}")
    print("\nYou can now embed these chunks!")


if __name__ == "__main__":
    main()
