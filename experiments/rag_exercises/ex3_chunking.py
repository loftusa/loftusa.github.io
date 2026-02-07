"""
Exercise 3: Chunking (split documents into pieces)

CONCEPT: Text splitting, chunk size, overlap

BACKGROUND:
Why chunk documents?
1. Embedding models have token limits (~256-512 tokens)
2. Smaller chunks = more precise retrieval
3. But too small = lose context

Chunking strategy:
- Split on natural boundaries (paragraphs, sentences)
- Use overlap so chunks share context at boundaries
- Keep metadata (source, position) for each chunk

Example:
    Original: "Sentence 1. Sentence 2. Sentence 3. Sentence 4."

    Chunks (size=2 sentences, overlap=1):
    - Chunk 0: "Sentence 1. Sentence 2."
    - Chunk 1: "Sentence 2. Sentence 3."  (overlaps with chunk 0)
    - Chunk 2: "Sentence 3. Sentence 4."  (overlaps with chunk 1)

YOUR GOAL:
1. Implement a simple text chunker
2. Split text into overlapping chunks
3. Preserve source metadata

RUN WHEN COMPLETE:
    uv run experiments/rag_exercises/ex3_chunking.py

SUCCESS CRITERIA:
- Chunks are roughly the target size
- Overlap works correctly
- No text is lost

WHAT YOU'RE LEARNING: How to prepare documents for embedding
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    text: str
    source: str  # Where this chunk came from (URL, filename)
    chunk_index: int  # Position in original document
    start_char: int  # Character offset in original
    end_char: int  # Character offset in original


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Split text into overlapping chunks.

    TODO: Implement chunking with overlap

    ALGORITHM:
    1. Start at position 0
    2. Take chunk_size characters
    3. Move forward by (chunk_size - chunk_overlap)
    4. Repeat until end of text

    HINT:
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_index=chunk_index,
                start_char=start,
                end_char=min(end, len(text))
            ))

            start += chunk_size - chunk_overlap
            chunk_index += 1

    Args:
        text: Full document text
        source: Source identifier (URL, filename)
        chunk_size: Target size in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of Chunk objects
    """
    # TODO: Your code here
    pass


def chunk_text_by_sentences(
    text: str,
    source: str,
    sentences_per_chunk: int = 5,
    sentence_overlap: int = 1,
) -> list[Chunk]:
    """Split text into chunks by sentence boundaries.

    TODO: Implement sentence-based chunking

    This is better than character-based because it respects
    natural language boundaries.

    ALGORITHM:
    1. Split text into sentences (on '. ')
    2. Group sentences into chunks
    3. Add overlap

    HINT:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Then group sentences similar to character chunking

    Args:
        text: Full document text
        source: Source identifier
        sentences_per_chunk: How many sentences per chunk
        sentence_overlap: How many sentences overlap

    Returns:
        List of Chunk objects
    """
    # TODO: Your code here (BONUS - implement if you have time)
    pass


def main():
    print("Exercise 3: Chunking\n")

    # Sample document
    sample_text = """
Machine learning interpretability is the study of understanding how AI models make decisions.
This field has become increasingly important as models are deployed in high-stakes applications.
Researchers use various techniques to peer inside neural networks.

One approach is mechanistic interpretability, which aims to reverse-engineer the computations
performed by individual neurons and circuits. This can reveal surprising structure in how
models process information.

Another approach is feature visualization, which generates inputs that maximally activate
specific neurons. This helps researchers understand what patterns the model has learned to detect.

The field continues to grow as models become more powerful and their decisions more consequential.
Understanding these systems is crucial for ensuring they behave safely and as intended.
    """.strip()

    source = "https://example.com/interpretability.txt"

    # Step 1: Character-based chunking
    print("=== Character-based Chunking ===")
    print(f"Original text length: {len(sample_text)} characters\n")

    chunks = chunk_text(sample_text, source, chunk_size=300, chunk_overlap=50)

    if chunks is None:
        print("ERROR: chunk_text() returned None. Implement it!")
        return

    print(f"Created {len(chunks)} chunks:\n")

    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index}:")
        print(f"  Characters {chunk.start_char}-{chunk.end_char}")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Preview: '{chunk.text[:60]}...'")
        print()

    # Verify overlap
    if len(chunks) >= 2:
        overlap_text = sample_text[chunks[1].start_char : chunks[0].end_char]
        print(f"Overlap between chunk 0 and 1: {len(overlap_text)} chars")
        print(f"Overlap text: '{overlap_text[:50]}...'\n")

    # Verify no text lost
    reconstructed = ""
    for i, chunk in enumerate(chunks):
        if i == 0:
            reconstructed = chunk.text
        else:
            # Add only the non-overlapping part
            overlap_size = chunks[i - 1].end_char - chunk.start_char
            reconstructed += chunk.text[overlap_size:]

    if len(reconstructed) >= len(sample_text) - 10:  # Allow small difference
        print("SUCCESS: No text lost in chunking!")
    else:
        print(f"WARNING: Reconstructed length ({len(reconstructed)}) < original ({len(sample_text)})")

    # Step 2: Sentence-based chunking (BONUS)
    print("\n=== Sentence-based Chunking (BONUS) ===")

    sentence_chunks = chunk_text_by_sentences(
        sample_text, source, sentences_per_chunk=3, sentence_overlap=1
    )

    if sentence_chunks is None:
        print("Sentence-based chunking not implemented yet (optional)")
    else:
        print(f"Created {len(sentence_chunks)} sentence-based chunks")
        for chunk in sentence_chunks[:2]:
            print(f"\nChunk {chunk.chunk_index}:")
            print(f"  '{chunk.text[:80]}...'")

    print("\n" + "=" * 60)
    print("You now understand chunking!")
    print("\nKey insight: Overlap ensures context isn't lost at boundaries.")


if __name__ == "__main__":
    main()
