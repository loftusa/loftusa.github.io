"""
Exercise 1: Embeddings (understand vector representations)

CONCEPT: Sentence-transformers, embeddings, cosine similarity

BACKGROUND:
Embeddings convert text into vectors (lists of numbers). The key property:
texts with similar meaning have similar vectors (high cosine similarity).

The model we use (all-MiniLM-L6-v2) outputs 384-dimensional vectors.
It's small, fast, and good enough for most RAG applications.

YOUR GOAL:
1. Load the sentence-transformers model
2. Embed a few sentences
3. Compute cosine similarity between them
4. Observe that similar sentences have higher similarity

RUN WHEN COMPLETE:
    uv run experiments/rag_exercises/ex1_embeddings.py

SUCCESS CRITERIA:
- Similar sentences have similarity > 0.7
- Dissimilar sentences have similarity < 0.3
- You understand what an embedding IS

WHAT YOU'RE LEARNING: The foundation of all vector search
"""

import numpy as np


def load_embedding_model():
    """Load the sentence-transformers model.

    TODO: Import SentenceTransformer and load 'all-MiniLM-L6-v2'

    HINT:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

    Returns:
        SentenceTransformer model
    """
    # TODO: Your code here
    pass


def embed_text(model, text: str) -> np.ndarray:
    """Convert text to a vector.

    TODO: Use model.encode() to get the embedding

    HINT: model.encode(text) returns a numpy array

    Args:
        model: SentenceTransformer model
        text: String to embed

    Returns:
        numpy array of shape (384,)
    """
    # TODO: Your code here
    pass


def embed_texts(model, texts: list[str]) -> np.ndarray:
    """Convert multiple texts to vectors (batched for efficiency).

    TODO: Use model.encode() with a list of texts

    HINT: model.encode(texts) returns shape (n_texts, 384)

    Args:
        model: SentenceTransformer model
        texts: List of strings to embed

    Returns:
        numpy array of shape (n_texts, 384)
    """
    # TODO: Your code here
    pass


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    TODO: Implement cosine similarity

    Formula: cos_sim = (a · b) / (||a|| * ||b||)

    HINT:
        - np.dot(a, b) gives dot product
        - np.linalg.norm(a) gives magnitude

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (float between -1 and 1, usually 0 to 1 for text)
    """
    # TODO: Your code here
    pass


def main():
    print("Exercise 1: Embeddings\n")

    # Step 1: Load the model
    print("Loading model...")
    model = load_embedding_model()
    if model is None:
        print("ERROR: load_embedding_model() returned None. Implement it!")
        return
    print(f"Model loaded: {model}\n")

    # Step 2: Define test sentences
    sentences = [
        "I research neural network interpretability.",
        "My work focuses on understanding how AI models work internally.",
        "I like to eat pizza on Fridays.",
        "The weather is nice today.",
        "Mechanistic interpretability studies neural network internals.",
    ]

    # Step 3: Embed all sentences
    print("Embedding sentences...")
    embeddings = embed_texts(model, sentences)
    if embeddings is None:
        print("ERROR: embed_texts() returned None. Implement it!")
        return
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Each embedding has {embeddings.shape[1]} dimensions\n")

    # Step 4: Compute pairwise similarities
    print("Pairwise cosine similarities:")
    print("-" * 60)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim is None:
                print("ERROR: cosine_similarity() returned None. Implement it!")
                return

            # Highlight high similarity pairs
            marker = " <-- SIMILAR!" if sim > 0.5 else ""
            print(f"[{i}] vs [{j}]: {sim:.3f}{marker}")
            print(f"    '{sentences[i][:40]}...'")
            print(f"    '{sentences[j][:40]}...'")
            print()

    # Step 5: Verify understanding
    print("-" * 60)
    print("EXPECTED RESULTS:")
    print("- Sentences 0, 1, 4 should be similar (all about interpretability)")
    print("- Sentences 2, 3 should be dissimilar from 0, 1, 4")
    print()
    print("If you see this pattern, you understand embeddings!")


if __name__ == "__main__":
    main()
