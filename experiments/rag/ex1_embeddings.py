"""
Exercise 1: Embeddings

CONCEPT: sentence-transformers, vector representations, cosine similarity

BACKGROUND:
Embeddings convert text to vectors (lists of 384 floats). Similar meaning =
similar vectors. This is the foundation of all vector search.

    model: all-MiniLM-L6-v2  (already installed via sentence-transformers)
    output dimension: 384

YOUR GOAL:
1. Load the embedding model
2. Embed these 5 sentences
3. Compute pairwise cosine similarity
4. Print a similarity matrix or table showing which sentences cluster together

TEST SENTENCES:
    "I research neural network interpretability."
    "My work focuses on understanding how AI models reason."
    "Mechanistic interpretability reverse-engineers circuits."
    "I like to eat pizza on Fridays."
    "The weather in Baltimore is rainy today."

SUCCESS CRITERIA:
- Sentences 0-2 (interpretability) should have pairwise similarity > 0.5
- Sentences 3-4 (pizza, weather) should have similarity < 0.3 with 0-2
- You can explain what shape the embeddings are and why

USEFUL DOCS:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.encode(["text1", "text2"])  -> np.ndarray of shape (2, 384)

    cosine similarity = (a · b) / (||a|| * ||b||)
    np.dot, np.linalg.norm

RUN:
    uv run experiments/rag/ex1_embeddings.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import seaborn as sns

def load_model(model_name="all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

def embed_text(model: SentenceTransformer, text: list[str]) -> np.ndarray:
    return model.encode(text)

if __name__ == "__main__":
    test_sentences = [
    "I research neural network interpretability.",
    "My work focuses on understanding how AI models reason.",
    "Mechanistic interpretability reverse-engineers circuits.",
    "I like to eat pizza on Fridays.",
    "The weather in Baltimore is rainy today.",
    ]

    model = load_model()
    embeddings = embed_text(model, test_sentences)  # (B, D)

    # Usually would have to normalize: M[i,j] /= (norm(E[i, :]) * norm(E[j, :]))
    # but `embed_text` produces vectors with norm 1
    embeddings_norm = norm(embeddings, axis=1, keepdims=True)  # (B, 1)
    similarity_matrix = (embeddings @ embeddings.T) / (embeddings_norm * embeddings_norm.T)  # (B, B)

    sns.heatmap(similarity_matrix, annot=True, fmt='.2f')
    print("\n".join(test_sentences))
