"""
Exercise 2: FAISS Index (efficient vector search)

CONCEPT: FAISS basics - create, add, search, save, load

BACKGROUND:
FAISS (Facebook AI Similarity Search) stores vectors and lets you find
the most similar ones to a query. Key operations:
- Create an index for vectors of dimension d
- Add vectors to the index
- Search for k nearest neighbors
- Save/load index to disk

Index types:
- IndexFlatL2: Brute-force, exact (we use this - simple and accurate)
- IndexIVFFlat: Faster but approximate (for millions of vectors)
- IndexHNSW: Graph-based, very fast (for production scale)

YOUR GOAL:
1. Create a FAISS index
2. Add some vectors
3. Search for nearest neighbors
4. Save and load the index

RUN WHEN COMPLETE:
    uv run experiments/rag_exercises/ex2_faiss_index.py

SUCCESS CRITERIA:
- Search returns correct nearest neighbors
- Index saves and loads correctly
- You understand add vs search operations

WHAT YOU'RE LEARNING: The data structure that makes RAG fast
"""

import numpy as np
from pathlib import Path

# Dimension of our embeddings (all-MiniLM-L6-v2 outputs 384-dim vectors)
EMBEDDING_DIM = 384
DATA_DIR = Path(__file__).parent / "data"


def create_index(dimension: int):
    """Create a FAISS index for vectors of given dimension.

    TODO: Create an IndexFlatL2 index

    HINT:
        import faiss
        index = faiss.IndexFlatL2(dimension)

    Args:
        dimension: Size of vectors (384 for our embeddings)

    Returns:
        FAISS index
    """
    # TODO: Your code here
    pass


def add_vectors(index, vectors: np.ndarray):
    """Add vectors to the index.

    TODO: Use index.add() to add vectors

    IMPORTANT: FAISS requires float32 arrays!

    HINT:
        vectors = vectors.astype(np.float32)
        index.add(vectors)

    Args:
        index: FAISS index
        vectors: numpy array of shape (n_vectors, dimension)
    """
    # TODO: Your code here
    pass


def search_index(index, query_vector: np.ndarray, k: int = 3):
    """Search for k nearest neighbors.

    TODO: Use index.search() to find nearest neighbors

    HINT:
        # Query must be 2D: (n_queries, dimension)
        query = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = index.search(query, k)

    Args:
        index: FAISS index
        query_vector: Vector to search for, shape (dimension,)
        k: Number of neighbors to return

    Returns:
        Tuple of (distances, indices)
        - distances: shape (1, k) - L2 distances to neighbors
        - indices: shape (1, k) - indices of neighbors in the index
    """
    # TODO: Your code here
    pass


def save_index(index, path: Path):
    """Save index to disk.

    TODO: Use faiss.write_index()

    HINT: faiss.write_index(index, str(path))

    Args:
        index: FAISS index
        path: Where to save
    """
    # TODO: Your code here
    pass


def load_index(path: Path):
    """Load index from disk.

    TODO: Use faiss.read_index()

    HINT: index = faiss.read_index(str(path))

    Args:
        path: Where to load from

    Returns:
        FAISS index
    """
    # TODO: Your code here
    pass


def main():
    print("Exercise 2: FAISS Index\n")

    # Step 1: Create some fake "document" vectors
    # In real RAG, these would come from sentence-transformers
    print("Creating sample vectors...")
    np.random.seed(42)

    # Simulate 5 document embeddings
    doc_vectors = np.random.randn(5, EMBEDDING_DIM).astype(np.float32)

    # Make documents 0 and 2 similar (for testing)
    doc_vectors[2] = doc_vectors[0] + np.random.randn(EMBEDDING_DIM) * 0.1

    doc_labels = [
        "Doc 0: About interpretability research",
        "Doc 1: About pizza recipes",
        "Doc 2: About neural network analysis (similar to Doc 0)",
        "Doc 3: About weather forecasting",
        "Doc 4: About machine learning basics",
    ]

    # Step 2: Create index
    print(f"Creating FAISS index (dimension={EMBEDDING_DIM})...")
    index = create_index(EMBEDDING_DIM)
    if index is None:
        print("ERROR: create_index() returned None. Implement it!")
        return
    print(f"Index created: {index}\n")

    # Step 3: Add vectors
    print("Adding vectors to index...")
    add_vectors(index, doc_vectors)
    print(f"Index now contains {index.ntotal} vectors\n")

    if index.ntotal == 0:
        print("ERROR: Index is empty. Implement add_vectors()!")
        return

    # Step 4: Search
    print("Searching for neighbors of Doc 0...")
    query = doc_vectors[0]  # Query with Doc 0's vector
    result = search_index(index, query, k=3)
    if result is None:
        print("ERROR: search_index() returned None. Implement it!")
        return

    distances, indices = result
    print(f"Top 3 results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  {i + 1}. {doc_labels[idx]} (distance: {dist:.4f})")

    print()
    print("EXPECTED: Doc 0 (exact match, dist=0) and Doc 2 (similar) should be top 2")
    print()

    # Step 5: Save and load
    DATA_DIR.mkdir(exist_ok=True)
    index_path = DATA_DIR / "test_faiss.index"

    print(f"Saving index to {index_path}...")
    save_index(index, index_path)

    if not index_path.exists():
        print("ERROR: Index file not created. Implement save_index()!")
        return
    print(f"Index saved ({index_path.stat().st_size} bytes)")

    print("Loading index from disk...")
    loaded_index = load_index(index_path)
    if loaded_index is None:
        print("ERROR: load_index() returned None. Implement it!")
        return

    print(f"Loaded index contains {loaded_index.ntotal} vectors")

    # Verify loaded index works
    distances2, indices2 = search_index(loaded_index, query, k=1)
    if indices2[0][0] == 0:
        print("SUCCESS: Loaded index returns correct results!\n")
    else:
        print("ERROR: Loaded index returns wrong results")

    # Cleanup
    index_path.unlink()
    print("Test index deleted.")
    print()
    print("You now understand FAISS basics!")


if __name__ == "__main__":
    main()
