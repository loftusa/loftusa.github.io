"""
Exercise 2: Vector Store with ChromaDB

CONCEPT: persistent vector database — store, search, and retrieve text by meaning

BACKGROUND:
ChromaDB stores text alongside its embedding and lets you search by similarity.
It can auto-embed using sentence-transformers (default: all-MiniLM-L6-v2).

Key API:
    import chromadb

    client = chromadb.PersistentClient(path="some/directory")
    collection = client.get_or_create_collection("my_collection")

    # Add documents (Chroma embeds them automatically)
    collection.add(
        ids=["id1", "id2"],
        documents=["text one", "text two"],
        metadatas=[{"source": "foo"}, {"source": "bar"}],
    )

    # Search by text (Chroma embeds the query automatically)
    results = collection.query(query_texts=["my question"], n_results=3)
    # results["documents"][0]  -> list of matching texts
    # results["distances"][0]  -> list of L2 distances (lower = more similar)
    # results["metadatas"][0]  -> list of metadata dicts

    # You can also supply your own embeddings:
    collection.add(ids=..., documents=..., embeddings=[[0.1, 0.2, ...], ...])
    collection.query(query_embeddings=[[0.1, 0.2, ...]], n_results=3)

    collection.count()  -> number of documents stored

YOUR GOAL:
1. Create a persistent Chroma collection
2. Add these 8 documents about Alex's work (with metadata):
    - "Alex authored a 524-page textbook on statistical network ML with Cambridge University Press."
    - "NNsight and NDIF democratize access to open-weight foundation model internals, published at ICLR 2025."
    - "Alex organized the 200-person New England Mechanistic Interpretability conference and raised $17k in grants."
    - "The m2g pipeline transforms diffusion MRI scans into brain graphs, halving runtime and cutting cloud costs by 40%."
    - "Alex won $100k as part of a 4-person team in the Vesuvius Kaggle ink detection competition, featured on the cover of Scientific American."
    - "Graspologic is an open-source graph statistics library later adopted by Microsoft Research."
    - "At Creyon Bio, Alex developed a contrastive learning pipeline for drug toxicity prediction, increasing AUC from 0.73 to 0.88."
    - "Alex is a PhD student at Northeastern researching mechanistic interpretability, data attribution, and LLM evaluation."
3. Query with several strings and print results
4. Verify that the index persists: run the script twice and confirm
   the second run loads from disk instead of re-adding

BONUS: Try both approaches:
  A. Let Chroma auto-embed (just pass documents)
  B. Compute embeddings yourself with sentence-transformers and pass them explicitly
  Compare — they should give the same results since Chroma defaults to the same model.

SUCCESS CRITERIA:
- "brain connectivity MRI" returns the m2g doc
- "interpretability tools for LLMs" returns the NNsight doc
- "kaggle competition" returns the Vesuvius doc
- "drug toxicity machine learning" returns the Creyon Bio doc
- Running twice doesn't duplicate documents

RUN:
    uv run experiments/rag/ex2_vector_store.py
"""

import chromadb


docs = [
    "Alex authored a 524-page textbook on statistical network ML with Cambridge University Press.",
    "NNsight and NDIF democratize access to open-weight foundation model internals, published at ICLR 2025.",
    "Alex organized the 200-person New England Mechanistic Interpretability conference and raised $17k in grants.",
    "The m2g pipeline transforms diffusion MRI scans into brain graphs, halving runtime and cutting cloud costs by 40%.",
    "Alex won $100k as part of a 4-person team in the Vesuvius Kaggle ink detection competition, featured on the cover of Scientific American.",
    "Graspologic is an open-source graph statistics library later adopted by Microsoft Research.",
    "At Creyon Bio, Alex developed a contrastive learning pipeline for drug toxicity prediction, increasing AUC from 0.73 to 0.88.",
    "Alex is a PhD student at Northeastern researching mechanistic interpretability, data attribution, and LLM evaluation."
]
ids = [str(i) for i in range(len(docs))]

chroma_client = chromadb.PersistentClient(path="./database")
collection = chroma_client.get_or_create_collection(name="test_docs")
collection.add(ids=ids, documents=docs)

if __name__ == "__main__":
    query_texts = ["brain connectivity MRI", "democratized interpretability internals ndif for LLMs", "kaggle competition", "drug toxicity machine learning"]

    db_query = collection.query(query_texts=query_texts)

    for i, query in enumerate(query_texts):
        print(f"Query: {query}")
        print(f"results: {db_query['documents'][i][0]}")
        print(f"distances: {db_query['distances'][i][0]}")