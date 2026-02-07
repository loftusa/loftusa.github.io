# RAG Learning Exercises

Learn Retrieval-Augmented Generation by building a system that indexes your resume's linked content.

## The Goal

By the end of these exercises, you'll have built a RAG system that:
1. Embeds text into vectors using sentence-transformers
2. Stores and searches vectors with FAISS
3. Extracts content from PDFs, GitHub, and web pages
4. Retrieves relevant context for chat queries
5. Integrates with your existing chat API

## Core Mental Model

```
RAG Flow:
1. OFFLINE: Documents → Chunks → Embeddings → FAISS Index (saved to disk)
2. ONLINE:  Query → Embedding → FAISS Search → Top-K Chunks → Inject into Prompt → LLM
```

Key concepts:
- **Embeddings**: Convert text to vectors (384 floats). Similar meaning = similar vectors.
- **FAISS Index**: Data structure for fast nearest-neighbor search
- **Chunking**: Split long documents into smaller pieces (~500 chars)
- **Retrieval**: Find the chunks most similar to the user's query

## Exercise Progression

| Exercise | Concept | What You Build |
|----------|---------|----------------|
| ex1_embeddings.py | sentence-transformers | Embed text, compute similarity |
| ex2_faiss_index.py | FAISS basics | Create, add, search, save/load index |
| ex3_chunking.py | Text splitting | Chunk documents with overlap |
| ex4_pdf_ingestion.py | Content extraction | Fetch and parse your thesis PDF |
| ex5_retrieval.py | End-to-end retrieval | Full pipeline: ingest → index → search |
| **ex6_chat_integration.py** | **Production RAG** | **Integrate with chat_api.py** |

## How to Work Through

1. Open an exercise file
2. Read the docstring for context and concepts
3. Fill in the TODOs (hints provided)
4. Run to verify: `uv run experiments/rag_exercises/ex1_embeddings.py`
5. Move to the next exercise

## Prototype URLs

Start with these 3 URLs to keep iteration fast:
1. **Thesis**: `https://alex-loftus.com/files/submitted_thesis.pdf`
2. **arXiv paper**: `https://arxiv.org/pdf/2407.14561.pdf` (NNsight)
3. **GitHub README**: `https://raw.githubusercontent.com/neurodata/m2g/main/README.md`

## Dependencies

These should already be in pyproject.toml, but verify:
```bash
uv add faiss-cpu pypdf2
```

Already installed:
- `sentence-transformers` (used in finetune/analyze_dpo_variance.py)
- `requests`

## CLI Commands

```bash
# Run any exercise
uv run experiments/rag_exercises/ex1_embeddings.py

# After building the full pipeline (ex5)
uv run experiments/rag_exercises/ex5_retrieval.py --query "what is NNsight?"

# Test integration (ex6)
uv run experiments/rag_exercises/ex6_chat_integration.py
```

## Verification Checklist

- [ ] Exercise 1: Embeddings work, similarity computed
- [ ] Exercise 2: FAISS index saves/loads correctly
- [ ] Exercise 3: Text chunks with overlap
- [ ] Exercise 4: Thesis PDF text extracted
- [ ] Exercise 5: Query retrieves relevant chunks
- [ ] **Exercise 6: Chat API uses RAG context**

## Files

```
experiments/rag_exercises/
├── README.md              # This file
├── ex1_embeddings.py      # Learn sentence-transformers
├── ex2_faiss_index.py     # Learn FAISS
├── ex3_chunking.py        # Learn text splitting
├── ex4_pdf_ingestion.py   # Learn content extraction
├── ex5_retrieval.py       # Full retrieval pipeline
├── ex6_chat_integration.py # Integrate with chat API
├── solutions.py           # Reference implementations
└── data/
    ├── faiss.index        # Saved FAISS index (generated)
    └── chunks.json        # Chunk metadata (generated)
```

## Key Learnings for Interviews

After completing these exercises, you can discuss:

1. **Embeddings**: "Sentence-transformers converts text to dense vectors where semantic similarity maps to cosine similarity."

2. **FAISS**: "FAISS is Facebook's library for efficient similarity search. IndexFlatL2 is brute-force but exact; for larger scale you'd use IVF or HNSW indices."

3. **Chunking**: "You chunk documents because embedding models have token limits (~256-512) and smaller chunks give more precise retrieval."

4. **RAG Trade-offs**: "Larger chunks = more context but less precision. Smaller chunks = more precision but may miss context. Overlap helps preserve continuity."

5. **Production Concerns**: "In production you'd batch embeddings, use approximate nearest neighbors, and cache frequent queries."

## Future Expansion

After the prototype works:
- Add YouTube transcript ingestion
- Add web page extraction (trafilatura)
- Index all 33 resume URLs
- Deploy to Fly.io with persistent storage
- Add query classification (when to use RAG vs not)
