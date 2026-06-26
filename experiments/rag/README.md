# RAG Learning Exercises

Build a Retrieval-Augmented Generation system for the resume chatbot, step by step.

## What's Here

**You already built** (the plumbing):
- `extract_urls.py` — Parse URLs from `resume.txt`, classify by type
- `loaders.py` — Fetch content from each URL type (html, arxiv, youtube, pdf, github)
- `chunker.py` — Split documents into overlapping text chunks

**Exercises** (what's left to learn):

| Exercise | Concept | What You Build |
|----------|---------|----------------|
| `ex1_embeddings.py` | sentence-transformers | Embed text → vectors, cosine similarity |
| `ex2_vector_store.py` | ChromaDB | Store + search vectors, manual vs auto-embed |
| `ex3_retrieval.py` | Full pipeline | Ingest all URLs → Chroma → query |
| `ex4_chat_integration.py` | Production RAG | Build RAG-enhanced prompts for `chat_api.py` |
| `ex5_langchain.py` | LangChain | Rebuild the same pipeline with LangChain abstractions |
| `ex6_llamaindex.py` | LlamaIndex | Rebuild the same pipeline with LlamaIndex abstractions |

## How to Work Through

```bash
# Exercise 1: embeddings
uv run experiments/rag/ex1_embeddings.py

# Exercise 2: Chroma
uv run experiments/rag/ex2_vector_store.py

# Exercise 3: full pipeline (ingests all URLs — takes a minute)
uv run experiments/rag/ex3_retrieval.py

# Exercise 4: chat integration
uv run experiments/rag/ex4_chat_integration.py

# Exercise 4 with live LLM responses:
uv run experiments/rag/ex4_chat_integration.py --live

# Exercise 5: LangChain
uv run experiments/rag/ex5_langchain.py

# Exercise 6: LlamaIndex
uv run experiments/rag/ex6_llamaindex.py
```

Each exercise has `# YOUR CODE HERE` placeholders with hints.
Reference implementations are in `solutions.py`.

## The Mental Model

```
OFFLINE (once):
  resume.txt → URLs → fetch content → chunk → embed → Chroma DB

ONLINE (every query):
  user question → embed → Chroma search → top-K chunks
       ↓
  [system prompt] + [resume] + [RAG chunks] + [question] → LLM → answer
```

## Files

```
experiments/rag/
├── extract_urls.py        # URL extraction (done)
├── loaders.py             # Content fetchers (done)
├── chunker.py             # Text chunking (done)
├── ex1_embeddings.py      # Exercise: embeddings
├── ex2_vector_store.py    # Exercise: ChromaDB
├── ex3_retrieval.py       # Exercise: full pipeline
├── ex4_chat_integration.py # Exercise: wire into chat API
├── ex5_langchain.py       # Exercise: same pipeline in LangChain
├── ex6_llamaindex.py      # Exercise: same pipeline in LlamaIndex
├── solutions.py           # Reference implementations
├── README.md              # This file
└── data/                  # Generated indexes (gitignored)
```

## Dependencies

In `pyproject.toml`:
- `chromadb` — vector database
- `pymupdf` — PDF extraction
- `beautifulsoup4` — HTML parsing
- `youtube-transcript-api` — YouTube transcripts
- `sentence-transformers` — embedding model
- `langchain`, `langchain-community`, `langchain-chroma`, `langchain-huggingface` — LangChain (ex5)
- `llama-index-core`, `llama-index-vector-stores-chroma`, `llama-index-embeddings-huggingface`, `llama-index-readers-web`, `llama-index-readers-file` — LlamaIndex (ex6)
