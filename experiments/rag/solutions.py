"""
Reference implementations for RAG exercises 1–4.
Try each exercise yourself first!
"""

import logging
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from chunker import chunk_document
from extract_urls import parse_urls, resume_filepath
from loaders import load_document

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_rag"
EXPERIMENTS_DIR = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Exercise 1: Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def ex1():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "I research neural network interpretability.",
        "My work focuses on understanding how AI models reason.",
        "Mechanistic interpretability reverse-engineers circuits.",
        "I like to eat pizza on Fridays.",
        "The weather in Baltimore is rainy today.",
    ]

    embeddings = model.encode(sentences)  # (B, D)
    print(f"Shape: {embeddings.shape}")

    # Normalize and compute full similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = (embeddings @ embeddings.T) / (norms * norms.T)

    import seaborn as sns
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f")
    print("\n".join(sentences))


# ═══════════════════════════════════════════════════════════════════════════════
# Exercise 2: Vector Store
# ═══════════════════════════════════════════════════════════════════════════════

def ex2():
    import shutil

    db_path = DATA_DIR / "chroma_ex2"
    if db_path.exists():
        shutil.rmtree(db_path)
    DATA_DIR.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name="demo")

    docs = [
        "Alex's thesis studies statistical methods for connectome analysis.",
        "The m2g pipeline processes diffusion MRI into brain graphs.",
        "NNsight is a library for neural network interpretability.",
        "Graspologic provides tools for statistical analysis of networks.",
        "Alex won a gold medal in the Vesuvius Challenge for ink detection.",
        "The chat API uses Cerebras for fast inference.",
    ]

    collection.add(
        ids=[f"doc_{i}" for i in range(len(docs))],
        documents=docs,
        metadatas=[{"source": "demo"} for _ in docs],
    )
    print(f"Added {collection.count()} documents")

    for query in ["brain connectivity", "machine learning interpretability"]:
        results = collection.query(query_texts=[query], n_results=3)
        print(f"\nQuery: \"{query}\"")
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            print(f"  (dist={dist:.3f}) {doc[:70]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Exercise 3: End-to-End Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def ex3(query: str = "what is the thesis about?", rebuild: bool = False):
    import shutil

    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    DATA_DIR.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="resume_rag")

    if collection.count() == 0:
        print("Ingesting all resume URLs …\n")
        urls = parse_urls(resume_filepath)
        chunk_id = 0

        for url_obj in urls:
            doc = load_document(url_obj)
            if doc is None:
                continue

            chunks = chunk_document(doc)
            if not chunks:
                continue

            collection.add(
                ids=[f"chunk_{chunk_id + i}" for i in range(len(chunks))],
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "source_url": c.source_url,
                        "title": c.title,
                        "doc_type": c.doc_type,
                        "chunk_index": c.chunk_index,
                    }
                    for c in chunks
                ],
            )
            chunk_id += len(chunks)
            print(f"  [{url_obj.classification:>8}] {len(chunks):>3} chunks  {doc['title'][:50]}")

        print(f"\nTotal: {collection.count()} chunks")
    else:
        print(f"Loaded existing index: {collection.count()} chunks")

    results = collection.query(query_texts=[query], n_results=5)
    print(f"\nQuery: \"{query}\"\n")
    for i, (doc, dist, meta) in enumerate(
        zip(results["documents"][0], results["distances"][0], results["metadatas"][0])
    ):
        sim = 1 / (1 + dist)
        print(f"{i+1}. (sim={sim:.3f}) [{meta.get('doc_type')}] {meta.get('title', '?')}")
        print(f"   {doc[:150]}…\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Exercise 4: Chat Integration
# ═══════════════════════════════════════════════════════════════════════════════

def ex4(live: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name="resume_rag")
    print(f"Loaded {collection.count()} chunks\n")

    system_prompt = (EXPERIMENTS_DIR / "system_prompt.txt").read_text()
    resume = (EXPERIMENTS_DIR / "resume.txt").read_text()

    queries = [
        "What methods does your thesis use?",
        "Tell me about the m2g pipeline.",
        "What's your favorite pizza topping?",
    ]

    for query in queries:
        # Retrieve
        results = collection.query(query_texts=[query], n_results=3)
        context_parts = []
        for doc, dist, meta in zip(
            results["documents"][0], results["distances"][0], results["metadatas"][0]
        ):
            if dist > 1.5:
                continue
            source = meta.get("title", meta.get("source_url", "?"))
            context_parts.append(f"[Source: {source}]\n{doc}")

        rag_context = "\n\n---\n\n".join(context_parts)

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": resume},
        ]
        if rag_context:
            messages.append({
                "role": "system",
                "content": "Additional context from Alex's work:\n\n" + rag_context,
            })
        messages.append({"role": "user", "content": query})

        print(f"{'═' * 60}")
        print(f"QUERY: \"{query}\"")
        print(f"  Messages: {len(messages)}, RAG context: {len(rag_context)} chars")

        if live:
            import os
            from dotenv import load_dotenv
            from cerebras.cloud.sdk import Cerebras

            load_dotenv(dotenv_path=EXPERIMENTS_DIR / ".env")
            llm = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
            resp = llm.chat.completions.create(
                model="zai-glm-4.7", messages=messages, stream=False
            )
            print(f"  Response: {resp.choices[0].message.content[:200]}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Exercise 5: LangChain
# ═══════════════════════════════════════════════════════════════════════════════

def ex5(query: str = "what is the thesis about?"):
    import shutil
    from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    lc_dir = str(DATA_DIR / "chroma_langchain")
    if (DATA_DIR / "chroma_langchain").exists():
        shutil.rmtree(DATA_DIR / "chroma_langchain")

    # 1. Load documents
    print("Loading documents …")
    thesis_docs = PyMuPDFLoader("https://alex-loftus.com/files/submitted_thesis.pdf").load()
    readme_docs = WebBaseLoader("https://raw.githubusercontent.com/neurodata/m2g/master/README.md").load()
    all_docs = thesis_docs + readme_docs
    print(f"  Loaded {len(all_docs)} documents")

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    print(f"  Split into {len(chunks)} chunks")

    # 3. Embed + store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=lc_dir)
    print(f"  Stored in Chroma ({vectorstore._collection.count()} chunks)")

    # 4. Retrieve
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)

    print(f"\nQuery: \"{query}\"\n")
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "?")
        print(f"{i+1}. [{source[:60]}]")
        print(f"   {doc.page_content[:150]}…\n")

    # Mapping to from-scratch code:
    print("─" * 60)
    print("LangChain class            →  Your from-scratch code")
    print("─" * 60)
    print("PyMuPDFLoader              →  loaders.pdf_loader()")
    print("WebBaseLoader              →  loaders.html_loader()")
    print("RecursiveCharacterTextSplitter → chunker.chunk_document()")
    print("HuggingFaceEmbeddings      →  ex1: SentenceTransformer.encode()")
    print("Chroma.from_documents()    →  ex2: collection.add()")
    print("vectorstore.as_retriever() →  ex3: collection.query()")


if __name__ == "__main__":
    import sys

    usage = "Usage: uv run solutions.py [ex1|ex2|ex3|ex4|ex5]"
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    match sys.argv[1]:
        case "ex1":
            ex1()
        case "ex2":
            ex2()
        case "ex3":
            ex3(
                query=sys.argv[2] if len(sys.argv) > 2 else "what is the thesis about?",
                rebuild="--rebuild" in sys.argv,
            )
        case "ex4":
            ex4(live="--live" in sys.argv)
        case "ex5":
            ex5(query=sys.argv[2] if len(sys.argv) > 2 else "what is the thesis about?")
        case _:
            print(usage)
