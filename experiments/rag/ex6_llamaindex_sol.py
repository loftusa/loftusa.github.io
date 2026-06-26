"""
Exercise 6: Rebuild the RAG system in LlamaIndex

CONCEPT: LlamaIndex vocabulary — Documents, Nodes, NodeParsers, Embeddings,
         VectorStoreIndex, Retrievers, QueryEngines

BACKGROUND:
The same pipeline from ex1–ex4, now in LlamaIndex. Key differences from LangChain:

    Your code                    LlamaIndex equivalent
    ─────────────────────────    ──────────────────────────────────────────
    loaders.py                   SimpleWebPageReader, PyMuPDFReader
    chunker.py                   SentenceSplitter (node parser)
    ex1 (embeddings)             HuggingFaceEmbedding (via Settings)
    ex2 (Chroma)                 ChromaVectorStore + StorageContext
    ex3 (retrieval)              index.as_retriever()
    ex4 (chat integration)       manual build_messages + OpenAI client

LlamaIndex vs LangChain — key vocabulary differences:
- LangChain "Document" = LlamaIndex "Document" (same concept)
- LangChain "chunks" = LlamaIndex "Nodes" (chunks with relationships)
- LangChain "TextSplitter" = LlamaIndex "NodeParser"
- LangChain "Chain" = LlamaIndex "QueryEngine"
- LlamaIndex is more opinionated: VectorStoreIndex.from_documents() does
  parse + embed + store in one call. Less config, more magic.
- LlamaIndex Nodes track parent/child relationships between chunks,
  which LangChain chunks don't.

RUN:
    uv run experiments/rag/ex6_llamaindex.py
    uv run experiments/rag/ex6_llamaindex.py --live
    uv run experiments/rag/ex6_llamaindex.py --query "connectome analysis"
    uv run experiments/rag/ex6_llamaindex.py --rebuild
"""

import os
import shutil
import tempfile

import chromadb
import click
import requests
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PyMuPDFReader

from extract_urls import parse_urls, Url

DATA_DIR = Path(__file__).parent / "data"
LI_CHROMA_DIR = DATA_DIR / "chroma_llamaindex"
EXPERIMENTS_DIR = Path(__file__).parent.parent
RESUME_FILEPATH = EXPERIMENTS_DIR / "resume.txt"
SYSTEM_PROMPT = (EXPERIMENTS_DIR / "system_prompt.txt").read_text()
RESUME = (EXPERIMENTS_DIR / "resume.txt").read_text()

load_dotenv(dotenv_path=EXPERIMENTS_DIR / ".env")

console = Console()

# ── Global settings (LlamaIndex's way of configuring defaults) ─────────────

# SentenceSplitter is LlamaIndex's equivalent of your chunker.py.
# It splits on sentences first, then falls back to smaller units.
# This maps to your _find_semantic_break() logic.
Settings.text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=200)

# HuggingFaceEmbedding = your ex1 sentence-transformers code.
# Chroma's default is all-MiniLM-L6-v2; match it here.
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Disable LLM for indexing (we call Cerebras manually)
Settings.llm = None


# ── Document loading ───────────────────────────────────────────────────────

def load_url_llamaindex(url: Url) -> list[Document]:
    """Load a URL into LlamaIndex Documents.

    Maps to your loaders.py dispatch:
      html/arxiv/github → SimpleWebPageReader
      pdf               → PyMuPDFReader (download to temp file first)
      youtube/scholar   → skip
    """
    if url.classification in ("youtube", "scholar"):
        return []

    if url.classification == "pdf":
        try:
            resp = requests.get(url.url, timeout=30)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name
            docs = PyMuPDFReader().load_data(file_path=tmp_path)
            os.unlink(tmp_path)
            for d in docs:
                d.metadata["doc_type"] = "pdf"
                d.metadata["source_url"] = url.url
            return docs
        except Exception as e:
            console.print(f"[dim]  skip pdf {url.url}: {e}[/dim]")
            return []

    # html, arxiv, github
    try:
        docs = SimpleWebPageReader(html_to_text=True).load_data([url.url])
        for d in docs:
            d.metadata["doc_type"] = url.classification
            d.metadata["source_url"] = url.url
        return docs
    except Exception as e:
        console.print(f"[dim]  skip {url.classification} {url.url}: {e}[/dim]")
        return []


def ingest_llamaindex() -> VectorStoreIndex:
    """Ingest all resume URLs → LlamaIndex Documents → VectorStoreIndex.

    VectorStoreIndex.from_documents() does three things at once:
    1. Parse documents into Nodes (your chunker.py)
    2. Embed each Node (your ex1 code)
    3. Store in the vector store (your ex2/ex3 code)
    """
    urls = parse_urls(RESUME_FILEPATH)
    all_docs: list[Document] = []

    for url in urls:
        docs = load_url_llamaindex(url)
        if docs:
            # Filter empty docs
            docs = [d for d in docs if d.text.strip()]
            all_docs.extend(docs)
            title = docs[0].metadata.get("source_url", url.url)[:60] if docs else "?"
            console.print(
                f"  [{url.classification:>8}] {len(docs):>3} docs    {title}"
            )

    console.print(f"\n[bold]Total: {len(all_docs)} documents (pre-chunking)[/bold]")

    # Set up Chroma as the backing vector store
    chroma_client = chromadb.PersistentClient(path=str(LI_CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("resume_llamaindex")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # This single call does: chunk → embed → store
    index = VectorStoreIndex.from_documents(
        all_docs,
        storage_context=storage_context,
        show_progress=True,
    )

    n = chroma_collection.count()
    console.print(f"[bold]Indexed: {n} nodes (post-chunking)[/bold]")
    return index


def load_index() -> VectorStoreIndex:
    """Load existing index from Chroma."""
    chroma_client = chromadb.PersistentClient(path=str(LI_CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("resume_llamaindex")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)


# ── Retrieval + chat (mirrors ex4) ────────────────────────────────────────

def retrieve_context(
    index: VectorStoreIndex, query: str, k: int = 5
) -> list[NodeWithScore]:
    """LlamaIndex equivalent of ex4's retrieve_context().

    index.as_retriever() returns a VectorIndexRetriever.
    .retrieve() returns NodeWithScore objects (nodes + similarity scores).
    """
    retriever = index.as_retriever(similarity_top_k=k)
    return retriever.retrieve(query)


def build_messages(
    query: str, nodes: list[NodeWithScore]
) -> list[dict[str, str]]:
    """Build LLM messages — identical logic to ex4."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RESUME},
    ]

    if nodes:
        context_block = "\n\n---\n\n".join(
            f"[Source: {n.metadata.get('source_url', '?')}]\n{n.get_content()}"
            for n in nodes
        )
        messages.append(
            {
                "role": "system",
                "content": (
                    "The following are additional context chunks retrieved from "
                    "Alex's papers, talks, and projects. Use them to give more "
                    "specific answers when relevant.\n\n" + context_block
                ),
            }
        )

    messages.append({"role": "user", "content": query})
    return messages


MODEL = "zai-glm-4.7"


def call_cerebras(messages: list[dict[str, str]]) -> str:
    """Call Cerebras API via OpenAI-compatible endpoint (same as ex4)."""
    from openai import OpenAI

    api_key = os.getenv("CEREBRAS_API_KEY")
    assert api_key, "CEREBRAS_API_KEY not set in experiments/.env"

    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
    completion = client.chat.completions.create(model=MODEL, messages=messages)
    return completion.choices[0].message.content


# ── CLI ────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "What methods does your thesis use?",
    "Tell me about the m2g pipeline.",
    "What's your favorite pizza topping?",
    "What did Alex work on with NNsight?",
]


@click.command()
@click.option("--live", is_flag=True, help="Call Cerebras API for responses.")
@click.option("--query", default=None, help="Single query instead of test suite.")
@click.option("--rebuild", is_flag=True, help="Delete and re-ingest.")
@click.option("--k", default=5, help="Number of nodes to retrieve.")
def main(live: bool, query: str | None, rebuild: bool, k: int) -> None:
    if rebuild and LI_CHROMA_DIR.exists():
        shutil.rmtree(LI_CHROMA_DIR)

    DATA_DIR.mkdir(exist_ok=True)

    # Ingest or load
    if not LI_CHROMA_DIR.exists() or rebuild:
        console.print("[yellow]Ingesting with LlamaIndex...[/yellow]\n")
        index = ingest_llamaindex()
    else:
        index = load_index()
        chroma_client = chromadb.PersistentClient(path=str(LI_CHROMA_DIR))
        n = chroma_client.get_or_create_collection("resume_llamaindex").count()
        console.print(f"[green]Loaded existing LlamaIndex index: {n} nodes[/green]")

    queries = [query] if query else TEST_QUERIES

    for q in queries:
        console.rule(f"[bold]{q}[/bold]")

        nodes = retrieve_context(index, q, k=k)

        if nodes:
            console.print(f"[cyan]Retrieved {len(nodes)} nodes:[/cyan]")
            for i, n in enumerate(nodes):
                source = n.metadata.get("source_url", "?")
                dtype = n.metadata.get("doc_type", "?")
                score = n.score
                console.print(
                    f"  {i + 1}. [dim](score={score:.3f}) [{dtype}][/dim] "
                    f"[yellow]{str(source)[:60]}[/yellow]"
                )
                console.print(f"     {n.get_content()}")
        else:
            console.print(
                "[dim]No relevant nodes — proceeding without RAG context.[/dim]"
            )

        messages = build_messages(q, nodes)
        sys_chars = sum(
            len(m["content"]) for m in messages if m["role"] == "system"
        )
        console.print(
            f"[dim]System context: {sys_chars:,} chars "
            f"({len(messages) - 1} system messages)[/dim]"
        )

        if live:
            console.print("[green]Calling Cerebras...[/green]")
            response = call_cerebras(messages)
            console.print(Panel(response, title="Response", border_style="green"))
        else:
            console.print("[dim](use --live to see LLM response)[/dim]")

        console.print()


if __name__ == "__main__":
    main()
