"""
Exercise 5: Rebuild the RAG system in LangChain

CONCEPT: LangChain vocabulary — Document Loaders, Text Splitters, Embeddings,
         Vector Stores, Retrievers, Chains

BACKGROUND:
The same pipeline you built by hand in ex1–ex4, using LangChain abstractions:

    Your code                    LangChain equivalent
    ─────────────────────────    ──────────────────────────────────────────
    loaders.py                   WebBaseLoader, PyMuPDFLoader
    chunker.py                   RecursiveCharacterTextSplitter
    ex1 (embeddings)             HuggingFaceEmbeddings
    ex2 (Chroma)                 langchain_chroma.Chroma
    ex3 (retrieval)              vectorstore.as_retriever()
    ex4 (chat integration)       manual build_messages + OpenAI client

MAPPING TO YOUR FROM-SCRATCH CODE:
- RecursiveCharacterTextSplitter is LangChain's version of your chunker.py.
  It tries separators in order: ["\\n\\n", "\\n", ". ", " ", ""] — which is
  essentially the same idea as your _find_semantic_break() snapping to
  paragraph > header > sentence boundaries, but implemented as a recursive
  split rather than a sliding window.

- Chroma.from_documents() combines your ex3 ingest() (chunking + storing)
  into one call. It handles embedding automatically via the embeddings arg.

- vectorstore.as_retriever() wraps your ex3 search() — returns list[Document]
  instead of raw Chroma query dicts.

- The LLM call is still manual (Cerebras via OpenAI client) since LangChain's
  ChatOpenAI expects an OpenAI-compatible streaming interface that Cerebras
  doesn't fully support.

INTERVIEW TALKING POINTS:
- "I've used LangChain's document loaders, text splitters, and retrievers"
- "RecursiveCharacterTextSplitter tries paragraph → sentence → word boundaries"
- "I also built the same pipeline from scratch so I understand what's under the hood"
- "LangChain is great for prototyping but adds overhead — in production I'd
   evaluate whether the abstraction is worth it for the use case"

RUN:
    uv run experiments/rag/ex5_langchain.py
    uv run experiments/rag/ex5_langchain.py --live
    uv run experiments/rag/ex5_langchain.py --query "connectome analysis"
    uv run experiments/rag/ex5_langchain.py --rebuild
"""

import os
import shutil

import click
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from extract_urls import parse_urls, Url

DATA_DIR = Path(__file__).parent / "data"
LC_CHROMA_DIR = DATA_DIR / "chroma_langchain"
EXPERIMENTS_DIR = Path(__file__).parent.parent
RESUME_FILEPATH = EXPERIMENTS_DIR / "resume.txt"
SYSTEM_PROMPT = (EXPERIMENTS_DIR / "system_prompt.txt").read_text()
RESUME = (EXPERIMENTS_DIR / "resume.txt").read_text()

load_dotenv(dotenv_path=EXPERIMENTS_DIR / ".env")

console = Console()

# ── LangChain equivalents of your from-scratch components ──────────────────

# chunker.py → RecursiveCharacterTextSplitter
# Uses separators ["\n\n", "\n", ". ", " ", ""] which map to your
# _SECTION_BREAK, _HEADER_BREAK, _SENTENCE_END patterns.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ex1 (sentence-transformers) → HuggingFaceEmbeddings
# Chroma's default is all-MiniLM-L6-v2; we match that here explicitly.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Document loading ───────────────────────────────────────────────────────

def load_url_langchain(url: Url) -> list[Document]:
    """Load a single URL using LangChain document loaders.

    Maps to your loaders.py dispatch logic:
      html/arxiv/github → WebBaseLoader
      pdf               → PyMuPDFLoader (via temp file)
      youtube/scholar    → skip
    """
    if url.classification in ("youtube", "scholar"):
        return []

    if url.classification == "pdf":
        # PyMuPDFLoader needs a file path or URL
        try:
            docs = PyMuPDFLoader(url.url).load()
            for d in docs:
                d.metadata["doc_type"] = "pdf"
                d.metadata["source_url"] = url.url
            return docs
        except Exception as e:
            console.print(f"[dim]  skip pdf {url.url}: {e}[/dim]")
            return []

    # html, arxiv, github — all fetched as web pages
    try:
        docs = WebBaseLoader(url.url).load()
        for d in docs:
            d.metadata["doc_type"] = url.classification
            d.metadata["source_url"] = url.url
        return docs
    except Exception as e:
        console.print(f"[dim]  skip {url.classification} {url.url}: {e}[/dim]")
        return []


def ingest_langchain() -> Chroma:
    """Ingest all resume URLs → LangChain Documents → split → Chroma.

    This is the LangChain equivalent of your ex3 ingest() function.
    """
    urls = parse_urls(RESUME_FILEPATH)
    all_docs: list[Document] = []

    for url in urls:
        docs = load_url_langchain(url)
        if docs:
            chunks = splitter.split_documents(docs)
            # Filter out empty/whitespace-only chunks
            chunks = [c for c in chunks if c.page_content.strip()]
            all_docs.extend(chunks)
            title = docs[0].metadata.get("title", url.url)[:60]
            console.print(
                f"  [{url.classification:>8}] {len(chunks):>3} chunks  {title}"
            )

    console.print(f"\n[bold]Total: {len(all_docs)} chunks[/bold]")

    # Chroma.from_documents = your collection.add() but handles embeddings
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(LC_CHROMA_DIR),
        collection_name="resume_langchain",
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load existing Chroma vectorstore (LangChain wrapper)."""
    return Chroma(
        persist_directory=str(LC_CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="resume_langchain",
    )


# ── Retrieval + chat (mirrors ex4) ────────────────────────────────────────

def retrieve_context(vectorstore: Chroma, query: str, k: int = 5) -> list[Document]:
    """LangChain equivalent of ex4's retrieve_context().

    Uses vectorstore.as_retriever() — the standard LangChain retriever interface.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def build_messages(query: str, docs: list[Document]) -> list[dict[str, str]]:
    """Build LLM messages — identical logic to ex4."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RESUME},
    ]

    if docs:
        context_block = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('title', d.metadata.get('source_url', '?'))}]\n"
            f"{d.page_content}"
            for d in docs
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
@click.option("--k", default=5, help="Number of chunks to retrieve.")
def main(live: bool, query: str | None, rebuild: bool, k: int) -> None:
    if rebuild and LC_CHROMA_DIR.exists():
        shutil.rmtree(LC_CHROMA_DIR)

    DATA_DIR.mkdir(exist_ok=True)

    # Ingest or load
    if not LC_CHROMA_DIR.exists() or rebuild:
        console.print("[yellow]Ingesting with LangChain...[/yellow]\n")
        vectorstore = ingest_langchain()
    else:
        vectorstore = load_vectorstore()
        n = vectorstore._collection.count()
        console.print(f"[green]Loaded existing LangChain index: {n} chunks[/green]")

    queries = [query] if query else TEST_QUERIES

    for q in queries:
        console.rule(f"[bold]{q}[/bold]")

        docs = retrieve_context(vectorstore, q, k=k)

        if docs:
            console.print(f"[cyan]Retrieved {len(docs)} chunks:[/cyan]")
            for i, d in enumerate(docs):
                source = d.metadata.get("title", d.metadata.get("source_url", "?"))
                dtype = d.metadata.get("doc_type", "?")
                console.print(
                    f"  {i + 1}. [dim][{dtype}][/dim] "
                    f"[yellow]{str(source)[:60]}[/yellow]"
                )
                console.print(f"     {d.page_content}")
        else:
            console.print(
                "[dim]No relevant chunks — proceeding without RAG context.[/dim]"
            )

        messages = build_messages(q, docs)
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
