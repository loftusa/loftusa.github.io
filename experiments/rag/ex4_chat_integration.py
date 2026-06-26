"""
Exercise 4: Chat Integration

CONCEPT: Inject retrieved context into the LLM prompt

BACKGROUND:
Your chat API currently sends this to the LLM:
    [system_prompt.txt] + [resume.txt] + [user messages]

With RAG it becomes:
    [system_prompt.txt] + [resume.txt] + [relevant chunks] + [user messages]

The retrieved chunks give the LLM specific knowledge from your papers,
thesis, GitHub READMEs, and YouTube talks.

RUN:
    uv run experiments/rag/ex4_chat_integration.py
    uv run experiments/rag/ex4_chat_integration.py --live
    uv run experiments/rag/ex4_chat_integration.py --live --query "Tell me about m2g"

Things to figure out:
  - how to figure out how big chunks should be?
  - how can I get queries returned to be more relevant?
    --> similarity is kind of a shitty metric. Wouldn't a key/query vector store system work better?
"""

import os

import chromadb
import click
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from ex3_retrieval import ingest

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_rag"
EXPERIMENTS_DIR = Path(__file__).parent.parent
SYSTEM_PROMPT = (EXPERIMENTS_DIR / "system_prompt.txt").read_text()
RESUME = (EXPERIMENTS_DIR / "resume.txt").read_text()

load_dotenv(dotenv_path=EXPERIMENTS_DIR / ".env")

console = Console()


def load_or_ingest_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="resume_rag")

    if collection.count() == 0:
        console.print("[yellow]Ingesting documents...[/yellow]")
        ingest(collection)
    else:
        console.print(
            f"[green]Loaded existing index: {collection.count()} chunks[/green]"
        )

    return collection


def retrieve_context(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
    max_distance: float = 1.3,
) -> list[dict]:
    """Retrieve relevant chunks for a query. Returns list of {source, text, distance}."""
    results = collection.query(query_texts=[query], n_results=n_results)

    chunks = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        if dist > max_distance:
            continue
        source = meta.get("title", meta.get("source_url", "?"))
        chunks.append({"source": source, "text": doc, "distance": dist})
    return chunks


def build_messages(
    query: str,
    rag_chunks: list[dict],
) -> list[dict[str, str]]:
    """Build the full message list for the LLM."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RESUME},
    ]

    if rag_chunks:
        context_block = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in rag_chunks
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
    """Call Cerebras API via OpenAI-compatible endpoint."""
    from openai import OpenAI

    api_key = os.getenv("CEREBRAS_API_KEY")
    assert api_key, "CEREBRAS_API_KEY not set in experiments/.env"

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content


TEST_QUERIES = [
    "What methods does your thesis use?",
    "Tell me about the m2g pipeline.",
    "What's your favorite pizza topping?",
    "What did Alex work on with NNsight?",
]


@click.command()
@click.option("--live", is_flag=True, help="Actually call Cerebras API for responses.")
@click.option("--query", default=None, help="Single query instead of test suite.")
@click.option("--n-results", default=5, help="Number of chunks to retrieve.")
@click.option(
    "--max-distance", default=1.3, type=float, help="Max distance threshold."
)
def main(
    live: bool, query: str | None, n_results: int, max_distance: float
) -> None:
    collection = load_or_ingest_chroma_collection()
    queries = [query] if query else TEST_QUERIES

    for q in queries:
        console.rule(f"[bold]{q}[/bold]")

        chunks = retrieve_context(
            collection, q, n_results=n_results, max_distance=max_distance
        )

        if chunks:
            console.print(f"[cyan]Retrieved {len(chunks)} chunks:[/cyan]")
            for i, c in enumerate(chunks):
                console.print(
                    f"  {i + 1}. [dim](dist={c['distance']:.3f})[/dim] "
                    f"[yellow]{c['source'][:60]}[/yellow]"
                )
                console.print(f"     {c['text']}...")
        else:
            console.print(
                "[dim]No relevant chunks — proceeding without RAG context.[/dim]"
            )

        messages = build_messages(q, chunks)
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
