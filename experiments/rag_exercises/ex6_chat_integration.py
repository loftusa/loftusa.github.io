"""
Exercise 6: Chat Integration (add RAG to your chat API)

CONCEPT: Integrating retrieval into the LLM prompt

BACKGROUND:
Now we connect the RAG system to your existing chat API.
The flow becomes:

1. User sends message
2. Search RAG index for relevant chunks
3. Inject chunks into the prompt
4. Send to LLM with enriched context
5. Return response

This is the final step before production deployment!

YOUR GOAL:
1. Load the RAG index
2. Create a function that builds RAG-enhanced prompts
3. Test with your local chat setup

RUN WHEN COMPLETE:
    uv run experiments/rag_exercises/ex6_chat_integration.py

SUCCESS CRITERIA:
- RAG context gets injected into prompts
- Answers are more specific than without RAG
- Ready to integrate into chat_api.py

WHAT YOU'RE LEARNING: How to wire RAG into a production system
"""

from pathlib import Path

# Import your RAG system
try:
    from ex5_retrieval import RAGIndex, DATA_DIR
except ImportError:
    print("ERROR: Complete ex5_retrieval.py first!")
    raise


def build_rag_prompt(
    rag_index: RAGIndex,
    user_message: str,
    system_prompt: str,
    resume: str,
    k: int = 3,
    min_similarity: float = 0.3,
) -> list[dict[str, str]]:
    """Build a prompt with RAG context.

    TODO: Search for relevant chunks and build the prompt

    ALGORITHM:
    1. Search RAG index for relevant chunks
    2. Filter by minimum similarity
    3. Format chunks as context
    4. Build the full prompt: [system, resume, rag_context, user]

    HINT:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": resume},
        ]

        # Search for relevant context
        results = rag_index.search(user_message, k=k)

        # Format retrieved context
        if results:
            context_parts = []
            for chunk, distance in results:
                similarity = 1 / (1 + distance)
                if similarity >= min_similarity:
                    context_parts.append(
                        f"[Source: {chunk.source}]\n{chunk.text}"
                    )

            if context_parts:
                rag_context = "\\n\\n---\\n\\n".join(context_parts)
                messages.append({
                    "role": "system",
                    "content": f"Additional context from Alex's work:\\n\\n{rag_context}"
                })

        messages.append({"role": "user", "content": user_message})
        return messages

    Args:
        rag_index: Loaded RAGIndex
        user_message: The user's question
        system_prompt: System prompt for the chatbot
        resume: Resume text
        k: Number of chunks to retrieve
        min_similarity: Minimum similarity threshold

    Returns:
        List of message dicts ready for the LLM
    """
    # TODO: Your code here
    pass


def demo_rag_integration():
    """Interactive demo of RAG-enhanced chat."""

    print("Exercise 6: Chat Integration\n")
    print("Loading RAG index...")

    # Check if index exists
    if not (DATA_DIR / "faiss.index").exists():
        print("ERROR: No index found. Run ex5_retrieval.py first!")
        print("  uv run experiments/rag_exercises/ex5_retrieval.py")
        return

    # Load RAG index
    rag = RAGIndex()
    rag.load()
    print(f"Loaded {len(rag.chunks)} chunks\n")

    # Load system prompt and resume from chat_api
    experiments_dir = Path(__file__).parent.parent
    system_prompt_path = experiments_dir / "system_prompt.txt"
    resume_path = experiments_dir / "resume.txt"

    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text()
    else:
        system_prompt = "You are a helpful assistant representing Alex Loftus."

    if resume_path.exists():
        resume = resume_path.read_text()
    else:
        resume = "Alex Loftus is a PhD student studying AI interpretability."

    # Test queries
    test_queries = [
        "What is your thesis about?",
        "Tell me about the m2g pipeline",
        "What is connectome analysis?",
        "Do you have experience with graph neural networks?",
    ]

    print("=" * 60)
    print("Testing RAG prompt building")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQUERY: {query}")
        print("-" * 40)

        messages = build_rag_prompt(
            rag_index=rag,
            user_message=query,
            system_prompt=system_prompt,
            resume=resume,
            k=3,
            min_similarity=0.2,
        )

        if messages is None:
            print("ERROR: build_rag_prompt() returned None. Implement it!")
            return

        # Show what context was retrieved
        rag_context_msg = None
        for msg in messages:
            if "Additional context" in msg.get("content", ""):
                rag_context_msg = msg
                break

        if rag_context_msg:
            print("RAG CONTEXT FOUND:")
            # Show first 300 chars of context
            context = rag_context_msg["content"]
            print(f"  {context[:300]}...")
        else:
            print("NO RAG CONTEXT (below similarity threshold)")

        print(f"\nTotal messages in prompt: {len(messages)}")

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter questions to see RAG context (type 'quit' to exit)\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            continue

        messages = build_rag_prompt(
            rag_index=rag,
            user_message=query,
            system_prompt=system_prompt,
            resume=resume,
            k=3,
        )

        if messages is None:
            print("ERROR: build_rag_prompt() not implemented")
            continue

        print("\nGenerated prompt structure:")
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"  [{i}] {role}: {content}")
        print()

    print("\nNext step: Integrate into experiments/chat_api.py!")
    print("""
To integrate:
1. Import RAGIndex in chat_api.py
2. Load index at startup (lazy load to save memory)
3. Modify build_messages() to use build_rag_prompt()
4. Deploy!

Example modification to chat_api.py:

    from rag_exercises.ex5_retrieval import RAGIndex

    _rag_index = None

    def get_rag_index():
        global _rag_index
        if _rag_index is None:
            _rag_index = RAGIndex()
            _rag_index.load()
        return _rag_index

    def build_messages(user_messages, use_rag=True):
        if use_rag:
            return build_rag_prompt(
                get_rag_index(),
                user_messages[-1].content,
                SYSTEM_PROMPT,
                RESUME
            )
        # ... existing code
""")


if __name__ == "__main__":
    demo_rag_integration()
