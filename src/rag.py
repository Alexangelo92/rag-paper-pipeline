"""End-to-end RAG: retrieve chunks from ChromaDB, build a grounded prompt, call the LLM.

Usage as a library:
    from src.rag import answer
    result = answer("What is endocytosis?")
    print(result["answer"])

Usage as a CLI:
    python -m src.rag "What is endocytosis?"
    python -m src.rag "What is endocytosis?" --top-k 5 --provider anthropic
"""

from __future__ import annotations

import argparse

from src.llm_client import complete
from src.retrieval import Chunk, Retriever

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "provided excerpts from a scientific paper. Cite the page number for each "
    "claim like (p. 3). If the excerpts don't contain the answer, say so "
    "instead of guessing."
)

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[Page {c.page}] {c.text}" for c in chunks)
    return f"Excerpts:\n{context}\n\nQuestion: {question}"


def answer(question: str, top_k: int = 3, provider: str | None = None, model: str | None = None) -> dict:
    """Retrieve context for `question` and generate a grounded answer."""
    chunks = _get_retriever().retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)
    reply = complete(SYSTEM_PROMPT, prompt, provider=provider, model=model)
    return {"question": question, "answer": reply, "sources": chunks}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question grounded in the paper corpus.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    result = answer(args.question, top_k=args.top_k, provider=args.provider, model=args.model)

    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for c in result["sources"]:
        print(f"  - page {c.page}, chunk {c.chunk_idx} (similarity {c.similarity})")


if __name__ == "__main__":
    _main()
