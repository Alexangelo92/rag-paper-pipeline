"""Load the ChromaDB collection built by notebook 02 and retrieve chunks for a query."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "output" / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "paper_chunks"


@dataclass
class Chunk:
    text: str
    page: int
    chunk_idx: int
    similarity: float


class Retriever:
    """Wraps the persisted ChromaDB collection + embedding model for querying."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"No ChromaDB store at {chroma_dir}. Run notebooks/02_retrieval.ipynb first."
            )
        self._model = SentenceTransformer(embedding_model)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = client.get_collection(collection_name)

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        query_embedding = self._model.encode([query])[0].tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append(
                Chunk(
                    text=doc,
                    page=meta["page"],
                    chunk_idx=meta["chunk_idx"],
                    similarity=round(1 - dist, 4),
                )
            )
        return chunks
