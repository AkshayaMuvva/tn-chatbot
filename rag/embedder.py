"""
Embedder: Manages the ChromaDB vector store using sentence-transformers.

Model: all-MiniLM-L6-v2
- 384-dim embeddings, runs locally (no API key)
- Excellent semantic/paraphrase understanding
- Handles queries like "affordable CSE college near Chennai" correctly
"""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
from typing import List, Tuple, Dict

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "tn_colleges"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Singleton client to avoid re-initializing
_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def _get_embedding_fn():
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu",
    )


def get_collection():
    """Return the ChromaDB collection (creates if not exists)."""
    global _collection
    if _collection is not None:
        return _collection
    client = _get_client()
    embedding_fn = _get_embedding_fn()
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def build_index(chunks: List[Tuple[str, Dict]], force_rebuild: bool = False) -> None:
    """
    Build (or rebuild) the ChromaDB index from chunks.
    chunks: list of (text, metadata) pairs
    """
    global _collection
    client = _get_client()
    embedding_fn = _get_embedding_fn()

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("🗑️  Deleted existing collection.")
        except Exception:
            pass
        _collection = None

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not force_rebuild:
        print(f"✅ ChromaDB already has {collection.count()} docs. Use --force to rebuild.")
        _collection = collection
        return

    print(f"📦 Indexing {len(chunks)} chunks into ChromaDB...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c[0] for c in batch]
        metadatas = [c[1] for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        collection.add(documents=texts, metadatas=metadatas, ids=ids)
        pct = min(100, int((i + len(batch)) / len(chunks) * 100))
        print(f"  ✓ {pct}%  ({i + len(batch)}/{len(chunks)})")

    print(f"✅ Index built: {collection.count()} documents stored.")
    _collection = collection


def is_index_ready() -> bool:
    """Check whether the ChromaDB index has been built."""
    try:
        col = get_collection()
        return col.count() > 0
    except Exception:
        return False
