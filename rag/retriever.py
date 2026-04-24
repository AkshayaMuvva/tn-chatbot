"""
Retriever: Hybrid vector + metadata search over ChromaDB.

Combines:
  1. Semantic similarity (sentence-transformers embeddings)
  2. Metadata filters (city, ownership, tier, fee range)

This ensures "affordable CSE college in Coimbatore" retrieves correctly
even when phrased differently each time.
"""
from typing import List, Dict, Any, Optional
from .embedder import get_collection


def semantic_search(
    query: str,
    top_k: int = 6,
    city: Optional[str] = None,
    ownership: Optional[str] = None,
    tier: Optional[str] = None,
    college_name: Optional[str] = None,
    max_fee: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid semantic + metadata search.

    Returns list of dicts: {text, metadata, score}
    Score is cosine similarity (0→1, higher=better).
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    # Build ChromaDB WHERE filter (equality only — partial matches post-filtered)
    where_clauses = []
    if city:
        where_clauses.append({"city": {"$eq": city}})
    if ownership:
        where_clauses.append({"ownership": {"$eq": ownership}})
    if tier:
        where_clauses.append({"tier": {"$eq": str(tier)}})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    fetch_k = min(top_k * 3, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        # Post-filter: partial college name match
        if college_name:
            stored = meta.get("college_name", "").lower()
            query_parts = college_name.lower().split()
            if not any(p in stored for p in query_parts if len(p) > 3):
                continue

        # Post-filter: max annual fee
        if max_fee is not None:
            try:
                fee = int(float(meta.get("fees_annual", "0")))
                if fee > max_fee:
                    continue
            except (ValueError, TypeError):
                pass

        hits.append({
            "text": doc,
            "metadata": meta,
            "score": round(1.0 - dist, 4),  # cosine dist → similarity
        })

    return hits[:top_k]


def get_college_chunks(college_name: str) -> List[Dict[str, Any]]:
    """
    Retrieve ALL chunks belonging to a specific college.
    Used by get_college_details to return comprehensive info.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_data = collection.get(include=["documents", "metadatas"])
    name_lower = college_name.lower()
    query_words = [w for w in name_lower.split() if len(w) > 3]

    hits = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        stored = meta.get("college_name", "").lower()
        if name_lower in stored or stored in name_lower:
            hits.append({"text": doc, "metadata": meta, "score": 1.0})
        elif query_words and any(w in stored for w in query_words):
            hits.append({"text": doc, "metadata": meta, "score": 0.9})

    return hits


def format_context(hits: List[Dict[str, Any]], max_chars: int = 7000) -> str:
    """
    Format retrieval hits as a grounded context string for the LLM.
    Includes source attribution to prevent hallucination.
    """
    if not hits:
        return (
            "⚠️ No relevant information found in the verified database. "
            "Do not invent or guess college details."
        )

    parts = []
    total = 0
    for i, hit in enumerate(hits, 1):
        college = hit["metadata"].get("college_name", "Unknown")
        branch = hit["metadata"].get("branch", "Unknown")
        relevance = int(hit["score"] * 100)
        header = f"=== SOURCE {i}: {college} | {branch} (Relevance: {relevance}%) ==="
        block = f"{header}\n{hit['text']}"

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 300:
                parts.append(block[:remaining] + "\n[...truncated for length]")
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)
