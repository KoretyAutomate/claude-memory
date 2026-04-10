"""ChromaDB vector store for semantic search."""

import chromadb

from . import config


def _get_client() -> chromadb.PersistentClient:
    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_PATH))


def _get_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_memory(id: str, content: str, metadata: dict) -> None:
    """Add or update a memory embedding."""
    client = _get_client()
    collection = _get_collection(client)

    clean_meta = {}
    for k, v in metadata.items():
        if v is not None and isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v

    collection.upsert(ids=[id], documents=[content], metadatas=[clean_meta])


def search(
    query: str,
    n_results: int = 10,
    project: str | None = None,
    type: str | None = None,
) -> list[dict]:
    """Semantic search. Returns list of {id, content, distance, metadata}."""
    client = _get_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        return []

    where_filter = None
    conditions = []
    if project:
        conditions.append({"project": {"$eq": project}})
    if type:
        conditions.append({"type": {"$eq": type}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        where=where_filter if where_filter else None,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
        })

    return hits


def delete_memory(id: str) -> None:
    """Remove a memory from the vector store."""
    client = _get_client()
    collection = _get_collection(client)
    collection.delete(ids=[id])


def count() -> int:
    """Number of entries in the vector store."""
    client = _get_client()
    collection = _get_collection(client)
    return collection.count()
