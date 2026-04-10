"""MCP server for Claude Memory.

Exposes 6 tools: memory_search, memory_write, memory_update, memory_delete,
memory_export, memory_status.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .db import (
    init_db, insert_memory, update_memory, delete_memory as db_delete,
    record_access, get_memory, get_status, run_decay_sweep, export_all,
)
from .embeddings import (
    add_memory as embed_add,
    search as embed_search,
    count as embed_count,
    delete_memory as embed_delete,
)
from .scoring import compute_score, extract_concepts


mcp = FastMCP("claude-memory")

_decay_ran = False


@mcp.tool()
def memory_search(
    query: str,
    project: str | None = None,
    type: str | None = None,
    n_results: int = 8,
) -> str:
    """Search memories by semantic similarity. Returns ranked results with scores.

    Args:
        query: Natural language search query.
        project: Filter to a specific project name.
        type: Filter by memory type (user, feedback, project, reference, lesson).
        n_results: Maximum results to return (default 8).
    """
    global _decay_ran
    if not _decay_ran:
        run_decay_sweep()
        _decay_ran = True

    candidates = embed_search(query, n_results=n_results * 2, project=project, type=type)

    if not candidates:
        return json.dumps({"results": [], "message": "No memories found."})

    query_concepts = extract_concepts(query)

    scored = []
    for hit in candidates:
        mem = get_memory(hit["id"])
        if not mem or mem["status"] not in ("active", "archived"):
            continue

        score = compute_score(
            semantic_distance=hit.get("distance", 1.0),
            last_accessed=mem.get("last_accessed"),
            access_count=mem.get("access_count", 0),
            query_concepts=query_concepts,
            memory_concepts=mem.get("concepts", "[]"),
            priority=mem.get("priority", "normal"),
        )

        scored.append({
            "id": mem["id"],
            "name": mem.get("name", ""),
            "type": mem.get("type", ""),
            "project": mem.get("project", ""),
            "priority": mem.get("priority", "normal"),
            "status": mem.get("status", "active"),
            "score": score,
            "access_count": mem.get("access_count", 0),
            "content": mem.get("content", "")[:500],
            "created": mem.get("created", ""),
            "last_verified": mem.get("last_verified", ""),
        })

        record_access(mem["id"], query, score)

    scored.sort(key=lambda x: -x["score"])
    results = scored[:n_results]

    return json.dumps({"results": results, "query_concepts": query_concepts}, indent=2)


@mcp.tool()
def memory_write(
    content: str,
    type: str = "project",
    project: str | None = None,
    name: str | None = None,
    description: str | None = None,
    priority: str = "normal",
    tags: list[str] | None = None,
) -> str:
    """Store a new memory. Auto-extracts concept tags for cross-domain retrieval.

    Args:
        content: The memory content to store.
        type: Memory type: user, feedback, project, reference, lesson.
        project: Project name or null for global memories.
        name: Short name for the memory.
        description: One-line description for index display.
        priority: 'pinned' (never decays) or 'normal' (default).
        tags: Optional manual tags.
    """
    concepts = extract_concepts(content)
    mem_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    success = insert_memory(
        id=mem_id,
        content=content,
        type=type,
        project=project,
        priority=priority,
        tags=tags,
        concepts=concepts,
        source="mcp",
        name=name,
        description=description,
        created=now,
        last_verified=now,
    )

    if not success:
        return json.dumps({"status": "duplicate", "message": "A memory with identical content already exists."})

    embed_add(mem_id, content, {
        "type": type,
        "project": project or "",
        "priority": priority,
        "status": "active",
    })

    return json.dumps({
        "status": "created",
        "id": mem_id,
        "concepts": concepts,
        "message": f"Memory '{name or mem_id}' stored successfully.",
    })


@mcp.tool()
def memory_update(
    id: str,
    content: str | None = None,
    priority: str | None = None,
    last_verified: str | None = None,
    status: str | None = None,
) -> str:
    """Update an existing memory.

    Args:
        id: The memory ID to update.
        content: New content (optional, replaces existing).
        priority: New priority: 'pinned' or 'normal'.
        last_verified: Set last_verified date (YYYY-MM-DD).
        status: New status: 'active', 'archived'.
    """
    kwargs = {}
    if priority:
        kwargs["priority"] = priority
    if last_verified:
        kwargs["last_verified"] = last_verified
    if status:
        kwargs["status"] = status

    success = update_memory(id, content=content, **kwargs)

    if not success:
        return json.dumps({"status": "not_found", "message": f"Memory '{id}' not found."})

    if content:
        mem = get_memory(id)
        embed_add(id, content, {
            "type": mem.get("type", "project"),
            "project": mem.get("project", ""),
            "priority": mem.get("priority", "normal"),
            "status": mem.get("status", "active"),
        })

    return json.dumps({"status": "updated", "id": id})


@mcp.tool()
def memory_delete(id: str) -> str:
    """Permanently delete a memory from both SQLite and ChromaDB.

    This is irreversible. Prefer memory_update(id, status='archived') if you
    want to keep the history. Use memory_delete only when the content is
    genuinely obsolete or wrong.

    Args:
        id: The memory ID to delete.
    """
    ok = db_delete(id)
    if not ok:
        return json.dumps({"status": "not_found", "message": f"Memory '{id}' not found."})

    embed_delete(id)
    return json.dumps({"status": "deleted", "id": id})


@mcp.tool()
def memory_export(
    output_path: str,
    include_archived: bool = True,
    include_retrieval_log: bool = False,
) -> str:
    """Dump all memories to a JSON file for backup or migration to another system.

    The file contains: exported_at timestamp, count, memories (full rows with
    parsed tags/concepts), and optionally retrieval_log.

    Args:
        output_path: Absolute path where the JSON file will be written.
        include_archived: If True, include archived memories (default True).
        include_retrieval_log: If True, include the full retrieval history (default False).
    """
    payload = export_all(
        include_archived=include_archived,
        include_retrieval_log=include_retrieval_log,
    )

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return json.dumps({
        "status": "exported",
        "path": str(path),
        "count": payload["count"],
        "include_archived": include_archived,
        "include_retrieval_log": include_retrieval_log,
    })


@mcp.tool()
def memory_status() -> str:
    """Get memory system status: counts, stale entries, retrieval stats."""
    status = get_status()
    status["chroma_count"] = embed_count()
    return json.dumps(status, indent=2, default=str)


def main():
    """Entry point for the MCP server."""
    init_db()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
