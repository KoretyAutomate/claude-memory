"""Prompt-relevance memory auto-injection (MVP: L2-only).

Selects the top-N memories above a relevance floor for a user prompt and
renders a `<memory-context>` block for injection via the Claude Code
`UserPromptSubmit` hook.

Design constraints:
- Reuses `scoring.compute_score` so injected and explicitly-searched memories
  rank by the same formula.
- Does NOT call `db.record_access` — auto-injection is not a user-initiated
  retrieval and should not inflate frequency scores.
- Opens SQLite in read-only URI mode to avoid contention with the MCP server.
- ChromaDB and sentence-transformers are lazy-imported inside `select_l2` so
  callers that short-circuit (kill switches) do not pay the cold-start cost.
- Memory content containing the literal `</memory-context>` is escaped to
  prevent prompt injection via the block-closer tag.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import config
from .scoring import compute_score, extract_concepts


# Default rendering thresholds. The CLI reads env vars and passes them in;
# these module-level defaults are only used by tests that call the functions
# directly.
DEFAULT_STALE_DAYS = 30
DEFAULT_MAX_AGE_DAYS = 90
BLOCK_OPEN = "<memory-context>"
BLOCK_CLOSE = "</memory-context>"
# Zero-width space inserted to neutralize a literal block-closer appearing
# inside a memory body — defangs prompt injection via the closing tag.
_ESCAPED_CLOSE = "</memory-context\u200b>"


@dataclass
class LayerResult:
    """Result of a single selection layer.

    `memories` is ordered by score, highest first. Each entry is a dict with
    at least: id, content, type, last_verified, score.
    """

    name: str
    memories: list[dict]
    budget_tokens: int
    floor_score: float = 0.0


# --------------------------------------------------------------------------- #
# Read-only SQLite access
# --------------------------------------------------------------------------- #

def _connect_readonly() -> sqlite3.Connection:
    """Open the memory DB in read-only URI mode with a short timeout.

    Raises `sqlite3.OperationalError` if the DB file does not exist — callers
    should catch this and treat it as "no memories".
    """
    # URI requires forward-slash paths even on Windows; Path.as_posix handles it.
    uri = f"file:{config.DB_PATH.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=0.5)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_memory_readonly(conn: sqlite3.Connection, memory_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Staleness / age helpers
# --------------------------------------------------------------------------- #

def _parse_verified(value: str | None) -> datetime | None:
    """Parse a `last_verified` field (YYYY-MM-DD or ISO8601) to aware UTC."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _days_since(value: str | None) -> float | None:
    dt = _parse_verified(value)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400


def _is_stale(last_verified: str | None, stale_days: int = DEFAULT_STALE_DAYS) -> bool:
    """NULL or older than `stale_days` → stale."""
    days = _days_since(last_verified)
    if days is None:
        return True
    return days > stale_days


def _is_expired(last_verified: str | None, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> bool:
    """Hard-exclude threshold. NULL is NOT expired (stale but usable)."""
    days = _days_since(last_verified)
    if days is None:
        return False
    return days > max_age_days


# --------------------------------------------------------------------------- #
# Content helpers
# --------------------------------------------------------------------------- #

def _escape_block_closer(content: str) -> str:
    """Defang any literal `</memory-context>` inside a memory body."""
    if BLOCK_CLOSE in content:
        return content.replace(BLOCK_CLOSE, _ESCAPED_CLOSE)
    return content


def count_tokens(text: str) -> int:
    """Best-effort token count. Uses tiktoken if available, else a 4-char heuristic."""
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def format_memory_entry(mem: dict, stale_days: int = DEFAULT_STALE_DAYS) -> str:
    """Render a single memory as a block entry with staleness marker."""
    mem_id = mem.get("id", "?")
    mem_type = mem.get("type") or "unknown"
    last_verified = mem.get("last_verified") or "never"
    stale = _is_stale(mem.get("last_verified"), stale_days)
    marker = " — STALE, verify before acting" if stale else ""
    header = f"[{mem_id} | {mem_type} | {last_verified}{marker}]"
    body = _escape_block_closer((mem.get("content") or "").strip())
    return f"{header} {body}"


# --------------------------------------------------------------------------- #
# L2 selector
# --------------------------------------------------------------------------- #

def select_l2(
    prompt: str,
    top_n: int = 3,
    floor: float = 0.5,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    pool_multiplier: int = 3,
) -> LayerResult:
    """Select the top-N prompt-relevant memories above the floor.

    Reuses `scoring.compute_score`. Does NOT call `db.record_access`.
    Excludes memories older than `max_age_days` (by `last_verified`).
    Returns a `LayerResult` with `memories` ordered by score desc.
    """
    if not prompt or not prompt.strip() or top_n <= 0:
        return LayerResult(name="l2", memories=[], budget_tokens=0, floor_score=floor)

    # Lazy import — chromadb + sentence-transformers is a ~1s cold start.
    try:
        from .embeddings import search as embed_search
    except Exception:
        return LayerResult(name="l2", memories=[], budget_tokens=0, floor_score=floor)

    # Fetch a larger candidate pool so scoring can re-rank vs. pure distance.
    pool_size = max(top_n * pool_multiplier, top_n)
    try:
        candidates = embed_search(prompt, n_results=pool_size)
    except Exception:
        return LayerResult(name="l2", memories=[], budget_tokens=0, floor_score=floor)

    if not candidates:
        return LayerResult(name="l2", memories=[], budget_tokens=0, floor_score=floor)

    query_concepts = extract_concepts(prompt)

    try:
        conn = _connect_readonly()
    except sqlite3.OperationalError:
        return LayerResult(name="l2", memories=[], budget_tokens=0, floor_score=floor)

    scored: list[dict] = []
    try:
        for hit in candidates:
            mem = _fetch_memory_readonly(conn, hit["id"])
            if not mem:
                continue
            if mem.get("status") != "active":
                continue
            if not (mem.get("content") or "").strip():
                continue
            if _is_expired(mem.get("last_verified"), max_age_days):
                continue

            score = compute_score(
                semantic_distance=hit.get("distance", 1.0),
                last_accessed=mem.get("last_accessed"),
                access_count=mem.get("access_count", 0),
                query_concepts=query_concepts,
                memory_concepts=mem.get("concepts", "[]"),
                priority=mem.get("priority", "normal"),
            )

            if score < floor:
                continue

            scored.append({
                "id": mem["id"],
                "type": mem.get("type", "unknown"),
                "last_verified": mem.get("last_verified"),
                "content": mem.get("content", ""),
                "score": score,
            })
    finally:
        conn.close()

    # Deterministic sort: score desc, then id asc for tiebreak.
    scored.sort(key=lambda m: (-m["score"], m["id"]))
    return LayerResult(
        name="l2",
        memories=scored[:top_n],
        budget_tokens=0,  # filled in by caller before build_block
        floor_score=floor,
    )


# --------------------------------------------------------------------------- #
# Block assembly
# --------------------------------------------------------------------------- #

def build_block(
    layers: list[LayerResult],
    total_budget: int,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> str:
    """Assemble a memory-context block from one or more layers.

    - Dedupes by memory ID across layers; earlier layers take precedence.
    - Applies each layer's `budget_tokens` cap.
    - Applies the overall `total_budget` cap across all layers combined.
    - Returns the empty string if nothing made it through.
    """
    if not layers or total_budget <= 0:
        return ""

    seen: set[str] = set()
    rendered_entries: list[str] = []
    total_used = 0

    for layer in layers:
        layer_used = 0
        for mem in layer.memories:
            mem_id = mem.get("id")
            if not mem_id or mem_id in seen:
                continue

            entry = format_memory_entry(mem, stale_days=stale_days)
            entry_tokens = count_tokens(entry)

            if layer_used + entry_tokens > layer.budget_tokens:
                continue
            if total_used + entry_tokens > total_budget:
                continue

            seen.add(mem_id)
            rendered_entries.append(entry)
            layer_used += entry_tokens
            total_used += entry_tokens

    if not rendered_entries:
        return ""

    body = "\n\n".join(rendered_entries)
    return f"{BLOCK_OPEN}\n{body}\n{BLOCK_CLOSE}"
