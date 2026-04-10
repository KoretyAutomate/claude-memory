"""SQLite metadata store for the memory system."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from . import config


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id            TEXT PRIMARY KEY,
            content       TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            type          TEXT NOT NULL DEFAULT 'project',
            project       TEXT,
            priority      TEXT NOT NULL DEFAULT 'normal',
            tags          TEXT DEFAULT '[]',
            concepts      TEXT DEFAULT '[]',
            access_count  INTEGER DEFAULT 0,
            last_accessed TEXT,
            created       TEXT NOT NULL,
            last_verified TEXT,
            updated       TEXT,
            status        TEXT NOT NULL DEFAULT 'active',
            source        TEXT DEFAULT 'manual',
            name          TEXT,
            description   TEXT
        );

        CREATE TABLE IF NOT EXISTS retrieval_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id     TEXT NOT NULL,
            query         TEXT NOT NULL,
            score         REAL,
            used          INTEGER DEFAULT 0,
            timestamp     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS concept_graph (
            concept       TEXT NOT NULL,
            memory_id     TEXT NOT NULL,
            weight        REAL DEFAULT 1.0,
            PRIMARY KEY (concept, memory_id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
        CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
        CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority);
        CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
        CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
        CREATE INDEX IF NOT EXISTS idx_retrieval_log_memory ON retrieval_log(memory_id);
    """)
    conn.commit()
    conn.close()


def content_hash(content: str) -> str:
    """SHA-256 hash prefix for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def insert_memory(
    id: str,
    content: str,
    type: str = "project",
    project: Optional[str] = None,
    priority: str = "normal",
    tags: Optional[list] = None,
    concepts: Optional[list] = None,
    source: str = "manual",
    name: Optional[str] = None,
    description: Optional[str] = None,
    created: Optional[str] = None,
    last_verified: Optional[str] = None,
) -> bool:
    """Insert a memory. Returns False if duplicate content exists."""
    conn = _connect()
    chash = content_hash(content)

    existing = conn.execute(
        "SELECT id FROM memories WHERE content_hash = ?", (chash,)
    ).fetchone()
    if existing:
        conn.close()
        return False

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO memories
           (id, content, content_hash, type, project, priority, tags, concepts,
            access_count, last_accessed, created, last_verified, updated, status,
            source, name, description)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, 'active', ?, ?, ?)""",
        (
            id, content, chash, type, project, priority,
            json.dumps(tags or []), json.dumps(concepts or []),
            now, created or now, last_verified, now,
            source, name, description,
        ),
    )

    for concept in (concepts or []):
        conn.execute(
            "INSERT OR IGNORE INTO concept_graph (concept, memory_id) VALUES (?, ?)",
            (concept, id),
        )

    conn.commit()
    conn.close()
    return True


def update_memory(id: str, content: Optional[str] = None, **kwargs) -> bool:
    """Update a memory by id. Returns False if not found."""
    conn = _connect()
    existing = conn.execute("SELECT id FROM memories WHERE id = ?", (id,)).fetchone()
    if not existing:
        conn.close()
        return False

    now = datetime.now(timezone.utc).isoformat()
    updates = {"updated": now}

    if content is not None:
        updates["content"] = content
        updates["content_hash"] = content_hash(content)

    for field in ("type", "project", "priority", "tags", "concepts",
                  "status", "name", "description", "last_verified"):
        if field in kwargs and kwargs[field] is not None:
            if field in ("tags", "concepts"):
                updates[field] = json.dumps(kwargs[field])
            else:
                updates[field] = kwargs[field]

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [id]
    conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()
    return True


def record_access(memory_id: str, query: str, score: float) -> None:
    """Record a retrieval event and bump access_count."""
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
        (now, memory_id),
    )
    conn.execute(
        "INSERT INTO retrieval_log (memory_id, query, score, timestamp) VALUES (?, ?, ?, ?)",
        (memory_id, query, score, now),
    )
    conn.commit()
    conn.close()


def get_memory(id: str) -> Optional[dict]:
    """Fetch a single memory by id."""
    conn = _connect()
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_active() -> list[dict]:
    """Fetch all active memories, ordered by last_accessed."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM memories WHERE status = 'active' ORDER BY last_accessed DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_status() -> dict:
    """System-wide status: counts by status/project/type, stale entries."""
    conn = _connect()

    total = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    by_status = conn.execute(
        "SELECT status, COUNT(*) as c FROM memories GROUP BY status"
    ).fetchall()
    by_project = conn.execute(
        "SELECT project, COUNT(*) as c FROM memories WHERE status='active' GROUP BY project"
    ).fetchall()
    by_type = conn.execute(
        "SELECT type, COUNT(*) as c FROM memories WHERE status='active' GROUP BY type"
    ).fetchall()
    pinned = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE priority='pinned' AND status='active'"
    ).fetchone()["c"]
    retrieval_count = conn.execute(
        "SELECT COUNT(*) as c FROM retrieval_log"
    ).fetchone()["c"]

    stale = conn.execute(
        """SELECT id, name, project, last_verified FROM memories
           WHERE status = 'active'
           AND (last_verified IS NULL OR last_verified < date('now', '-30 days'))
           ORDER BY last_verified ASC"""
    ).fetchall()

    conn.close()

    return {
        "total": total,
        "by_status": {r["status"]: r["c"] for r in by_status},
        "by_project": {(r["project"] or "global"): r["c"] for r in by_project},
        "by_type": {r["type"]: r["c"] for r in by_type},
        "pinned": pinned,
        "total_retrievals": retrieval_count,
        "stale_memories": [
            {"id": r["id"], "name": r["name"], "project": r["project"],
             "last_verified": r["last_verified"]}
            for r in stale
        ],
    }


def delete_memory(id: str) -> bool:
    """Permanently delete a memory and its retrieval log / concept edges.

    Returns False if the memory doesn't exist.
    """
    conn = _connect()
    existing = conn.execute("SELECT id FROM memories WHERE id = ?", (id,)).fetchone()
    if not existing:
        conn.close()
        return False

    conn.execute("DELETE FROM retrieval_log WHERE memory_id = ?", (id,))
    conn.execute("DELETE FROM concept_graph WHERE memory_id = ?", (id,))
    conn.execute("DELETE FROM memories WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return True


def export_all(include_archived: bool = True, include_retrieval_log: bool = False) -> dict:
    """Export all memories (and optionally retrieval log) as a serializable dict.

    The returned dict has keys: `memories`, `exported_at`, `count`, plus
    `retrieval_log` if requested. Each memory is a plain dict with all columns.
    """
    conn = _connect()

    if include_archived:
        rows = conn.execute("SELECT * FROM memories ORDER BY created").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM memories WHERE status = 'active' ORDER BY created"
        ).fetchall()

    memories = []
    for r in rows:
        mem = dict(r)
        # Parse JSON fields back to lists for cleaner output
        for field in ("tags", "concepts"):
            try:
                mem[field] = json.loads(mem.get(field) or "[]")
            except (TypeError, json.JSONDecodeError):
                mem[field] = []
        memories.append(mem)

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(memories),
        "memories": memories,
    }

    if include_retrieval_log:
        log_rows = conn.execute(
            "SELECT * FROM retrieval_log ORDER BY timestamp"
        ).fetchall()
        payload["retrieval_log"] = [dict(r) for r in log_rows]

    conn.close()
    return payload


def run_decay_sweep() -> dict:
    """Archive memories not accessed in ARCHIVE_AFTER_DAYS (skip pinned)."""
    conn = _connect()

    archived = conn.execute(
        f"""UPDATE memories SET status = 'archived'
           WHERE status = 'active'
           AND priority != 'pinned'
           AND last_accessed < datetime('now', '-{config.ARCHIVE_AFTER_DAYS} days')"""
    ).rowcount

    conn.commit()
    conn.close()
    return {"archived": archived}
