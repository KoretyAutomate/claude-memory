"""Migrate markdown memory files into SQLite + ChromaDB.

Reads .md files from a directory, parses YAML-like frontmatter + body,
and imports them into the memory system. Idempotent (deduplicates by content hash).

Usage:
    claude-memory-migrate                          # uses default dir
    claude-memory-migrate /path/to/memory/files    # custom dir
"""

import re
import sys
from pathlib import Path

from .db import init_db, insert_memory
from .embeddings import add_memory as embed_add
from .scoring import extract_concepts
from .config import MEMORY_MD_DIR


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from markdown. Returns (metadata, body)."""
    meta = {}
    body = text

    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if match:
        fm_text = match.group(1)
        body = text[match.end():]

        for line in fm_text.strip().split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip()
                meta[key] = value

    return meta, body.strip()


def migrate(memory_dir: Path | None = None) -> dict:
    """
    Import all .md files from a directory into the memory system.

    Args:
        memory_dir: Directory containing .md files. Defaults to MEMORY_MD_DIR.

    Returns:
        dict with 'migrated' and 'skipped' counts.
    """
    init_db()
    target_dir = memory_dir or MEMORY_MD_DIR

    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return {"migrated": 0, "skipped": 0, "error": "directory not found"}

    md_files = sorted(target_dir.glob("*.md"))
    # Skip index files
    md_files = [f for f in md_files if f.name.upper() not in ("MEMORY.MD", "README.MD", "INDEX.MD")]

    print(f"Found {len(md_files)} memory files in {target_dir}\n")

    migrated = 0
    skipped = 0

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)

        if not body.strip():
            print(f"  [SKIP] {md_file.name} (empty body)")
            skipped += 1
            continue

        name = meta.get("name", md_file.stem)
        description = meta.get("description", "")
        mem_type = meta.get("type", "project")
        priority = meta.get("priority", "normal")
        created = meta.get("created", "")
        last_verified = meta.get("last_verified", "")

        concepts = extract_concepts(body)
        mem_id = md_file.stem

        success = insert_memory(
            id=mem_id,
            content=body,
            type=mem_type,
            priority=priority,
            concepts=concepts,
            source=f"migration:{md_file.name}",
            name=name,
            description=description,
            created=created or None,
            last_verified=last_verified or None,
        )

        if success:
            embed_add(mem_id, body, {
                "type": mem_type,
                "project": "",
                "priority": priority,
                "status": "active",
            })
            migrated += 1
            print(f"  [OK] {md_file.name} -> id={mem_id}, concepts={concepts[:3]}")
        else:
            skipped += 1
            print(f"  [SKIP] {md_file.name} (duplicate content)")

    print(f"\nMigration complete: {migrated} imported, {skipped} skipped.")
    return {"migrated": migrated, "skipped": skipped}


def main():
    """CLI entry point."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = MEMORY_MD_DIR

    migrate(target)


if __name__ == "__main__":
    main()
