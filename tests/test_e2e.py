"""End-to-end tests for the full memory system."""

from pathlib import Path

from claude_memory.db import init_db, insert_memory, get_memory, get_status, record_access
from claude_memory.embeddings import add_memory as embed_add, search, count, delete_memory
from claude_memory.scoring import compute_score, extract_concepts
from claude_memory.migrate import migrate, parse_frontmatter


class TestFullWorkflow:
    def test_write_search_update_cycle(self):
        """Complete lifecycle: write -> search -> access -> verify scoring."""
        init_db()

        # Write
        concepts = extract_concepts("Authentication system uses JWT tokens with RSA256 signing")
        insert_memory(
            id="e2e-1",
            content="Authentication system uses JWT tokens with RSA256 signing",
            type="project",
            project="WebApp",
            priority="normal",
            concepts=concepts,
        )
        embed_add("e2e-1", "Authentication system uses JWT tokens with RSA256 signing", {
            "type": "project", "project": "WebApp", "priority": "normal", "status": "active",
        })

        # Search
        results = search("JWT authentication tokens", n_results=3)
        assert len(results) > 0
        assert results[0]["id"] == "e2e-1"

        # Record access
        record_access("e2e-1", "JWT authentication tokens", 0.85)
        mem = get_memory("e2e-1")
        assert mem["access_count"] == 1

        # Score should reflect access
        score = compute_score(
            semantic_distance=results[0]["distance"],
            last_accessed=mem["last_accessed"],
            access_count=mem["access_count"],
            query_concepts=extract_concepts("JWT authentication"),
            memory_concepts=mem["concepts"],
        )
        assert score > 0.3

    def test_pinned_never_lost(self):
        """Pinned memories always score higher than equivalent normal ones."""
        init_db()

        insert_memory(id="e2e-pin", content="Never force-push to main branch",
                       type="feedback", priority="pinned", concepts=["git", "safety"])
        insert_memory(id="e2e-norm", content="Use feature branches for development",
                       type="feedback", priority="normal", concepts=["git", "workflow"])
        embed_add("e2e-pin", "Never force-push to main branch",
                  {"type": "feedback", "priority": "pinned", "status": "active", "project": ""})
        embed_add("e2e-norm", "Use feature branches for development",
                  {"type": "feedback", "priority": "normal", "status": "active", "project": ""})

        results = search("git branch safety rules", n_results=2)
        pin_result = next(r for r in results if r["id"] == "e2e-pin")
        norm_result = next(r for r in results if r["id"] == "e2e-norm")

        pin_mem = get_memory("e2e-pin")
        norm_mem = get_memory("e2e-norm")

        pin_score = compute_score(
            pin_result["distance"], pin_mem["last_accessed"],
            pin_mem["access_count"], ["git", "safety"], pin_mem["concepts"], "pinned",
        )
        norm_score = compute_score(
            norm_result["distance"], norm_mem["last_accessed"],
            norm_mem["access_count"], ["git", "safety"], norm_mem["concepts"], "normal",
        )
        assert pin_score > norm_score

    def test_cross_project_retrieval(self):
        """Memories from different projects are retrievable by concept."""
        init_db()

        insert_memory(id="e2e-proj-a", content="GPU runs out of memory when loading large models",
                       type="lesson", project="ProjectA", concepts=["gpu", "oom"])
        insert_memory(id="e2e-proj-b", content="Server crashes with out of memory error on GPU",
                       type="lesson", project="ProjectB", concepts=["gpu", "oom"])
        embed_add("e2e-proj-a", "GPU runs out of memory when loading large models",
                  {"type": "lesson", "project": "ProjectA", "priority": "normal", "status": "active"})
        embed_add("e2e-proj-b", "Server crashes with out of memory error on GPU",
                  {"type": "lesson", "project": "ProjectB", "priority": "normal", "status": "active"})

        # Unfiltered search returns both
        results = search("GPU memory issues", n_results=5)
        ids = [r["id"] for r in results]
        assert "e2e-proj-a" in ids
        assert "e2e-proj-b" in ids

        # Filtered search returns only one project
        results = search("GPU memory issues", n_results=5, project="ProjectA")
        ids = [r["id"] for r in results]
        assert "e2e-proj-a" in ids
        assert "e2e-proj-b" not in ids


class TestMigration:
    def test_parse_frontmatter(self):
        text = """---
name: Test Memory
description: A test
type: project
priority: pinned
created: 2026-01-01
---

# Content here

Some body text.
"""
        meta, body = parse_frontmatter(text)
        assert meta["name"] == "Test Memory"
        assert meta["type"] == "project"
        assert meta["priority"] == "pinned"
        assert "# Content here" in body
        assert "Some body text." in body

    def test_parse_no_frontmatter(self):
        text = "Just plain content without frontmatter."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_migrate_directory(self, tmp_path):
        init_db()

        # Create test markdown files
        (tmp_path / "note1.md").write_text("""---
name: Note One
type: project
priority: normal
---

This is the first test note about database migrations.
""")
        (tmp_path / "note2.md").write_text("""---
name: Note Two
type: feedback
priority: pinned
---

Always run tests before deploying to production.
""")
        # Index file should be skipped
        (tmp_path / "MEMORY.md").write_text("# Index\n- Note 1\n- Note 2\n")

        result = migrate(tmp_path)
        assert result["migrated"] == 2
        assert result["skipped"] == 0

        # Verify in DB
        mem1 = get_memory("note1")
        assert mem1 is not None
        assert mem1["name"] == "Note One"

        mem2 = get_memory("note2")
        assert mem2 is not None
        assert mem2["priority"] == "pinned"

    def test_migrate_idempotent(self, tmp_path):
        init_db()

        (tmp_path / "idem.md").write_text("---\nname: Idem\n---\n\nIdempotent test content.")

        result1 = migrate(tmp_path)
        assert result1["migrated"] == 1

        result2 = migrate(tmp_path)
        assert result2["migrated"] == 0
        assert result2["skipped"] == 1

    def test_migrate_skips_empty(self, tmp_path):
        init_db()

        (tmp_path / "empty.md").write_text("---\nname: Empty\n---\n\n")

        result = migrate(tmp_path)
        assert result["skipped"] == 1
