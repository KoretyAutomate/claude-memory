"""Tests for SQLite metadata store."""

from claude_memory.db import (
    init_db, insert_memory, update_memory, get_memory,
    get_all_active, get_status, record_access, run_decay_sweep,
    content_hash,
)


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_content(self):
        assert content_hash("hello") != content_hash("world")

    def test_length(self):
        assert len(content_hash("test")) == 16


class TestInsertMemory:
    def test_basic_insert(self):
        init_db()
        result = insert_memory(id="test-1", content="Test memory content")
        assert result is True

    def test_duplicate_rejected(self):
        init_db()
        insert_memory(id="dup-orig", content="Unique duplicate test content")
        result = insert_memory(id="dup-copy", content="Unique duplicate test content")
        assert result is False

    def test_with_all_fields(self):
        init_db()
        result = insert_memory(
            id="full-1",
            content="Full memory with all fields",
            type="lesson",
            project="MyProject",
            priority="pinned",
            tags=["tag1", "tag2"],
            concepts=["concept1"],
            source="test",
            name="Full Memory",
            description="A test memory",
            created="2026-01-01",
            last_verified="2026-01-01",
        )
        assert result is True

        mem = get_memory("full-1")
        assert mem["type"] == "lesson"
        assert mem["project"] == "MyProject"
        assert mem["priority"] == "pinned"
        assert mem["name"] == "Full Memory"


class TestUpdateMemory:
    def test_update_content(self):
        init_db()
        insert_memory(id="upd-1", content="Original content")
        result = update_memory("upd-1", content="Updated content")
        assert result is True

        mem = get_memory("upd-1")
        assert mem["content"] == "Updated content"

    def test_update_priority(self):
        init_db()
        insert_memory(id="upd-2", content="Some content")
        update_memory("upd-2", priority="pinned")

        mem = get_memory("upd-2")
        assert mem["priority"] == "pinned"

    def test_update_nonexistent(self):
        init_db()
        result = update_memory("nonexistent", content="new")
        assert result is False


class TestGetMemory:
    def test_found(self):
        init_db()
        insert_memory(id="get-1", content="Findable memory")
        mem = get_memory("get-1")
        assert mem is not None
        assert mem["content"] == "Findable memory"

    def test_not_found(self):
        init_db()
        assert get_memory("nonexistent") is None


class TestGetAllActive:
    def test_returns_only_active(self):
        init_db()
        insert_memory(id="active-1", content="Active memory one")
        insert_memory(id="active-2", content="Active memory two")
        insert_memory(id="archived-1", content="Archived memory")
        update_memory("archived-1", status="archived")

        active = get_all_active()
        ids = [m["id"] for m in active]
        assert "active-1" in ids
        assert "active-2" in ids
        assert "archived-1" not in ids


class TestRecordAccess:
    def test_increments_count(self):
        init_db()
        insert_memory(id="acc-1", content="Access test memory")

        record_access("acc-1", "test query", 0.85)
        mem = get_memory("acc-1")
        assert mem["access_count"] == 1

        record_access("acc-1", "another query", 0.72)
        mem = get_memory("acc-1")
        assert mem["access_count"] == 2


class TestGetStatus:
    def test_counts(self):
        init_db()
        insert_memory(id="st-1", content="Status test one", type="project", priority="pinned")
        insert_memory(id="st-2", content="Status test two", type="feedback")
        insert_memory(id="st-3", content="Status test three", type="project")

        status = get_status()
        assert status["total"] == 3
        assert status["pinned"] == 1
        assert status["by_type"]["project"] == 2
        assert status["by_type"]["feedback"] == 1


class TestDecaySweep:
    def test_skips_pinned(self):
        init_db()
        insert_memory(id="decay-pinned", content="Pinned memory", priority="pinned")
        result = run_decay_sweep()
        mem = get_memory("decay-pinned")
        assert mem["status"] == "active"

    def test_skips_recently_accessed(self):
        init_db()
        insert_memory(id="decay-recent", content="Recent memory")
        result = run_decay_sweep()
        mem = get_memory("decay-recent")
        assert mem["status"] == "active"
