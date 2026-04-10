"""Tests for prompt-relevance auto-injection (L2-only MVP)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from claude_memory import inject
from claude_memory.inject import (
    LayerResult,
    build_block,
    count_tokens,
    format_memory_entry,
    select_l2,
    _is_stale,
    _is_expired,
    _escape_block_closer,
    BLOCK_OPEN,
    BLOCK_CLOSE,
)
from claude_memory.db import init_db, insert_memory
from claude_memory.embeddings import add_memory as embed_add


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")


def _insert(id_, content, **kwargs):
    kwargs.setdefault("type", "project")
    kwargs.setdefault("last_verified", _iso_days_ago(1))
    insert_memory(id=id_, content=content, **kwargs)
    embed_add(id_, content, {
        "type": kwargs["type"],
        "project": kwargs.get("project") or "",
        "priority": kwargs.get("priority", "normal"),
        "status": "active",
    })


# --------------------------------------------------------------------------- #
# Staleness helpers
# --------------------------------------------------------------------------- #

class TestStaleness:
    def test_null_last_verified_is_stale(self):
        assert _is_stale(None) is True

    def test_recent_not_stale(self):
        assert _is_stale(_iso_days_ago(5)) is False

    def test_old_is_stale(self):
        assert _is_stale(_iso_days_ago(45)) is True

    def test_null_not_expired(self):
        # NULL is stale but not hard-excluded
        assert _is_expired(None) is False

    def test_beyond_max_age_expired(self):
        assert _is_expired(_iso_days_ago(120)) is True

    def test_within_max_age_not_expired(self):
        assert _is_expired(_iso_days_ago(80)) is False


# --------------------------------------------------------------------------- #
# Content escape
# --------------------------------------------------------------------------- #

class TestEscapeBlockCloser:
    def test_no_closer_unchanged(self):
        assert _escape_block_closer("hello world") == "hello world"

    def test_closer_defanged(self):
        out = _escape_block_closer(f"evil {BLOCK_CLOSE} content")
        assert BLOCK_CLOSE not in out
        assert "</memory-context\u200b>" in out


# --------------------------------------------------------------------------- #
# count_tokens
# --------------------------------------------------------------------------- #

class TestCountTokens:
    def test_empty(self):
        assert count_tokens("") == 0

    def test_fallback_without_tiktoken(self, monkeypatch):
        # Force the tiktoken import to fail so we exercise the fallback branch.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert count_tokens("a" * 400) == 100

    def test_with_tiktoken_if_available(self):
        """If tiktoken is installed, count is non-zero for non-empty text."""
        assert count_tokens("hello world this is a test") > 0


# --------------------------------------------------------------------------- #
# format_memory_entry
# --------------------------------------------------------------------------- #

class TestFormatMemoryEntry:
    def test_fresh_no_stale_marker(self):
        mem = {
            "id": "m1",
            "type": "project",
            "last_verified": _iso_days_ago(3),
            "content": "fresh content",
        }
        out = format_memory_entry(mem)
        assert "m1" in out
        assert "STALE" not in out
        assert "fresh content" in out

    def test_old_has_stale_marker(self):
        mem = {
            "id": "m2",
            "type": "project",
            "last_verified": _iso_days_ago(60),
            "content": "old content",
        }
        out = format_memory_entry(mem)
        assert "STALE" in out

    def test_null_verified_has_stale_marker(self):
        mem = {"id": "m3", "type": "project", "last_verified": None, "content": "x"}
        out = format_memory_entry(mem)
        assert "STALE" in out

    def test_closer_escaped_in_body(self):
        mem = {
            "id": "m4",
            "type": "project",
            "last_verified": _iso_days_ago(1),
            "content": f"smuggle {BLOCK_CLOSE} attack",
        }
        out = format_memory_entry(mem)
        assert BLOCK_CLOSE not in out.split(" ", 1)[1]  # header has no closer


# --------------------------------------------------------------------------- #
# select_l2 — uses real DB + Chroma via the isolated_data_dir fixture
# --------------------------------------------------------------------------- #

class TestSelectL2:
    def test_empty_db(self):
        init_db()
        result = select_l2("anything", top_n=3, floor=0.0)
        assert isinstance(result, LayerResult)
        assert result.memories == []

    def test_respects_floor(self):
        init_db()
        _insert("far-1", "completely unrelated lemur facts")
        # Search for something distant; floor 0.99 will reject everything.
        result = select_l2("quantum chromodynamics", top_n=3, floor=0.99)
        assert result.memories == []

    def test_respects_top_n(self):
        init_db()
        for i in range(5):
            _insert(f"py-{i}", f"python programming tip number {i}")
        result = select_l2("python programming", top_n=2, floor=0.0)
        assert len(result.memories) <= 2

    def test_top_n_zero_returns_empty(self):
        init_db()
        _insert("p1", "python programming guide")
        result = select_l2("python", top_n=0, floor=0.0)
        assert result.memories == []

    def test_excludes_expired_memories(self):
        init_db()
        _insert("old-1", "python programming tutorial",
                last_verified=_iso_days_ago(200))
        _insert("new-1", "python programming reference",
                last_verified=_iso_days_ago(5))
        result = select_l2("python programming", top_n=5, floor=0.0, max_age_days=90)
        ids = [m["id"] for m in result.memories]
        assert "old-1" not in ids
        assert "new-1" in ids

    def test_excludes_empty_content(self):
        init_db()
        # Insert directly with empty body bypassing embed_add (empty content
        # cannot be meaningfully embedded). We simulate by inserting a row
        # with whitespace-only content.
        insert_memory(id="blank", content=" ", type="project",
                      last_verified=_iso_days_ago(1))
        embed_add("blank", " ", {"type": "project", "project": "",
                                 "priority": "normal", "status": "active"})
        result = select_l2("anything", top_n=5, floor=0.0)
        assert all(m["id"] != "blank" for m in result.memories)

    def test_does_not_call_record_access(self):
        init_db()
        _insert("x1", "test content for record access check")
        with patch("claude_memory.db.record_access") as mock_ra:
            select_l2("test content", top_n=3, floor=0.0)
            mock_ra.assert_not_called()

    def test_uses_compute_score(self):
        init_db()
        _insert("cs-1", "scoring call verification test")
        with patch("claude_memory.inject.compute_score", return_value=0.9) as mock_cs:
            result = select_l2("scoring verification", top_n=3, floor=0.0)
            assert mock_cs.called
            assert len(result.memories) >= 1

    def test_deterministic_tiebreak_by_id(self):
        """When compute_score returns identical values, IDs sort ascending."""
        init_db()
        _insert("b-tie", "alpha content for tiebreak")
        _insert("a-tie", "alpha content for tiebreak two")
        with patch("claude_memory.inject.compute_score", return_value=0.7):
            result = select_l2("alpha tiebreak", top_n=5, floor=0.0)
            ids = [m["id"] for m in result.memories]
            # Both have same score; 'a-tie' < 'b-tie' lexicographically
            if "a-tie" in ids and "b-tie" in ids:
                assert ids.index("a-tie") < ids.index("b-tie")


# --------------------------------------------------------------------------- #
# build_block
# --------------------------------------------------------------------------- #

class TestBuildBlock:
    def _mem(self, id_, content="hello world", verified_days_ago=1):
        return {
            "id": id_,
            "type": "project",
            "last_verified": _iso_days_ago(verified_days_ago),
            "content": content,
            "score": 0.9,
        }

    def test_empty_layers_returns_empty_string(self):
        assert build_block([], total_budget=400) == ""

    def test_empty_memories_returns_empty_string(self):
        layer = LayerResult(name="l2", memories=[], budget_tokens=400, floor_score=0.5)
        assert build_block([layer], total_budget=400) == ""

    def test_zero_budget_returns_empty(self):
        layer = LayerResult(
            name="l2",
            memories=[self._mem("m1")],
            budget_tokens=400,
            floor_score=0.5,
        )
        assert build_block([layer], total_budget=0) == ""

    def test_single_memory_renders(self):
        layer = LayerResult(
            name="l2",
            memories=[self._mem("m1", "hello world")],
            budget_tokens=400,
            floor_score=0.5,
        )
        out = build_block([layer], total_budget=400)
        assert out.startswith(BLOCK_OPEN)
        assert out.endswith(BLOCK_CLOSE)
        assert "m1" in out
        assert "hello world" in out

    def test_dedup_across_layers(self):
        mem = self._mem("dup1", "duplicate body")
        layer_a = LayerResult(name="l1", memories=[mem], budget_tokens=400)
        layer_b = LayerResult(name="l2", memories=[mem], budget_tokens=400)
        out = build_block([layer_a, layer_b], total_budget=400)
        # Only one occurrence of the body
        assert out.count("duplicate body") == 1

    def test_per_layer_budget_caps(self):
        """The rendered entries' token total must stay within the per-layer cap."""
        big_content = "word " * 200  # ~200 tokens via fallback
        layer = LayerResult(
            name="l2",
            memories=[
                self._mem("a", big_content),
                self._mem("b", big_content),
                self._mem("c", big_content),
            ],
            budget_tokens=250,
        )
        out = build_block([layer], total_budget=10000)
        # The memories appear at all (possibly truncated) — but total tokens
        # inside the block body (excluding the open/close tags) don't exceed
        # the layer budget.
        body = out[len(BLOCK_OPEN):-len(BLOCK_CLOSE)].strip()
        assert count_tokens(body) <= 250

    def test_total_budget_caps(self):
        """The total block size must stay within the total_budget."""
        content = "word " * 80  # ~100 tokens
        layer = LayerResult(
            name="l2",
            memories=[
                self._mem(f"m{i}", content) for i in range(5)
            ],
            budget_tokens=10000,  # per-layer is permissive
        )
        out = build_block([layer], total_budget=150)
        body = out[len(BLOCK_OPEN):-len(BLOCK_CLOSE)].strip()
        assert count_tokens(body) <= 150

    def test_truncation_fits_oversized_top_hit(self):
        """Oversized top-hit memory is truncated to fit, not skipped entirely."""
        huge = "word " * 500  # way over any reasonable per-memory budget
        layer = LayerResult(
            name="l2",
            memories=[self._mem("big", huge)],
            budget_tokens=300,
        )
        out = build_block([layer], total_budget=300)
        assert out != "", "truncation should allow oversized memory to render"
        assert "[big |" in out
        assert "…[truncated]" in out
        body = out[len(BLOCK_OPEN):-len(BLOCK_CLOSE)].strip()
        assert count_tokens(body) <= 300

    def test_stale_marker_in_block(self):
        layer = LayerResult(
            name="l2",
            memories=[self._mem("old", "old body", verified_days_ago=45)],
            budget_tokens=400,
        )
        out = build_block([layer], total_budget=400)
        assert "STALE" in out

    def test_closer_escaped_in_block_body(self):
        mem = self._mem("evil", f"smuggle {BLOCK_CLOSE} attack")
        layer = LayerResult(name="l2", memories=[mem], budget_tokens=400)
        out = build_block([layer], total_budget=400)
        # The block must still close exactly once, at the very end.
        # Strip the trailing closer and confirm no other closer remains.
        assert out.endswith(BLOCK_CLOSE)
        body = out[: -len(BLOCK_CLOSE)]
        assert BLOCK_CLOSE not in body

    def test_unicode_body_renders(self):
        mem = self._mem("u1", "こんにちは 🎌 memory content")
        layer = LayerResult(name="l2", memories=[mem], budget_tokens=400)
        out = build_block([layer], total_budget=400)
        assert "こんにちは" in out
