"""Shared fixtures for tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect all data to a temp directory so tests never touch real data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("CLAUDE_MEMORY_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CLAUDE_MEMORY_COLLECTION", "test_memories")

    # Force config module to re-evaluate paths
    import claude_memory.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(cfg, "DB_PATH", data_dir / "memory.db")
    monkeypatch.setattr(cfg, "CHROMA_PATH", data_dir / "chroma")
    monkeypatch.setattr(cfg, "COLLECTION_NAME", "test_memories")

    yield data_dir
