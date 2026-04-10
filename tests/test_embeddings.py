"""Tests for ChromaDB vector store."""

from claude_memory.embeddings import add_memory, search, delete_memory, count


class TestAddAndSearch:
    def test_add_and_count(self):
        add_memory("emb-1", "Machine learning with Python", {"type": "project"})
        assert count() >= 1

    def test_search_returns_results(self):
        add_memory("emb-s1", "Docker containers for web deployment", {"type": "project"})
        add_memory("emb-s2", "Kubernetes orchestration and scaling", {"type": "project"})
        add_memory("emb-s3", "Baking chocolate chip cookies recipe", {"type": "reference"})

        results = search("container orchestration", n_results=3)
        assert len(results) > 0
        # Container-related result should be the top hit, not cookies
        assert results[0]["id"] in ("emb-s1", "emb-s2")

    def test_search_with_project_filter(self):
        add_memory("emb-f1", "Alpha project config", {"type": "project", "project": "Alpha"})
        add_memory("emb-f2", "Beta project config", {"type": "project", "project": "Beta"})

        results = search("project config", n_results=5, project="Alpha")
        for r in results:
            assert r["metadata"].get("project") == "Alpha"

    def test_search_empty_collection(self, isolated_data_dir, monkeypatch):
        """Search on empty collection returns empty list."""
        import claude_memory.config as cfg
        monkeypatch.setattr(cfg, "COLLECTION_NAME", "empty_test_collection")
        from claude_memory import embeddings
        # Create a fresh client pointing to the new collection
        client = embeddings._get_client()
        coll = client.get_or_create_collection("empty_test_collection")
        assert coll.count() == 0

    def test_delete(self):
        add_memory("emb-del", "Temporary memory to delete", {"type": "project"})
        initial = count()
        delete_memory("emb-del")
        assert count() == initial - 1

    def test_upsert_updates(self):
        add_memory("emb-ups", "Original content", {"type": "project"})
        c1 = count()
        add_memory("emb-ups", "Updated content", {"type": "project"})
        assert count() == c1  # No duplicate


class TestSearchResults:
    def test_result_structure(self):
        add_memory("emb-struct", "Test structure memory", {"type": "feedback", "project": "X"})
        results = search("test structure", n_results=1)
        assert len(results) > 0
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "distance" in r
        assert "metadata" in r

    def test_distance_is_numeric(self):
        add_memory("emb-dist", "Distance check memory", {"type": "project"})
        results = search("distance check", n_results=1)
        assert isinstance(results[0]["distance"], (int, float))
