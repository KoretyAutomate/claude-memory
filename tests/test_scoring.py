"""Tests for scoring algorithm and concept extraction."""

from datetime import datetime, timezone

from claude_memory.scoring import compute_score, extract_concepts


class TestComputeScore:
    def test_perfect_match(self):
        """Distance 0 = perfect semantic match."""
        score = compute_score(
            semantic_distance=0.0,
            last_accessed=datetime.now(timezone.utc).isoformat(),
            access_count=10,
            query_concepts=["python"],
            memory_concepts=["python", "code"],
        )
        assert score > 0.8

    def test_poor_match(self):
        """High distance = poor semantic match."""
        score = compute_score(
            semantic_distance=1.5,
            last_accessed=None,
            access_count=0,
            query_concepts=[],
            memory_concepts=[],
        )
        assert score < 0.2

    def test_pinned_boost(self):
        """Pinned memories score higher than normal."""
        kwargs = dict(
            semantic_distance=0.3,
            last_accessed=datetime.now(timezone.utc).isoformat(),
            access_count=5,
            query_concepts=["test"],
            memory_concepts=["test"],
        )
        pinned = compute_score(**kwargs, priority="pinned")
        normal = compute_score(**kwargs, priority="normal")
        assert pinned > normal
        assert abs(pinned / normal - 1.5) < 0.01

    def test_recency_matters(self):
        """Recent access scores higher than old access."""
        recent = compute_score(
            semantic_distance=0.5,
            last_accessed=datetime.now(timezone.utc).isoformat(),
            access_count=1,
            query_concepts=[],
            memory_concepts=[],
        )
        old = compute_score(
            semantic_distance=0.5,
            last_accessed="2020-01-01T00:00:00+00:00",
            access_count=1,
            query_concepts=[],
            memory_concepts=[],
        )
        assert recent > old

    def test_frequency_matters(self):
        """Higher access count scores higher."""
        frequent = compute_score(
            semantic_distance=0.5,
            last_accessed=datetime.now(timezone.utc).isoformat(),
            access_count=20,
            query_concepts=[],
            memory_concepts=[],
        )
        rare = compute_score(
            semantic_distance=0.5,
            last_accessed=datetime.now(timezone.utc).isoformat(),
            access_count=0,
            query_concepts=[],
            memory_concepts=[],
        )
        assert frequent > rare

    def test_concept_boost(self):
        """Matching concepts add a boost."""
        with_concept = compute_score(
            semantic_distance=0.5,
            last_accessed=None,
            access_count=0,
            query_concepts=["gpu", "memory"],
            memory_concepts=["gpu", "cuda"],
        )
        without_concept = compute_score(
            semantic_distance=0.5,
            last_accessed=None,
            access_count=0,
            query_concepts=["gpu", "memory"],
            memory_concepts=["unrelated"],
        )
        assert with_concept > without_concept

    def test_memory_concepts_as_json_string(self):
        """Accepts JSON string for memory_concepts."""
        score = compute_score(
            semantic_distance=0.3,
            last_accessed=None,
            access_count=0,
            query_concepts=["test"],
            memory_concepts='["test", "example"]',
        )
        assert score > 0

    def test_null_last_accessed(self):
        """Handles None last_accessed gracefully."""
        score = compute_score(
            semantic_distance=0.5,
            last_accessed=None,
            access_count=0,
            query_concepts=[],
            memory_concepts=[],
        )
        assert score >= 0


class TestExtractConcepts:
    def test_basic_extraction(self):
        concepts = extract_concepts("Python is great for machine learning and data science")
        assert len(concepts) > 0
        assert all(isinstance(c, str) for c in concepts)

    def test_compound_terms_boosted(self):
        """Hyphenated/underscored terms should rank higher."""
        concepts = extract_concepts(
            "The gpu-memory management system handles cuda-cores efficiently. "
            "Python code runs the memory allocation."
        )
        compound = [c for c in concepts if '-' in c or '_' in c]
        assert len(compound) > 0

    def test_stop_words_excluded(self):
        concepts = extract_concepts("the is a an this that which what where when")
        assert len(concepts) == 0

    def test_max_concepts(self):
        concepts = extract_concepts("a b c d e f g h i j " * 10, max_concepts=3)
        assert len(concepts) <= 3

    def test_empty_content(self):
        concepts = extract_concepts("")
        assert concepts == []
