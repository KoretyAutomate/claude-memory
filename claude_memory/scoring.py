"""Retrieval scoring and concept extraction."""

import json
import math
import re
from datetime import datetime, timezone

from .config import W_SEMANTIC, W_RECENCY, W_FREQUENCY, W_CONCEPT, PINNED_MULTIPLIER


# Common English stop words for concept extraction
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "mine",
    "yours", "hers", "ours", "theirs", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "and", "but", "or", "if", "also", "e", "g", "etc",
    "use", "used", "using", "new", "get", "set", "run", "see", "file",
    "true", "false", "none", "null", "yes", "no", "via",
})


def compute_score(
    semantic_distance: float,
    last_accessed: str | None,
    access_count: int,
    query_concepts: list[str],
    memory_concepts: str | list,
    priority: str = "normal",
) -> float:
    """
    Compute retrieval score for a memory candidate.

    Formula:
        score = (W_SEMANTIC * semantic_similarity)
              + (W_RECENCY  * recency_score)
              + (W_FREQUENCY * frequency_score)
              + (W_CONCEPT  * concept_boost)

    Pinned memories get a PINNED_MULTIPLIER boost.
    """
    # Semantic similarity: cosine distance [0, 2] -> similarity [0, 1]
    semantic_sim = max(0.0, 1.0 - semantic_distance)

    # Recency: exponential decay, half-life ~23 days
    recency = 0.0
    if last_accessed:
        try:
            last_dt = datetime.fromisoformat(last_accessed)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            days_ago = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
            recency = math.exp(-0.03 * max(0, days_ago))
        except (ValueError, TypeError):
            recency = 0.0

    # Frequency: log scale, caps at 32 accesses
    frequency = min(math.log2(access_count + 1) / 5.0, 1.0)

    # Concept boost: 1.0 if any query concept matches memory concepts
    if isinstance(memory_concepts, str):
        try:
            memory_concepts = json.loads(memory_concepts)
        except (json.JSONDecodeError, TypeError):
            memory_concepts = []

    concept_boost = 0.0
    if query_concepts and memory_concepts:
        query_set = {c.lower() for c in query_concepts}
        memory_set = {c.lower() for c in memory_concepts}
        if query_set & memory_set:
            concept_boost = 1.0

    score = (
        W_SEMANTIC * semantic_sim
        + W_RECENCY * recency
        + W_FREQUENCY * frequency
        + W_CONCEPT * concept_boost
    )

    if priority == "pinned":
        score *= PINNED_MULTIPLIER

    return round(score, 4)


def extract_concepts(content: str, max_concepts: int = 5) -> list[str]:
    """
    Extract concept tags from content using keyword frequency.
    No LLM call, no external dependencies. Returns top N concepts.
    """
    words = re.findall(r'[a-z][a-z0-9_-]+', content.lower())
    words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Boost compound terms (hyphenated/underscored) as more specific
    scored = {}
    for w, count in freq.items():
        scored[w] = count * 2 if ('-' in w or '_' in w) else count

    sorted_concepts = sorted(scored.items(), key=lambda x: -x[1])
    return [c[0] for c in sorted_concepts[:max_concepts]]
