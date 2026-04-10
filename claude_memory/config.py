"""Configuration for Claude Memory.

All paths and settings are configurable via environment variables.
Defaults follow XDG conventions with ~/.claude-memory as the base.
"""

import os
from pathlib import Path


def _data_dir() -> Path:
    """Resolve data directory from env or default."""
    if env := os.environ.get("CLAUDE_MEMORY_DATA_DIR"):
        return Path(env)
    return Path.home() / ".claude-memory" / "data"


def _memory_dir() -> Path:
    """Resolve the directory containing markdown memory files for migration."""
    if env := os.environ.get("CLAUDE_MEMORY_MD_DIR"):
        return Path(env)
    # Default: Claude Code's standard memory location
    return Path.home() / ".claude" / "memory"


DATA_DIR = _data_dir()
DB_PATH = DATA_DIR / "memory.db"
CHROMA_PATH = DATA_DIR / "chroma"
MEMORY_MD_DIR = _memory_dir()

# Scoring weights (override via env for tuning)
W_SEMANTIC = float(os.environ.get("CLAUDE_MEMORY_W_SEMANTIC", "0.50"))
W_RECENCY = float(os.environ.get("CLAUDE_MEMORY_W_RECENCY", "0.25"))
W_FREQUENCY = float(os.environ.get("CLAUDE_MEMORY_W_FREQUENCY", "0.20"))
W_CONCEPT = float(os.environ.get("CLAUDE_MEMORY_W_CONCEPT", "0.05"))
PINNED_MULTIPLIER = float(os.environ.get("CLAUDE_MEMORY_PINNED_MULT", "1.5"))

# Decay thresholds (days)
ARCHIVE_AFTER_DAYS = int(os.environ.get("CLAUDE_MEMORY_ARCHIVE_DAYS", "30"))

# ChromaDB collection name
COLLECTION_NAME = os.environ.get("CLAUDE_MEMORY_COLLECTION", "claude_memories")
