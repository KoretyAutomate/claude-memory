# Claude Memory

Semantic memory system for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via MCP (Model Context Protocol).

Gives Claude Code persistent, cross-session memory with **semantic search**, **frequency-based scoring**, **automatic decay**, and **concept extraction** -- in ~600 lines of Python.

## Why This Exists

Claude Code has a built-in file-based memory system (`MEMORY.md` + markdown files). It works well at small scale, but has no semantic search -- lookups are keyword-based. As memories grow, finding the right one becomes harder.

Claude Memory adds a semantic search layer on top, without replacing the existing system:

- **Search by meaning**, not just keywords -- "GPU memory issues" finds memories about OOM errors even if "OOM" isn't in the title
- **Frequency weighting** -- memories you use often rank higher
- **Automatic decay** -- unused memories archive after 30 days (configurable)
- **Pinned memories** -- safety-critical knowledge (rules, lessons from failures) never decays
- **Retrieval logging** -- track what memories actually get used, for data-driven tuning
- **Concept extraction** -- auto-tags memories for cross-project discovery

## Architecture

```
                  Claude Code
                      |
                  MCP (stdio)
                      |
              +-----------------+
              | claude-memory   |
              |  MCP Server     |
              +--------+--------+
                       |
          +------------+------------+
          |                         |
    +-----+------+          +------+-----+
    |   SQLite   |          |  ChromaDB  |
    |  metadata  |          |  vectors   |
    |  scoring   |          |  semantic  |
    |  logging   |          |  search    |
    +------------+          +------------+
```

**SQLite** stores metadata: access counts, timestamps, priority, concepts, tags, retrieval logs.

**ChromaDB** stores vector embeddings (all-MiniLM-L6-v2) for semantic search.

Both are local, zero cloud dependencies.

### Scoring Formula

When you search, candidates are ranked by:

```
score = (0.50 x semantic_similarity)    # How close the meaning matches
      + (0.25 x recency_score)          # Exponential decay, half-life ~23 days
      + (0.20 x frequency_score)        # log2(access_count), caps at 32
      + (0.05 x concept_boost)          # 1.0 if query/memory share a concept tag
```

**Pinned memories** get a 1.5x multiplier -- they never get buried by frequently-accessed but less important knowledge.

All weights are configurable via environment variables.

### Decay

Memories not accessed for 30 days are automatically archived (excluded from default results, still searchable). Pinned memories are exempt. The sweep runs lazily on the first search of each session.

## Installation

```bash
pip install claude-memory
```

Or from source:

```bash
git clone https://github.com/KoretyAutomate/claude-memory.git
cd claude-memory
pip install -e .
```

### Requirements

- Python 3.10+
- `chromadb >= 1.0.0`
- `mcp >= 1.0.0`

## Setup with Claude Code

### Register the MCP server

```bash
claude mcp add claude-memory -- claude-memory-server
```

Or if installed from source:

```bash
claude mcp add claude-memory -- python -m claude_memory.server
```

### Verify connection

```bash
claude mcp list
# Should show: claude-memory ... Connected
```

### Import existing memories (optional)

If you have existing markdown memory files:

```bash
# Default: reads from ~/.claude/memory/
claude-memory-migrate

# Custom directory:
claude-memory-migrate /path/to/your/memory/files
```

The migration parses YAML frontmatter (`name`, `type`, `priority`, `created`, `last_verified`) and imports the body into both SQLite and ChromaDB.

## MCP Tools

Once registered, Claude Code gets 4 tools:

### `memory_search`

Search memories by semantic similarity. Returns ranked results with scores.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language search query |
| `project` | string | null | Filter to a specific project |
| `type` | string | null | Filter by type: user, feedback, project, reference, lesson |
| `n_results` | int | 8 | Maximum results |

### `memory_write`

Store a new memory. Auto-extracts concept tags.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | string | required | The memory content |
| `type` | string | "project" | Memory type |
| `project` | string | null | Project name |
| `name` | string | null | Short name |
| `description` | string | null | One-line description |
| `priority` | string | "normal" | `pinned` or `normal` |
| `tags` | list[str] | null | Manual tags |

### `memory_update`

Update an existing memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | string | required | Memory ID |
| `content` | string | null | New content |
| `priority` | string | null | New priority |
| `last_verified` | string | null | Verification date (YYYY-MM-DD) |
| `status` | string | null | `active` or `archived` |

### `memory_status`

Returns system health: counts by status/project/type, stale entries (not verified in 30+ days), total retrievals.

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_MEMORY_DATA_DIR` | `~/.claude-memory/data` | Where SQLite + ChromaDB data lives |
| `CLAUDE_MEMORY_MD_DIR` | `~/.claude/memory` | Source directory for migration |
| `CLAUDE_MEMORY_W_SEMANTIC` | `0.50` | Semantic similarity weight |
| `CLAUDE_MEMORY_W_RECENCY` | `0.25` | Recency weight |
| `CLAUDE_MEMORY_W_FREQUENCY` | `0.20` | Frequency weight |
| `CLAUDE_MEMORY_W_CONCEPT` | `0.05` | Concept boost weight |
| `CLAUDE_MEMORY_PINNED_MULT` | `1.5` | Score multiplier for pinned memories |
| `CLAUDE_MEMORY_ARCHIVE_DAYS` | `30` | Days until unused memories are archived |
| `CLAUDE_MEMORY_COLLECTION` | `claude_memories` | ChromaDB collection name |

## Memory Frontmatter

When writing markdown memory files, include these frontmatter fields:

```yaml
---
name: Descriptive name
description: One-line description for index display
type: project          # user, feedback, project, reference, lesson
priority: pinned       # pinned (never decays) or normal
created: 2026-04-09
last_verified: 2026-04-09
---
```

**When to pin**: Safety rules, architecture decisions, lessons from production failures -- anything that's critical but rarely accessed.

## Development

```bash
git clone https://github.com/KoretyAutomate/claude-memory.git
cd claude-memory
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Design Decisions

- **Supplement, not replacement** -- markdown files remain the primary storage. This adds semantic search on top.
- **No LLM calls for indexing** -- concept extraction uses TF-based keyword frequency, keeping it fast and fully offline.
- **No compression** -- memories are stored verbatim. Lossy compression (like AAAK) is [proven to reduce accuracy](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS-mempalace.md).
- **Pinned priority** -- frequency-weighted systems dangerously deprioritize rare-but-critical knowledge (e.g., "never force-push to main"). Pinned memories solve this.
- **Retrieval logging** -- every search result is logged with query, score, and timestamp. This enables future analysis of what memories are actually useful.
- **Lazy decay** -- the archive sweep runs on first search per session, not via cron. No background processes to manage.

## License

MIT
