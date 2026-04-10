"""CLI entry point for prompt-relevance auto-injection.

Invoked by the Claude Code `UserPromptSubmit` hook. Reads the prompt from
stdin, runs the L2 selector, and writes a `<memory-context>` block to
stdout. Claude Code injects stdout into the conversation before Claude
sees the prompt.

Critical safety rules (the hook must NEVER block a user prompt):
- A wall-clock watchdog aborts to empty stdout past CLAUDE_MEMORY_INJECT_TIMEOUT_MS.
- `except BaseException` catches every failure path (ImportError, corrupted
  DB, SystemExit from libraries, etc.) and emits empty stdout with exit 0.
- ChromaDB and sentence-transformers are imported lazily inside the try
  block, after kill-switch checks, so short-circuited prompts don't pay
  the ~1s cold-start cost.
- Three runtime kill switches in order: `CLAUDE_MEMORY_AUTO_INJECT=0`
  env var, `~/.claude-memory/inject.pause` file, `!nomem` prompt sigil.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


PAUSE_FILE = Path.home() / ".claude-memory" / "inject.pause"
NOMEM_SIGIL = "!nomem"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _read_prompt() -> str:
    """Read prompt from stdin (primary) or CLAUDE_USER_PROMPT env (fallback)."""
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            if data:
                return data
    except BaseException:
        pass
    return os.environ.get("CLAUDE_USER_PROMPT", "")


def _start_watchdog(timeout_ms: int) -> threading.Timer:
    """Start a wall-clock watchdog that exits the process cleanly on timeout."""
    def _timeout():
        # Best-effort audit log of the timeout before dying.
        try:
            from . import inject_audit
            inject_audit.log(
                prompt="",
                cwd=os.getcwd(),
                outcome="watchdog_timeout",
            )
        except BaseException:
            pass
        os._exit(0)

    timer = threading.Timer(timeout_ms / 1000.0, _timeout)
    timer.daemon = True
    timer.start()
    return timer


def main() -> int:
    """Entry point. Returns 0 always. Writes the block to stdout."""
    prompt = _read_prompt()
    cwd = os.getcwd()

    # --- Kill-switch: env var -------------------------------------------
    if os.environ.get("CLAUDE_MEMORY_AUTO_INJECT", "1") == "0":
        _safe_audit(prompt, cwd, outcome="killed_by_env")
        return 0

    # --- Kill-switch: pause file ----------------------------------------
    try:
        if PAUSE_FILE.exists():
            _safe_audit(prompt, cwd, outcome="killed_by_file")
            return 0
    except BaseException:
        pass

    # --- Kill-switch: prompt sigil --------------------------------------
    if prompt.lstrip().startswith(NOMEM_SIGIL):
        _safe_audit(prompt, cwd, outcome="killed_by_sigil")
        return 0

    if not prompt.strip():
        return 0

    timeout_ms = _env_int("CLAUDE_MEMORY_INJECT_TIMEOUT_MS", 2000)
    watchdog = _start_watchdog(timeout_ms)

    total_budget = _env_int("CLAUDE_MEMORY_INJECT_BUDGET", 400)
    top_n = _env_int("CLAUDE_MEMORY_INJECT_TOP_N", 3)
    floor = _env_float("CLAUDE_MEMORY_INJECT_FLOOR", 0.5)
    max_age_days = _env_int("CLAUDE_MEMORY_INJECT_MAX_AGE_DAYS", 90)

    try:
        # Lazy imports — chromadb + sentence-transformers is the cold-start cost.
        from .inject import select_l2, build_block

        layer = select_l2(
            prompt=prompt,
            top_n=top_n,
            floor=floor,
            max_age_days=max_age_days,
        )
        layer.budget_tokens = total_budget  # single-layer: per-layer == total

        block = build_block([layer], total_budget=total_budget)

        if block:
            sys.stdout.write(block)
            sys.stdout.flush()

        outcome = "injected" if block else "below_floor"
        _safe_audit(
            prompt,
            cwd,
            outcome=outcome,
            memory_ids=[m["id"] for m in layer.memories] if block else [],
            scores=[m["score"] for m in layer.memories] if block else [],
            total_tokens=_safe_count(block),
            floor_used=floor,
            budget_used=total_budget,
        )
    except BaseException:
        _safe_audit(prompt, cwd, outcome="exception")
    finally:
        try:
            watchdog.cancel()
        except BaseException:
            pass

    return 0


def _safe_audit(prompt: str, cwd: str, **kwargs) -> None:
    try:
        from . import inject_audit
        inject_audit.log(prompt=prompt, cwd=cwd, **kwargs)
    except BaseException:
        pass


def _safe_count(text: str) -> int:
    try:
        from .inject import count_tokens
        return count_tokens(text)
    except BaseException:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
