"""Audit log for prompt-relevance auto-injection.

First-class measurement device, not a removable hack. The log exists so we
can answer "is injection actually helping?" with data:
- What fraction of prompts trigger injection?
- What's the score distribution of injected memories?
- How often does the budget cap bite?
- How often does the watchdog fire?

Privacy: only salted SHA256 hashes of the prompt and cwd are stored — never
the raw prompt, the raw cwd, or the memory content. The salt is a 16-byte
random value generated on first use, stored at ~/.claude-memory/audit_salt
with mode 0600. Rotating or deleting the salt invalidates the ability to
correlate across days.

Disabled entries: the function silently returns on any failure — disk full,
permission denied, missing directory. Audit must NEVER propagate errors up
to the CLI or the user prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path


AUDIT_DIR = Path.home() / ".claude-memory" / "inject_audit"
SALT_PATH = Path.home() / ".claude-memory" / "audit_salt"


def _enabled() -> bool:
    return os.environ.get("CLAUDE_MEMORY_INJECT_AUDIT", "1") == "1"


def _load_or_create_salt() -> bytes:
    """Load the per-user audit salt, creating it on first use.

    Returns b"" on any failure — the caller still writes the log line with a
    degraded (un-salted) hash rather than silently dropping the entry.
    """
    try:
        if SALT_PATH.exists():
            return SALT_PATH.read_bytes()
        SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(SALT_PATH.parent, 0o700)
        except OSError:
            pass
        salt = secrets.token_bytes(16)
        SALT_PATH.write_bytes(salt)
        try:
            os.chmod(SALT_PATH, 0o600)
        except OSError:
            pass
        return salt
    except OSError:
        return b""


def _hash(value: str, salt: bytes) -> str:
    h = hashlib.sha256()
    h.update(salt)
    h.update(value.encode("utf-8", errors="replace"))
    return h.hexdigest()[:32]


def _ensure_audit_dir() -> Path | None:
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(AUDIT_DIR, 0o700)
        except OSError:
            pass
        return AUDIT_DIR
    except OSError:
        return None


def log(
    prompt: str,
    cwd: str,
    outcome: str,
    memory_ids: list[str] | None = None,
    scores: list[float] | None = None,
    total_tokens: int = 0,
    floor_used: float = 0.0,
    budget_used: int = 0,
) -> None:
    """Append one JSONL entry to the audit log.

    `outcome` is one of: injected, below_floor, killed_by_sigil,
    killed_by_file, killed_by_env, watchdog_timeout, exception.

    This function MUST NOT raise — any error is swallowed so injection
    cannot be blocked by audit-log problems.
    """
    if not _enabled():
        return

    try:
        audit_dir = _ensure_audit_dir()
        if audit_dir is None:
            return

        salt = _load_or_create_salt()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": _hash(prompt, salt),
            "prompt_len": len(prompt),
            "cwd_hash": _hash(cwd, salt),
            "memory_ids": memory_ids or [],
            "scores": [round(s, 4) for s in (scores or [])],
            "total_tokens": total_tokens,
            "floor_used": floor_used,
            "budget_used": budget_used,
            "outcome": outcome,
        }

        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path = audit_dir / f"{day}.jsonl"
        line = json.dumps(entry, ensure_ascii=False) + "\n"

        # Open with mode 0600 on creation
        fd = os.open(
            str(log_path),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except BaseException:
        # Never propagate — audit is a best-effort observer.
        return
