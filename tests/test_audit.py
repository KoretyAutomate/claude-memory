"""Tests for the auto-injection audit log."""

import json
import os
import stat
from pathlib import Path

import pytest

from claude_memory import inject_audit


@pytest.fixture
def isolated_audit(tmp_path, monkeypatch):
    """Redirect the audit log and salt file to a temp location."""
    audit_dir = tmp_path / "inject_audit"
    salt_path = tmp_path / "audit_salt"
    monkeypatch.setattr(inject_audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(inject_audit, "SALT_PATH", salt_path)
    monkeypatch.setenv("CLAUDE_MEMORY_INJECT_AUDIT", "1")
    yield audit_dir, salt_path


def _read_last_line(audit_dir: Path) -> dict:
    files = sorted(audit_dir.glob("*.jsonl"))
    assert files, f"no audit files in {audit_dir}"
    lines = files[-1].read_text(encoding="utf-8").splitlines()
    assert lines, f"audit file empty: {files[-1]}"
    return json.loads(lines[-1])


class TestLogging:
    def test_writes_line_when_enabled(self, isolated_audit):
        audit_dir, _ = isolated_audit
        inject_audit.log(
            prompt="hello world",
            cwd="/tmp/project",
            outcome="injected",
            memory_ids=["m1", "m2"],
            scores=[0.91, 0.77],
            total_tokens=120,
            floor_used=0.5,
            budget_used=400,
        )
        entry = _read_last_line(audit_dir)
        assert entry["outcome"] == "injected"
        assert entry["memory_ids"] == ["m1", "m2"]
        assert entry["scores"] == [0.91, 0.77]
        assert entry["prompt_len"] == len("hello world")
        assert entry["total_tokens"] == 120

    def test_silent_when_disabled(self, isolated_audit, monkeypatch):
        audit_dir, _ = isolated_audit
        monkeypatch.setenv("CLAUDE_MEMORY_INJECT_AUDIT", "0")
        inject_audit.log(prompt="x", cwd="/tmp", outcome="injected")
        assert not list(audit_dir.glob("*.jsonl")), "log file should not exist when disabled"

    def test_no_raw_prompt_or_cwd_in_entry(self, isolated_audit):
        audit_dir, _ = isolated_audit
        inject_audit.log(
            prompt="my secret prompt about credit card 4111",
            cwd="/home/user/secret/project",
            outcome="injected",
        )
        entry = _read_last_line(audit_dir)
        raw = json.dumps(entry)
        assert "secret" not in raw
        assert "4111" not in raw
        assert "/home/user" not in raw
        assert "prompt_hash" in entry
        assert "cwd_hash" in entry
        assert len(entry["prompt_hash"]) == 32
        assert len(entry["cwd_hash"]) == 32


class TestSalt:
    def test_salt_created_on_first_use(self, isolated_audit):
        audit_dir, salt_path = isolated_audit
        assert not salt_path.exists()
        inject_audit.log(prompt="x", cwd="/tmp", outcome="injected")
        assert salt_path.exists()
        assert len(salt_path.read_bytes()) == 16

    def test_salt_reused_across_calls(self, isolated_audit):
        audit_dir, salt_path = isolated_audit
        inject_audit.log(prompt="x", cwd="/tmp", outcome="injected")
        first = salt_path.read_bytes()
        inject_audit.log(prompt="y", cwd="/tmp", outcome="injected")
        second = salt_path.read_bytes()
        assert first == second

    def test_same_prompt_same_salt_same_hash(self, isolated_audit):
        audit_dir, _ = isolated_audit
        inject_audit.log(prompt="repeatable", cwd="/tmp", outcome="injected")
        inject_audit.log(prompt="repeatable", cwd="/tmp", outcome="injected")
        files = sorted(audit_dir.glob("*.jsonl"))
        lines = files[-1].read_text().splitlines()
        entries = [json.loads(l) for l in lines]
        assert entries[0]["prompt_hash"] == entries[1]["prompt_hash"]

    def test_different_salt_different_hash(self, isolated_audit, tmp_path):
        audit_dir, salt_path = isolated_audit
        inject_audit.log(prompt="same text", cwd="/tmp", outcome="injected")
        first_hash = _read_last_line(audit_dir)["prompt_hash"]

        # Clear audit + rotate salt
        for f in audit_dir.glob("*.jsonl"):
            f.unlink()
        salt_path.unlink()

        inject_audit.log(prompt="same text", cwd="/tmp", outcome="injected")
        second_hash = _read_last_line(audit_dir)["prompt_hash"]
        assert first_hash != second_hash


class TestPermissions:
    def test_audit_dir_is_0700(self, isolated_audit):
        audit_dir, _ = isolated_audit
        inject_audit.log(prompt="x", cwd="/tmp", outcome="injected")
        mode = stat.S_IMODE(audit_dir.stat().st_mode)
        assert mode == 0o700, f"audit dir mode {oct(mode)} != 0o700"

    def test_audit_file_is_0600(self, isolated_audit):
        audit_dir, _ = isolated_audit
        inject_audit.log(prompt="x", cwd="/tmp", outcome="injected")
        files = list(audit_dir.glob("*.jsonl"))
        assert files
        mode = stat.S_IMODE(files[0].stat().st_mode)
        assert mode == 0o600, f"audit file mode {oct(mode)} != 0o600"

    def test_salt_file_is_0600(self, isolated_audit):
        _, salt_path = isolated_audit
        inject_audit.log(prompt="x", cwd="/tmp", outcome="injected")
        mode = stat.S_IMODE(salt_path.stat().st_mode)
        assert mode == 0o600, f"salt file mode {oct(mode)} != 0o600"


class TestResilience:
    def test_permission_denied_does_not_raise(self, isolated_audit, monkeypatch):
        """If the audit dir can't be created, log() returns silently."""
        audit_dir, _ = isolated_audit

        def boom(*args, **kwargs):
            raise PermissionError("nope")

        monkeypatch.setattr(Path, "mkdir", boom)
        # Must not raise
        inject_audit.log(prompt="x", cwd="/tmp", outcome="injected")

    def test_invalid_utf8_handled(self, isolated_audit):
        """Prompts with non-UTF8 surrogates should not crash."""
        audit_dir, _ = isolated_audit
        bad = "hello \ud83d world"  # unpaired surrogate
        inject_audit.log(prompt=bad, cwd="/tmp", outcome="injected")
        entry = _read_last_line(audit_dir)
        assert entry["outcome"] == "injected"
