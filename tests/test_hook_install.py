"""Tests for the UserPromptSubmit hook installer."""

import json
import stat
from pathlib import Path

import pytest

from claude_memory import hook_install


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect all paths to a temp directory so tests never touch ~/.claude/."""
    monkeypatch.setattr(hook_install, "HOME", tmp_path)
    monkeypatch.setattr(hook_install, "SHIM_DIR", tmp_path / ".claude-memory" / "hook")
    monkeypatch.setattr(
        hook_install, "SHIM_PATH", tmp_path / ".claude-memory" / "hook" / "inject.sh"
    )
    monkeypatch.setattr(hook_install, "SETTINGS_PATH", tmp_path / ".claude" / "settings.json")
    yield tmp_path


def _read_settings(home: Path) -> dict:
    return json.loads((home / ".claude" / "settings.json").read_text())


class TestShim:
    def test_writes_shim_executable(self, fake_home):
        path = hook_install.write_shim()
        assert path.exists()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & 0o100, f"shim not executable: {oct(mode)}"

    def test_shim_exits_zero_when_cli_missing(self, fake_home):
        path = hook_install.write_shim()
        content = path.read_text()
        assert "command -v claude-memory-inject" in content
        assert "exit 0" in content


class TestInstall:
    def test_creates_settings_when_missing(self, fake_home):
        hook_install.install()
        settings = _read_settings(fake_home)
        ups = settings["hooks"]["UserPromptSubmit"]
        assert len(ups) == 1
        assert "inject.sh" in ups[0]["hooks"][0]["command"]

    def test_idempotent(self, fake_home):
        hook_install.install()
        hook_install.install()
        ups = _read_settings(fake_home)["hooks"]["UserPromptSubmit"]
        assert len(ups) == 1, f"duplicate entry created: {ups}"

    def test_preserves_existing_hooks(self, fake_home):
        # Pre-seed settings with an unrelated PostToolUse hook
        settings_path = fake_home / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({
            "model": "opus",
            "hooks": {
                "PostToolUse": [
                    {"matcher": "Edit", "hooks": [{"type": "command", "command": "echo hi"}]}
                ]
            }
        }))

        hook_install.install()
        settings = _read_settings(fake_home)
        # PostToolUse preserved
        assert settings["hooks"]["PostToolUse"][0]["matcher"] == "Edit"
        # UserPromptSubmit added
        assert "UserPromptSubmit" in settings["hooks"]
        # Other top-level keys preserved
        assert settings["model"] == "opus"

    def test_backup_created_when_settings_exists(self, fake_home):
        settings_path = fake_home / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({"model": "opus"}))
        hook_install.install()
        backups = list(settings_path.parent.glob("*.bak-*"))
        assert backups, "no backup created"

    def test_print_only_does_not_write(self, fake_home, capsys):
        hook_install.install(print_only=True)
        # No settings file should be created
        assert not (fake_home / ".claude" / "settings.json").exists()
        captured = capsys.readouterr()
        assert "proposed settings.json" in captured.out

    def test_refuses_when_settings_invalid_json(self, fake_home):
        settings_path = fake_home / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text("{not json")
        with pytest.raises(SystemExit):
            hook_install.install()


class TestUninstall:
    def test_removes_our_entry(self, fake_home):
        hook_install.install()
        hook_install.uninstall()
        settings = _read_settings(fake_home)
        # hooks block should be empty or absent
        assert not settings.get("hooks", {}).get("UserPromptSubmit")

    def test_preserves_other_hooks(self, fake_home):
        settings_path = fake_home / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({
            "hooks": {
                "PostToolUse": [
                    {"matcher": "Edit", "hooks": [{"type": "command", "command": "echo hi"}]}
                ]
            }
        }))
        hook_install.install()
        hook_install.uninstall()
        settings = _read_settings(fake_home)
        assert settings["hooks"]["PostToolUse"][0]["matcher"] == "Edit"

    def test_uninstall_with_no_settings_is_safe(self, fake_home, capsys):
        hook_install.uninstall()
        captured = capsys.readouterr()
        assert "nothing to do" in captured.out
