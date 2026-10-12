import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.env import load_env, require_key


def test_values_load_from_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('A_KEY="value-a"\nB_KEY=value-b\n', encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)

    assert load_env(env_file) is True
    assert os.environ["A_KEY"] == "value-a"
    assert os.environ["B_KEY"] == "value-b"


def test_exported_variable_beats_the_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("A_KEY", "from-shell")

    load_env(env_file)

    assert os.environ["A_KEY"] == "from-shell"


def test_unedited_template_does_not_register_as_set(tmp_path, monkeypatch):
    """`cp .env.example .env` leaves KEY= -- that must not count as configured."""
    env_file = tmp_path / ".env"
    env_file.write_text("A_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)

    load_env(env_file)

    assert "A_KEY" not in os.environ


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") is False


def test_require_key_reports_every_accepted_name(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.eval.env.ENV_PATH", tmp_path / "absent.env")
    monkeypatch.delenv("FIRST_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        require_key("FIRST_KEY", "SECOND_KEY")

    assert "FIRST_KEY" in str(excinfo.value) and "SECOND_KEY" in str(excinfo.value)


def test_example_template_is_tracked_and_has_no_values():
    example = ROOT / ".env.example"

    assert example.is_file(), ".env.example should be committed as the template"
    for line in example.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            assert value.strip() == "", f"{key} must not carry a real value"
