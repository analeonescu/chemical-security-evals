import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.run_inspect_eval import resolve_eval_dir as resolve_dir
from scripts.eval.score_eval import save_results


def test_save_results_appends_to_existing_json(tmp_path):
    output_dir = tmp_path / "results"
    output_file = output_dir / "scores_test_model.json"

    save_results([{"sample_id": "a", "log_file": "log1.eval"}], "test/model", output_dir)
    save_results([{"sample_id": "b", "log_file": "log2.eval"}], "test/model", output_dir)

    saved = output_file.read_text(encoding="utf-8")
    assert '"sample_id": "a"' in saved
    assert '"sample_id": "b"' in saved


def test_resolve_numeric(tmp_path):
    logs_root = tmp_path / "logs"
    logs_root.mkdir()
    target = logs_root / "1"
    target.mkdir()
    (target / "prompt.txt").write_text("Prompt for run 1", encoding="utf-8")

    assert resolve_dir(logs_root, 1) == target.resolve()
    assert resolve_dir(logs_root, "1") == target.resolve()
    assert resolve_dir(logs_root, str(target)) == target.resolve()
    assert (resolve_dir(logs_root, 1) / "prompt.txt").exists()


def test_resolve_none():
    assert resolve_dir(Path("logs"), None) is None


def test_resolve_missing(tmp_path):
    logs_root = tmp_path / "logs"
    logs_root.mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_dir(logs_root, 99)
