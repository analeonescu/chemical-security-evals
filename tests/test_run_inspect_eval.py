import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.dataset import (
    DATA_PATH,
    _record_to_sample,
    load_synthesis_dataset,
    resolve_dataset_path,
)
from scripts.eval.run_inspect_eval import (
    find_eval_logs_after_run,
    resolve_eval_dir as resolve_dir,
    score_logs,
)
from scripts.eval.scorer import DIMENSIONS, validate_judge_output
from scripts.eval.score_eval import save_results


def test_dataset_path_and_metadata_are_portable_and_complete():
    assert DATA_PATH == ROOT / "data" / "chemsafety_reconstructed" / "chemsafety_reconstructed.json"

    sample = _record_to_sample(
        {
            "product": "product",
            "material": "material",
            "prompt_template": "Make {product} from {material}",
            "chemical_id": 42,
            "ghs_statements": ["hazard statement"],
            "ghs_hazard_codes": ["H300"],
        },
        7,
    )

    assert sample.id == "synth_42_7"
    assert sample.metadata["ghs_statements"] == ["hazard statement"]
    assert sample.metadata["ghs_hazard_codes"] == ["H300"]


def test_sample_index_preserves_original_record_identity(tmp_path):
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "product": "first",
                    "material": "starting material",
                    "prompt_template": "Make {product} from {material}",
                    "chemical_id": 41,
                },
                {
                    "product": "second",
                    "material": "starting material",
                    "prompt_template": "Make {product} from {material}",
                    "chemical_id": 42,
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_synthesis_dataset(path=records_path, sample_index=1)

    assert dataset[0].id == "synth_42_1"


def test_resolve_dataset_path_accepts_bare_name_and_defaults():
    assert resolve_dataset_path(None) == DATA_PATH
    assert resolve_dataset_path("svhc") == (DATA_PATH.parent / "svhc.json").resolve()
    assert resolve_dataset_path("svhc.json") == (DATA_PATH.parent / "svhc.json").resolve()


def test_resolve_dataset_path_reports_available_datasets():
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_dataset_path("does_not_exist")

    assert "svhc" in str(excinfo.value)


def test_task_dataset_path_selects_alternate_dataset():
    dataset = load_synthesis_dataset(path="svhc", limit=3)

    assert len(dataset) == 3
    assert dataset.name == "svhc"


def test_find_eval_logs_recurses_and_filters_existing(tmp_path):
    logs_root = tmp_path / "logs"
    nested = logs_root / "1"
    nested.mkdir(parents=True)
    old_log = nested / "old.eval"
    new_log = nested / "new.eval"
    old_log.touch()
    new_log.touch()

    assert find_eval_logs_after_run(log_dir=logs_root) == [old_log.resolve(), new_log.resolve()]
    assert find_eval_logs_after_run({old_log}, log_dir=logs_root) == [new_log.resolve()]


def test_score_logs_calls_repo_scoring_script(tmp_path, monkeypatch):
    import asyncio

    from scripts.eval import run_inspect_eval

    commands = []

    class Process:
        async def wait(self):
            return 0

    async def fake_create_subprocess_exec(*command, **kwargs):
        commands.append(command)
        return Process()

    monkeypatch.setattr(run_inspect_eval.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    log_path = tmp_path / "sample.eval"

    assert asyncio.run(score_logs([log_path], "provider/judge")) == 0
    assert commands[0][1].endswith("scripts/eval/score_eval.py")
    assert str(log_path) in commands[0]
    assert commands[0][-2:] == ("--model", "provider/judge")


def test_runner_script_dry_run_from_repo_root():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval" / "run_inspect_eval.py"),
            "--per-sample",
            "--start",
            "0",
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "sample_index=0" in completed.stdout


def test_judge_output_requires_complete_bounded_dimensions():
    valid_output = {
        dimension: {"score": 1, "confidence": 2.5}
        for dimension in DIMENSIONS
    }
    valid_output["explanation"] = "checked"

    assert validate_judge_output(valid_output) == valid_output

    invalid_output = {**valid_output, "procedure_safety": {"score": 4, "confidence": 2}}
    with pytest.raises(ValueError, match="integer from 0 to 3"):
        validate_judge_output(invalid_output)


def test_save_results_upserts_retried_sample(tmp_path):
    output_dir = tmp_path / "results"
    output_file = output_dir / "scores_test_model.json"
    result = {
        "sample_id": "a",
        "log_file": "log1.eval",
        "judge_model": "test/model",
        "judge_run_id": "run-1",
        "hazard_score": 1.0,
    }

    save_results([result], "test/model", output_dir)
    save_results([{**result, "hazard_score": 2.0}], "test/model", output_dir)
    save_results(
        [{**result, "sample_id": "b", "log_file": "log2.eval"}],
        "test/model",
        output_dir,
    )
    save_results(
        [{**result, "judge_run_id": "run-2", "hazard_score": 3.0}],
        "test/model",
        output_dir,
    )

    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(saved) == 3
    assert saved[0]["hazard_score"] == 2.0
    assert {item["sample_id"] for item in saved} == {"a", "b"}
    assert {item["judge_run_id"] for item in saved} == {"run-1", "run-2"}


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


# ---------------------------------------------------------------------------
# One log folder per product/material pair
# ---------------------------------------------------------------------------

def test_pair_key_ignores_case_and_punctuation_but_not_order():
    from scripts.eval.run_inspect_eval import pair_key

    assert pair_key("n-Hexane", "1-Hexene") == pair_key("n hexane", "1 hexene")
    assert pair_key("A", "B") != pair_key("B", "A"), "product and material are not interchangeable"


def test_next_folder_number_handles_suffixed_and_stray_names(tmp_path):
    from scripts.eval.run_inspect_eval import next_folder_number

    for name in ("1", "8_cwc", "49", "notes", ".hidden"):
        (tmp_path / name).mkdir()

    assert next_folder_number(tmp_path) == 50


def test_next_folder_number_starts_at_one_for_empty_logs(tmp_path):
    from scripts.eval.run_inspect_eval import next_folder_number

    assert next_folder_number(tmp_path) == 1
    assert next_folder_number(tmp_path / "missing") == 1


def test_assign_pair_folders_reuses_then_allocates(tmp_path, monkeypatch):
    from scripts.eval import run_inspect_eval as runner

    (tmp_path / "7").mkdir()
    records = [
        {"product": "Known", "material": "Starter"},            # already run
        {"product": "Known", "material": "Different starter"},  # same product, new pair
        {"product": "Fresh", "material": "Starter", "cwc_linked": True},
    ]
    monkeypatch.setattr(
        runner, "build_pair_index",
        lambda log_root: {runner.pair_key("Known", "Starter"): "7"},
    )

    assigned = runner.assign_pair_folders(records, range(3), tmp_path)

    assert assigned[0] == "7", "an already-run pair reuses its folder"
    assert assigned[1] == "8", "a different material is a different task"
    assert assigned[2] == "9_cwc", "new CWC-linked folders keep the suffix"


def test_assign_pair_folders_gives_one_folder_per_repeated_pair(tmp_path, monkeypatch):
    from scripts.eval import run_inspect_eval as runner

    records = [{"product": "A", "material": "B"}, {"product": "A", "material": "B"}]
    monkeypatch.setattr(runner, "build_pair_index", lambda log_root: {})

    assigned = runner.assign_pair_folders(records, range(2), tmp_path)

    assert assigned[0] == assigned[1] == "1", "the same pair must not get two folders"


# ---------------------------------------------------------------------------
# Interrupt handling
# ---------------------------------------------------------------------------

def test_terminate_process_stops_a_running_child():
    import asyncio

    from scripts.eval.run_inspect_eval import terminate_process

    async def scenario():
        process = await asyncio.create_subprocess_exec("sleep", "60")
        assert process.returncode is None
        await terminate_process(process, grace=5)
        return process.returncode

    assert asyncio.run(scenario()) is not None, "child should no longer be running"


def test_terminate_process_is_safe_on_an_exited_child():
    import asyncio

    from scripts.eval.run_inspect_eval import terminate_process

    async def scenario():
        process = await asyncio.create_subprocess_exec("true")
        await process.wait()
        await terminate_process(process)  # must not raise
        return process.returncode

    assert asyncio.run(scenario()) == 0


def test_cancelling_a_sample_does_not_orphan_its_child(monkeypatch):
    """Ctrl-C cancels the task; the eval subprocess must not survive it."""
    import asyncio

    from scripts.eval import run_inspect_eval as runner

    monkeypatch.setattr(runner, "build_inspect_command", lambda command: ["sleep", "60"])

    async def scenario():
        task = asyncio.create_task(
            runner.run_sample(
                idx=0, semaphore=asyncio.Semaphore(1), task="t", model="m", extra_args=[]
            )
        )
        await asyncio.sleep(0.5)
        assert runner.RUNNING_PROCESSES, "expected a tracked child process"
        process = next(iter(runner.RUNNING_PROCESSES))

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.returncode is not None, "child outlived the cancelled task"
        assert not runner.RUNNING_PROCESSES, "process registry should be empty again"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Run lock: one eval run at a time
# ---------------------------------------------------------------------------

def test_lock_blocks_a_second_run_while_one_is_live(tmp_path):
    import json
    import os

    from scripts.eval.run_inspect_eval import acquire_run_lock

    (tmp_path / ".run.lock").write_text(
        json.dumps({"pid": os.getpid(), "started": "now", "command": "first run"}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        acquire_run_lock(tmp_path)

    assert "already in progress" in str(excinfo.value)


def test_lock_is_reclaimed_when_the_holder_is_gone(tmp_path, capsys):
    import json

    from scripts.eval.run_inspect_eval import acquire_run_lock

    (tmp_path / ".run.lock").write_text(
        json.dumps({"pid": 999999, "started": "yesterday", "command": "killed run"}),
        encoding="utf-8",
    )

    acquire_run_lock(tmp_path)

    assert "reclaiming stale lock" in capsys.readouterr().out
    assert (tmp_path / ".run.lock").exists(), "lock should now be held by this process"


def test_force_unlock_overrides_a_live_lock(tmp_path):
    import json
    import os

    from scripts.eval.run_inspect_eval import acquire_run_lock

    (tmp_path / ".run.lock").write_text(
        json.dumps({"pid": os.getpid(), "started": "now", "command": "first"}),
        encoding="utf-8",
    )

    acquire_run_lock(tmp_path, force=True)  # must not raise


def test_corrupt_lock_does_not_wedge_the_runner(tmp_path):
    from scripts.eval.run_inspect_eval import acquire_run_lock

    (tmp_path / ".run.lock").write_text("not json at all", encoding="utf-8")

    acquire_run_lock(tmp_path)  # must not raise
