"""Wrapper for running Inspect AI evals in a repeatable way.

Examples:

smoke test (verify Inspect AI works):
    python run_inspect_eval.py --smoke-test

eval:
    python run_inspect_eval.py --limit 10
    python run_inspect_eval.py --per-sample
    python run_inspect_eval.py --per-sample --start 0 --limit 10
    python run_inspect_eval.py --per-sample --limit 100 --max-concurrency 8

eval + judge:
    python run_inspect_eval.py --per-sample --judge --judge-model google/gemini-3.1-flash-lite
    
for only judging, use ./score_eval.py

The script reads defaults from inspect_eval_defaults.json, can launch one eval
per prompt sample, and optionally score generated logs with a judge model.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.env import load_env
from scripts.eval.dataset import (
    DATA_PATH,
    load_synthesis_records,
    resolve_dataset_path,
)


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_eval_dir(eval_root: Path, eval_id: int | str | None) -> Path | None:
    if eval_id is None:
        return None

    if isinstance(eval_id, str):
        candidate = Path(eval_id)
        if candidate.is_absolute() or candidate.exists():
            resolved = candidate if candidate.is_absolute() else (eval_root / candidate).resolve()
            if resolved.is_dir():
                return resolved
            if resolved.is_file():
                return resolved.parent
            raise FileNotFoundError(f"No eval directory found for '{eval_id}'")

        if candidate.suffix:
            candidate = candidate.parent

    numeric = str(eval_id)
    candidate = (eval_root / numeric).resolve()
    if candidate.is_dir():
        return candidate

    fallback = (Path.cwd() / numeric).resolve()
    if fallback.is_dir():
        return fallback

    raise FileNotFoundError(f"No numbered eval directory found for '{eval_id}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Inspect AI evals from a config file."
    )

    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick smoke test to verify Inspect AI works (runs 2 samples with default config)")
    parser.add_argument("--config", default="inspect_eval_defaults.json",
        help="Path to a JSON config file; in the future, this should be changed to a .config format.",
    )
    parser.add_argument("--task", help="Task file to evaluate")
    parser.add_argument("--eval-id", type=int, help="Run a numbered eval folder such as 1, 2, 3 under an evals/ directory")
    parser.add_argument("--eval-dir", default="evals", help="Root directory containing numbered eval folders")
    parser.add_argument("--prompt-file", help="Optional prompt override file to use for the selected eval folder")
    parser.add_argument("--dataset",
                        help="Dataset JSON to evaluate: a path, or a bare name such as 'svhc' "
                             "resolved against data/chemsafety_reconstructed (default: chemsafety_reconstructed)")
    parser.add_argument("--model",help="Model to use for evaluation")
    parser.add_argument("--limit",type=int,help="Maximum number of prompts to process",)
    parser.add_argument("--start",type=int,default=0,help="Starting prompt index")
    parser.add_argument("--per-sample",action="store_true",help="Run one eval per prompt sample",)
    parser.add_argument("--max-concurrency",type=int,default=10,
                        help="Maximum number of eval processes running simultaneously"
                        )
    parser.add_argument("--eval-repeat", type=int, default=1, help="Repeat the same eval run N times")
    parser.add_argument("--judge",action="store_true",help="Score generated eval logs after evaluation")
    parser.add_argument("--judge-model",help="Model to use as judge")
    parser.add_argument("--judge-repeat", type=int, default=1, help="Repeat the same judge scoring N times on the same log file")
    parser.add_argument("--display",
                        choices=["full", "conversation", "rich", "plain", "log", "none"],
                        help="Inspect display mode. Per-sample runs default to 'plain': the "
                             "'full' display is a full-screen app, and several at once fight "
                             "over the terminal and swallow Ctrl-C")
    parser.add_argument("--temperature", type=float,
                        help="Sampling temperature for the model under test; pin it so runs "
                             "are reproducible")
    parser.add_argument("--seed", type=int, help="Sampling seed, where the provider supports one")
    parser.add_argument("--force-unlock", action="store_true",
                        help="Start even if another run appears to hold the lock")
    parser.add_argument("--only-missing", action="store_true",
                        help="Run only samples with no generation log yet; use this to resume "
                             "an interrupted run without repeating work")
    parser.add_argument("--flat-logs", action="store_true",
                        help="Write logs wherever Inspect defaults to, instead of one "
                             "numbered folder per product/material pair")
    parser.add_argument("--dry-run",action="store_true",help="Print commands without running them")
    parser.add_argument("extra_args",nargs=argparse.REMAINDER,help="Additional Inspect AI CLI args")

    return parser


def shutil_which(name: str) -> Optional[str]:
    for path in os.environ.get("PATH","").split(os.pathsep):
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def build_inspect_command(command: list[str]) -> list[str]:
    if shutil_which("inspect"):
        return ["inspect"] + command

    return [sys.executable,"-m","inspect_ai"] + command


PAIR_INDEX_NAME = ".pair_index.json"


def pair_key(product, material) -> str:
    """Identity of a task: the product/material pair, normalised."""
    clean = lambda value: re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return f"{clean(product)}::{clean(material)}"


def _folder_fingerprint(log_root: Path) -> list:
    return sorted(
        (d.name, len(list(d.glob("*.eval"))))
        for d in log_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def build_pair_index(log_root: Path) -> dict:
    """Map each product/material pair to the folder already holding its runs.

    Reading every .eval is slow, so the result is cached beside the logs and
    rebuilt whenever the set of folders or their file counts changes.
    """
    from inspect_ai.log import read_eval_log

    if not log_root.exists():
        return {}

    fingerprint = _folder_fingerprint(log_root)
    cache_path = log_root / PAIR_INDEX_NAME
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                return cached.get("pairs", {})
        except (json.JSONDecodeError, OSError):
            pass

    pairs: dict[str, str] = {}
    for folder in sorted(log_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        for log_path in folder.glob("*.eval"):
            try:
                log = read_eval_log(log_path)
            except Exception:
                continue
            for sample in (log.samples or []):
                metadata = sample.metadata or {}
                key = pair_key(metadata.get("product"), metadata.get("material"))
                pairs.setdefault(key, folder.name)

    try:
        cache_path.write_text(
            json.dumps({"fingerprint": fingerprint, "pairs": pairs}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

    return pairs


def next_folder_number(log_root: Path) -> int:
    used = set()
    if log_root.exists():
        for folder in log_root.iterdir():
            if not folder.is_dir():
                continue
            base = folder.name.split("_", 1)[0]
            if base.isdigit():
                used.add(int(base))
    return max(used, default=0) + 1


def assign_pair_folders(records, indexes, log_root: Path, suffix_cwc=True) -> dict:
    """Pick a log folder per sample index: an existing one for a pair already
    run, otherwise the next free number."""
    pairs = build_pair_index(log_root)
    next_number = next_folder_number(log_root)
    assigned = {}

    for idx in indexes:
        record = records[idx]
        key = pair_key(record.get("product"), record.get("material"))
        folder = pairs.get(key)
        if folder is None:
            folder = str(next_number)
            # Keep the existing convention marking CWC-linked tasks.
            if suffix_cwc and record.get("cwc_linked"):
                folder += "_cwc"
            next_number += 1
            pairs[key] = folder
        assigned[idx] = folder

    return assigned


# Children spawned by this process, so Ctrl-C can stop them.
RUNNING_PROCESSES: set = set()

LOCK_NAME = ".run.lock"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_run_lock(log_root: Path, force: bool = False) -> Path:
    """Refuse to start while another run holds the lock.

    Two concurrent runs each see the same samples as missing and launch them
    all again, duplicating work and burning folder numbers. A stale lock from a
    killed run is reclaimed automatically.
    """
    log_root.mkdir(parents=True, exist_ok=True)
    lock_path = log_root / LOCK_NAME

    if lock_path.exists() and not force:
        try:
            holder = json.loads(lock_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            holder = {}
        pid = holder.get("pid")
        if isinstance(pid, int) and _process_alive(pid):
            raise SystemExit(
                f"Another run is already in progress (pid {pid}, started "
                f"{holder.get('started', 'unknown')}).\n"
                f"Wait for it to finish, or stop it and re-run. "
                f"Use --force-unlock to override, or delete {lock_path}."
            )
        print(f"reclaiming stale lock from pid {pid}")

    lock_path.write_text(
        json.dumps({
            "pid": os.getpid(),
            "started": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv[1:]),
        }),
        encoding="utf-8",
    )
    def release() -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(release)
    return lock_path


async def terminate_process(process, grace: float = 10.0) -> None:
    """Ask a child to stop, then kill it if it ignores the request."""
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


async def run_sample(idx: int, semaphore: asyncio.Semaphore, task: str,
                     model: str, extra_args: list[str], log_dir: str | None = None) -> int:
    async with semaphore:
        command = ["eval",task,"--model",model,"--limit","1","-T",f"sample_index={idx}",]
        command.extend(extra_args)
        full_command = build_inspect_command(command)
        print(f"\nRunning sample {idx}:" + (f"  -> logs/{log_dir}" if log_dir else ""))
        print(" ".join(full_command))

        # The wrapper cannot pass --log-dir through to Inspect (extra args are
        # consumed by argparse), so the destination is set via the environment.
        env = os.environ.copy()
        if log_dir:
            env["INSPECT_LOG_DIR"] = str(Path("logs") / log_dir)

        process = await asyncio.create_subprocess_exec(
            *full_command, cwd=str(Path.cwd()), env=env
        )
        RUNNING_PROCESSES.add(process)

        try:
            return_code = await process.wait()
        except asyncio.CancelledError:
            # Ctrl-C: stop the child rather than orphaning it.
            await terminate_process(process)
            raise
        finally:
            RUNNING_PROCESSES.discard(process)

        if return_code != 0:
            print(f"Sample {idx} failed with exit code {return_code}")

        return return_code


async def run_samples(indexes, task: str, model: str, extra_args: list[str],
                      max_concurrency: int, log_dirs: dict | None = None) -> int:
    semaphore = asyncio.Semaphore(max_concurrency)
    log_dirs = log_dirs or {}

    tasks = [asyncio.create_task(
        run_sample(
                idx=idx,
                semaphore=semaphore,
                task=task,
                model=model,
                extra_args=extra_args,
                log_dir=log_dirs.get(idx),
            )
        )
        for idx in indexes
    ]

    for task_result in asyncio.as_completed(tasks):
        return_code = await task_result

        if return_code != 0:
            for task in tasks:
                task.cancel()

            return return_code

    return 0


def find_eval_logs_after_run(
    previous_logs: set[Path] | None = None,
    log_dir: Path | None = None,
) -> list[Path]:
    log_dir = log_dir or Path.cwd() / "logs"

    if not log_dir.exists():
        return []

    logs = [path.resolve() for path in log_dir.rglob("*.eval")]
    if previous_logs is not None:
        previous = {path.resolve() for path in previous_logs}
        logs = [path for path in logs if path not in previous]

    return sorted(logs, key=lambda path: path.stat().st_mtime)


async def score_logs(logs: list[Path], judge_model: str) -> int:
    for log_path in logs:
        full_command = [
            sys.executable,
            str(Path(__file__).with_name("score_eval.py")),
            str(log_path),
            "--model",
            judge_model,
        ]

        print("\nScoring:")
        print(" ".join(full_command))

        process = await asyncio.create_subprocess_exec(
            *full_command,
            cwd=str(Path.cwd()),
        )

        return_code = await process.wait()

        if return_code != 0:
            return return_code

    return 0


async def main() -> int:
    load_env()  # API keys from .env, inherited by the inspect subprocess
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    config = load_config(config_path)

    # Handle smoke test mode
    if args.smoke_test:
        task = args.task or config.get("task", "scripts/eval/task.py")
        model = args.model or config.get("model", "google/gemini-3.5-flash-lite")
        
        command = ["eval", task, "--model", model, "--limit", "0"]
        full_command = build_inspect_command(command)
        
        print("Running smoke test (verifying Inspect AI interface):")
        print(" ".join(full_command))

        # Keep throwaway smoke logs out of the analysed log tree.
        env = os.environ.copy()
        env["INSPECT_LOG_DIR"] = str(Path("logs") / "_smoke")

        completed = subprocess.run(full_command, cwd=str(Path.cwd()), env=env)
        return int(completed.returncode)

    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.eval_repeat < 1:
        raise ValueError("--eval-repeat must be >= 1")
    if args.judge_repeat < 1:
        raise ValueError("--judge-repeat must be >= 1")
    if args.judge and not args.judge_model:
        raise ValueError("--judge requires --judge-model")

    eval_dir = None
    if args.eval_id is not None:
        eval_dir = resolve_eval_dir(Path(args.eval_dir), args.eval_id)
        task_dir = eval_dir / "task.py" if eval_dir else None
        if task_dir and task_dir.exists():
            task = str(task_dir)
        else:
            task = args.task or config.get("task", "task.py")
    else:
        task = args.task or config.get("task", "task.py")

    if eval_dir is not None:
        task_dir = eval_dir / "task.py"
        if task_dir.exists() and not args.task:
            task = str(task_dir)
        if args.prompt_file is None:
            prompt_file = eval_dir / "prompt.txt"
            if prompt_file.exists():
                prompt_file = str(prompt_file)
            else:
                prompt_file = str(eval_dir / "prompt.md") if (eval_dir / "prompt.md").exists() else None
        else:
            prompt_file = args.prompt_file
        if prompt_file:
            extra_args = list(args.extra_args or [])
            extra_args.extend(["-T", f"eval_dir={eval_dir}", "-T", f"prompt_file={prompt_file}"])
        else:
            extra_args = list(args.extra_args or [])
            if eval_dir:
                extra_args.extend(["-T", f"eval_dir={eval_dir}"])
    else:
        extra_args = list(args.extra_args or [])

    if not extra_args and "extra_args" in config:
        extra_args = list(config.get("extra_args", []))

    model = args.model or config.get("model", "google/gemini-3.5-flash-lite")
    limit = (args.limit if args.limit is not None else config.get("limit", 10))

    # "Run what's left" means scan the whole dataset, not the configured
    # default page size; an explicit --limit still caps how many run.
    if args.only_missing and args.limit is None:
        limit = None

    if args.eval_repeat > 1:
        extra_args = ["--epochs", str(args.eval_repeat), *extra_args]

    # A non-default dataset has to reach the task as well as the sample count below.
    dataset_arg = args.dataset or config.get("dataset")
    dataset_path = resolve_dataset_path(dataset_arg)
    if dataset_arg:
        extra_args = [*extra_args, "-T", f"dataset_path={dataset_path}"]

    # Generation config is recorded in the log, so pinning it here makes a run
    # reproducible rather than relying on provider defaults.
    if args.temperature is not None:
        extra_args = [*extra_args, "--temperature", str(args.temperature)]
    if args.seed is not None:
        extra_args = [*extra_args, "--seed", str(args.seed)]

    # Concurrent children must not each start a full-screen display.
    display = args.display or ("plain" if args.per_sample else None)
    if display:
        extra_args = [*extra_args, "--display", display]

    previous_logs = set(find_eval_logs_after_run())

    log_root = Path.cwd() / "logs"

    acquire_run_lock(log_root, force=args.force_unlock)

    if args.per_sample:
        records = load_synthesis_records(dataset_path)

        if args.only_missing:
            # Scan the whole dataset, then let --limit cap how many of the
            # remaining samples actually run.
            indexes = list(range(args.start, len(records)))
            generated = build_pair_index(log_root)
            before = len(indexes)
            indexes = [
                idx for idx in indexes
                if pair_key(records[idx].get("product"), records[idx].get("material"))
                not in generated
            ]
            print(f"skipping {before - len(indexes)} already generated; "
                  f"{len(indexes)} remaining")
            if not indexes:
                print("nothing left to run")
                return 0
            if args.limit is not None:
                indexes = indexes[: args.limit]
                print(f"running {len(indexes)} of them this batch")
        else:
            end = (len(records) if limit is None
                   else min(len(records), args.start + limit))
            indexes = list(range(args.start, end))

        # One folder per product/material pair: reuse the folder a pair was run
        # in before, otherwise allocate the next number.
        log_dirs = {}
        if not args.flat_logs:
            log_dirs = assign_pair_folders(records, indexes, log_root)
            reused = sum(
                1 for idx, folder in log_dirs.items()
                if folder in {d.name for d in log_root.iterdir() if d.is_dir()}
            ) if log_root.exists() else 0
            print(f"log folders: {reused} existing reused, "
                  f"{len(set(log_dirs.values())) - reused} new")

        if args.dry_run:
            for idx in indexes:
                command = ["eval", task, "--model", model, "--limit", "1", "-T", f"sample_index={idx}"]
                command.extend(extra_args)
                destination = f"  [logs/{log_dirs[idx]}]" if idx in log_dirs else ""
                print(f"Would run:{destination}")
                print("inspect " + " ".join(command))
            return 0

        result = await run_samples(
            indexes=indexes,
            task=task,
            model=model,
            extra_args=extra_args,
            max_concurrency=args.max_concurrency,
            log_dirs=log_dirs,
        )

        if result != 0:
            return result

    else:
        command = ["eval", task, "--model", model, "--limit", str(limit)]
        command.extend(extra_args)

        # A single run covers many pairs, so it cannot be split per pair; it
        # gets one new numbered folder of its own.
        run_folder = None if args.flat_logs else str(next_folder_number(log_root))

        if args.dry_run:
            print("Would run:" + (f"  [logs/{run_folder}]" if run_folder else ""))
            print("inspect " + " ".join(command))
            return 0

        full_command = build_inspect_command(command)

        print("Running:" + (f"  -> logs/{run_folder}" if run_folder else ""))
        print(" ".join(full_command))

        env = os.environ.copy()
        if run_folder:
            env["INSPECT_LOG_DIR"] = str(Path("logs") / run_folder)

        completed = subprocess.run(
            full_command,
            cwd=str(Path.cwd()),
            env=env,
        )

        if completed.returncode != 0:
            return int(completed.returncode)

    if args.judge:
        logs = find_eval_logs_after_run(previous_logs=previous_logs)

        if not logs:
            print("No eval logs found to score.")
            return 1

        judge_result = 0
        for _ in range(args.judge_repeat):
            judge_result = await score_logs(logs=logs, judge_model=args.judge_model)
            if judge_result != 0:
                return judge_result

        return judge_result

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        # 130 is the conventional exit code for SIGINT.
        print("\nInterrupted -- stopping eval processes.")
        exit_code = 130

    raise SystemExit(exit_code)
