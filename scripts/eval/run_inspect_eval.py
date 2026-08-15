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
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.eval.dataset import DATA_PATH, load_synthesis_records


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


async def run_sample(idx: int, semaphore: asyncio.Semaphore, task: str,
                     model: str, extra_args: list[str]) -> int:
    async with semaphore:
        command = ["eval",task,"--model",model,"--limit","1","-T",f"sample_index={idx}",]
        command.extend(extra_args)
        full_command = build_inspect_command(command)
        print(f"\nRunning sample {idx}:")
        print(" ".join(full_command))

        process = await asyncio.create_subprocess_exec(*full_command,cwd=str(Path.cwd()))

        return_code = await process.wait()

        if return_code != 0:
            print(f"Sample {idx} failed with exit code {return_code}")

        return return_code


async def run_samples(start: int, end: int, task: str, model: str, extra_args: list[str], 
                      max_concurrency: int) -> int:
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [asyncio.create_task(
        run_sample(
                idx=idx,
                semaphore=semaphore,
                task=task,
                model=model,
                extra_args=extra_args,
            )
        )
        for idx in range(start, end)
    ]

    for task_result in asyncio.as_completed(tasks):
        return_code = await task_result

        if return_code != 0:
            for task in tasks:
                task.cancel()

            return return_code

    return 0


def find_eval_logs_after_run() -> list[Path]:
    log_dir = Path.cwd() / "logs"

    if not log_dir.exists():
        return []

    return sorted(
        log_dir.glob("*.eval"),
        key=lambda path: path.stat().st_mtime,  # can change depending on name convention
    )


async def score_logs(logs: list[Path], judge_model: str) -> int:
    for log_path in logs:
        command = ["score",str(log_path),"--model",judge_model]
        full_command = build_inspect_command(command)

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
        
        completed = subprocess.run(full_command, cwd=str(Path.cwd()))
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

    if args.eval_repeat > 1:
        extra_args = ["--epochs", str(args.eval_repeat), *extra_args]

    if args.per_sample:
        records = load_synthesis_records(DATA_PATH)

        end = (min(len(records), args.start + limit) if limit is not None else len(records))

        if args.dry_run:
            for idx in range(args.start, end):
                command = ["eval", task, "--model", model, "--limit", "1", "-T", f"sample_index={idx}"]
                command.extend(extra_args)
                print("Would run:")
                print("inspect " + " ".join(command))
            return 0

        result = await run_samples(
            start=args.start,
            end=end,
            task=task,
            model=model,
            extra_args=extra_args,
            max_concurrency=args.max_concurrency,
        )

        if result != 0:
            return result

    else:
        command = ["eval", task, "--model", model, "--limit", str(limit)]
        command.extend(extra_args)

        if args.dry_run:
            print("Would run:")
            print("inspect " + " ".join(command))
            return 0

        full_command = build_inspect_command(command)

        print("Running:")
        print(" ".join(full_command))

        completed = subprocess.run(
            full_command,
            cwd=str(Path.cwd()),
        )

        if completed.returncode != 0:
            return int(completed.returncode)

    if args.judge:
        logs = find_eval_logs_after_run()

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
    raise SystemExit(
        asyncio.run(main())
    )
