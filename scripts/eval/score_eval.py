"""Utility for scoring existing Inspect AI eval logs.

Examples:
    python scripts/eval/score_eval.py logs/sample.eval \
        --model google/gemini-3.1-flash-lite

    python scripts/eval/score_eval.py logs/*.eval \
        --model google/gemini-3.1-flash-lite

Outputs:
    results/scores_google_gemini_3_1_flash_lite.json

The original .eval logs are updated with appended judge scores.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Must precede the scripts.* imports: running this file directly puts its own
# directory on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402
from inspect_ai.scorer import Score  # noqa: E402

from scripts.eval.env import load_env  # noqa: E402
from scripts.eval.scorer import (  # noqa: E402
    DIMENSIONS,
    EXPOSURE_WEIGHTS,
    _judge,
    validate_judge_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score existing Inspect AI eval logs with a judge model."
    )

    parser.add_argument("log_files",nargs="+",help="Path(s) to .eval files. Wildcards supported.")
    parser.add_argument("--model",required=True,help="Judge model to use.")
    parser.add_argument("--output-dir",default="results",help="Directory for JSON score outputs.")
    parser.add_argument("--timeout", type = float, default=300,help="judge timout")
    parser.add_argument("--skip-scored", action="store_true",
                        help="Skip samples this judge has already scored, so a re-run "
                             "extends coverage instead of adding repeat scores")
    parser.add_argument("--max-consecutive-errors", type=int, default=5,
                        help="Give up on this judge after N failures in a row and exit 2, "
                             "so a caller can move on to the next model (0 disables)")
    return parser

def resolve_log_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(Path(match).resolve() for match in matches)
        else:
            files.append(Path(pattern).resolve())
    return files


class JudgeUnavailable(RuntimeError):
    """Raised when a judge fails repeatedly, so the caller can move on."""


def already_scored_by(sample, judge_model: str) -> bool:
    """True if this sample already carries a score from this judge."""
    for score in (sample.scores or {}).values():
        metadata = getattr(score, "metadata", None) or {}
        if metadata.get("judge_model") == judge_model:
            return True
    return False


def scorer_name(judge_model: str, existing_scores: dict | None = None) -> str:
    base_name = ("judge_"+ judge_model.replace("/", "_").replace("-", "_").replace(".", "_"))

    if not existing_scores or base_name not in existing_scores:
        return base_name

    i = 1
    while f"{base_name}_{i}" in existing_scores:
        i += 1

    return f"{base_name}_{i}"

def process_judge_result(judge_result: dict[str, Any],precursor_accessibility: str) -> dict[str, Any]:
    judge_result = validate_judge_output(judge_result)
    dimensions = {key: judge_result[key] for key in DIMENSIONS}

    scores = [value["score"] for value in dimensions.values()]

    hazard_score = (sum(scores) / len(scores)
                    if scores
                    else None
                    )

    exposure_weight = EXPOSURE_WEIGHTS.get(
        precursor_accessibility,
        EXPOSURE_WEIGHTS["unknown"],
    )

    composite_risk = (hazard_score * exposure_weight
        if hazard_score is not None
        else None
    )

    return {
        "hazard_score": hazard_score,
        "composite_risk": composite_risk,
        "dimensions": dimensions,
        "exposure_weight": exposure_weight,
        "explanation": judge_result.get("explanation","")
        }


async def score_log(
    log_path: Path,
    judge_model: str,
    timeout: float,
    output_dir: Path,
    judge_run_id: str | None = None,
    skip_scored: bool = False,
    max_consecutive_errors: int = 0,
    error_state: dict | None = None,
) -> list[dict[str, Any]]:

    print(f"Scoring {log_path}", flush=True)

    log = read_eval_log(log_path)
    judge_run_id = judge_run_id or uuid.uuid4().hex
    eval_info = getattr(log, "eval", None)
    model_under_test = getattr(eval_info, "model", None)
    results = []
    total_samples = len(log.samples)
    error_state = error_state if error_state is not None else {"consecutive": 0}

    def note_failure(reason: str) -> None:
        error_state["consecutive"] += 1
        if max_consecutive_errors and error_state["consecutive"] >= max_consecutive_errors:
            raise JudgeUnavailable(
                f"{judge_model}: {error_state['consecutive']} consecutive failures "
                f"(last: {reason})"
            )

    for idx, sample in enumerate(log.samples, start=1):
        print(f"  sample {idx}/{total_samples}",flush=True)

        metadata = sample.metadata or {}
        result_context = {
            "sample_id": sample.id,
            "prompt_id": sample.id,
            "chemical_id": metadata.get("chemical_id"),
            "log_file": str(log_path.resolve()),
            "model_under_test": model_under_test,
            "judge_model": judge_model,
            "judge_run_id": judge_run_id,
        }
        
        if sample.scores is None:
            sample.scores = {}

        if skip_scored and already_scored_by(sample, judge_model):
            print("    already scored by this judge, skipping", flush=True)
            continue

        name = scorer_name(judge_model, sample.scores)

        try:
            raw_judge_result = await asyncio.wait_for(
                _judge(
                    model_name=judge_model,
                    question=sample.input,
                    response=sample.output.completion,
                    is_controlled=metadata.get("is_controlled",True),
                    ghs_statements=metadata.get("ghs_statements",[]),
                    precursor_accessibility=metadata.get(
                        "precursor_accessibility",
                        "unknown"
                        ),
                ),
                timeout=timeout,
            )

        except asyncio.TimeoutError:
            print(f"  TIMEOUT sample {idx}",flush=True)
            print('retrying with 2x timeout')
            print("if that still doesn't work, add a timeout flag in the parser")
            
            try:
                raw_judge_result = await asyncio.wait_for(
                    _judge(
                        model_name=judge_model,
                        question=sample.input,
                        response=sample.output.completion,
                        is_controlled=metadata.get("is_controlled", True),
                        ghs_statements=metadata.get("ghs_statements", []),
                        precursor_accessibility=metadata.get(
                            "precursor_accessibility",
                            "unknown",
                        ),
                    ),
                    timeout= (2 * timeout),
                )

            except asyncio.TimeoutError:
                results.append(
                    {
                        **result_context,
                        "error": "timeout",
                    }
                )
                note_failure("timeout")
                
                save_results(results=[results[-1]], judge_model=judge_model, output_dir=output_dir)

                continue

        except Exception as e:
            print(f"  FAILED sample {idx}: {e}",flush=True)

            results.append(
                {
                    **result_context,
                    "error": str(e),
                }
            )
            note_failure(str(e)[:120])
            save_results(results=[results[-1]], judge_model=judge_model, output_dir=output_dir)

            continue

        print("RAW JUDGE RESULT:")
        print(json.dumps(raw_judge_result, indent=2))
        
        judge_result = process_judge_result(
            raw_judge_result,
            metadata.get(
                "precursor_accessibility",
                "unknown",
            ),
        )
        if judge_result["hazard_score"] is None:
            print("WARNING: judge returned no scores")
            print(raw_judge_result)

            results.append(
                {
                    **result_context,
                    "error": "no hazard score returned",
                    "raw_judge_result": raw_judge_result,
                }
            )

            save_results(
                results=[results[-1]],
                judge_model=judge_model,
                output_dir=output_dir,
            )

            continue

        score = Score(
            value=judge_result["hazard_score"],
            explanation=judge_result["explanation"],
            metadata={
                "judge_model": judge_model,
                "dimensions": judge_result["dimensions"],
                "composite_risk": judge_result["composite_risk"],
                "exposure_weight": judge_result["exposure_weight"],
            },
        )


        if sample.scores is None:
            sample.scores = {}

        sample.scores[name] = score

        results.append(
            {
                **result_context,
                "hazard_score": judge_result["hazard_score"],
                "composite_risk": judge_result["composite_risk"],
                "dimensions": judge_result["dimensions"],
                "exposure_weight": judge_result["exposure_weight"],
                "explanation": judge_result["explanation"],
            }
        )
        save_results(results=[results[-1]], judge_model=judge_model, output_dir=output_dir)

        error_state["consecutive"] = 0
        print(f"  finished sample {idx}/{total_samples}",flush=True)


    write_eval_log(log,log_path)

    print(f"Finished {log_path}",flush=True)

    return results


async def run_all(
    log_files: list[Path],
    judge_model: str,
    timeout: float,
    output_dir: Path,
    skip_scored: bool = False,
    max_consecutive_errors: int = 0,
) -> list[dict[str, Any]]:
    judge_run_id = uuid.uuid4().hex
    all_results = []
    # Shared across logs so a judge that is down is detected quickly.
    error_state = {"consecutive": 0}

    for index, log_file in enumerate(log_files, start=1):
        print(f"[{index}/{len(log_files)}] {log_file.parent.name}/{log_file.name}", flush=True)
        results = await score_log(
            log_file,
            judge_model,
            timeout,
            output_dir,
            judge_run_id=judge_run_id,
            skip_scored=skip_scored,
            max_consecutive_errors=max_consecutive_errors,
            error_state=error_state,
        )
        all_results.extend(results)

    return all_results


def save_results(
    results: list[dict[str, Any]],
    judge_model: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (judge_model.replace("/", "_").replace("-", "_").replace(".", "_"))
    output_file = (output_dir/ f"scores_{filename}.json")

    existing: list[dict[str, Any]] = []
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                existing = [item for item in loaded if isinstance(item, dict)]
            elif isinstance(loaded, dict):
                existing = [loaded]
        except (json.JSONDecodeError, OSError):
            existing = []

    merged_results = list(existing)
    result_indexes = {
        (
            item.get("log_file"),
            item.get("sample_id"),
            item.get("judge_model"),
            item.get("judge_run_id"),
        ): index
        for index, item in enumerate(merged_results)
        if item.get("log_file") is not None and item.get("sample_id") is not None
    }
    for result in results:
        key = (
            result.get("log_file"),
            result.get("sample_id"),
            result.get("judge_model"),
            result.get("judge_run_id"),
        )
        if key in result_indexes:
            merged_results[result_indexes[key]] = result
        else:
            result_indexes[key] = len(merged_results)
            merged_results.append(result)

    with output_file.open("w",encoding="utf-8",) as handle:
        json.dump(merged_results,handle,indent=2)

    print(f"Saved scores to {output_file}",flush=True)


def main() -> int:
    load_env()  # API keys from .env, so they need not be exported
    parser = build_parser()
    args = parser.parse_args()
    logs = resolve_log_files(args.log_files)
    for log in logs:
        if not log.exists():
            raise FileNotFoundError(f"Missing log: {log}")
    try:
        asyncio.run(run_all(
            log_files=logs,
            judge_model=args.model,
            timeout=args.timeout,
            output_dir=Path(args.output_dir),
            skip_scored=args.skip_scored,
            max_consecutive_errors=args.max_consecutive_errors,
        ))
    except JudgeUnavailable as error:
        print(f"\nGiving up on this judge: {error}", flush=True)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())