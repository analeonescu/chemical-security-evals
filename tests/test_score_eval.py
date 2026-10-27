import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.score_eval import JudgeUnavailable, already_scored_by, scorer_name


class _Score:
    def __init__(self, judge_model=None):
        self.metadata = {"judge_model": judge_model} if judge_model else {}


class _Sample:
    def __init__(self, scores=None):
        self.scores = scores


def test_already_scored_by_matches_on_judge_model():
    sample = _Sample({"judge_a": _Score("google/gemini-3.5-flash")})

    assert already_scored_by(sample, "google/gemini-3.5-flash")
    assert not already_scored_by(sample, "groq/llama-3.3-70b-versatile")


def test_already_scored_by_handles_unscored_samples():
    assert not already_scored_by(_Sample(None), "any/model")
    assert not already_scored_by(_Sample({}), "any/model")


def test_already_scored_by_ignores_scores_without_metadata():
    sample = _Sample({"legacy": _Score()})

    assert not already_scored_by(sample, "google/gemini-3.5-flash")


def test_scorer_name_suffixes_repeat_runs():
    assert scorer_name("google/gemini-3.5-flash") == "judge_google_gemini_3_5_flash"

    existing = {"judge_google_gemini_3_5_flash": object()}
    assert scorer_name("google/gemini-3.5-flash", existing).endswith("_1")

    existing["judge_google_gemini_3_5_flash_1"] = object()
    assert scorer_name("google/gemini-3.5-flash", existing).endswith("_2")


def test_judge_unavailable_is_an_exception():
    with pytest.raises(JudgeUnavailable):
        raise JudgeUnavailable("judge down")


def test_count_scores_by_counts_repeats():
    from scripts.eval.score_eval import count_scores_by

    sample = _Sample({
        "judge_a": _Score("google/gemini-3.5-flash"),
        "judge_a_1": _Score("google/gemini-3.5-flash"),
        "judge_b": _Score("groq/qwen/qwen3.8-27b"),
        "legacy": _Score(),
    })

    assert count_scores_by(sample, "google/gemini-3.5-flash") == 2
    assert count_scores_by(sample, "groq/qwen/qwen3.8-27b") == 1
    assert count_scores_by(sample, "never/used") == 0


def test_count_scores_by_handles_unscored_samples():
    from scripts.eval.score_eval import count_scores_by

    assert count_scores_by(_Sample(None), "any/model") == 0
    assert count_scores_by(_Sample({}), "any/model") == 0


def test_target_count_arithmetic_tops_up_rather_than_restarting():
    """Existing scores count toward the target, so a re-run only fills the gap."""
    from scripts.eval.score_eval import count_scores_by

    sample = _Sample({"judge_a": _Score("m"), "judge_a_1": _Score("m")})
    existing = count_scores_by(sample, "m")

    assert max(0, 4 - existing) == 2, "two more scores needed to reach four"
    assert max(0, 2 - existing) == 0, "already at target, nothing to do"
