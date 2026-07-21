"""
Entry point: `inspect eval task.py --model <provider/model>`

Examples (all free-tier):
  inspect eval task.py --model google/gemini-2.5-flash
  inspect eval task.py --model groq/llama-3.3-70b-versatile
  inspect eval task.py -T harness=cot --model groq/qwen-2.5-72b-instruct

Model-under-test is set via --model. Judges are fixed inside scorer.py
(swap the judge_a/judge_b defaults there, or pass through the task
parameters below if you want to sweep judges too).
"""

from inspect_ai import Task, task
from dataset import load_synthesis_dataset
from solvers import plain_solver, cot_solver, name_hack_solver
from scorer import dual_judge_scorer

HARNESSES = {
    "plain": plain_solver,
    "cot": cot_solver,
    "name_hack": name_hack_solver,
}


@task
def chemsafety_synthesis(harness: str = "plain", limit: int | None = 200) -> Task:
    if harness not in HARNESSES:
        raise ValueError(f"Unknown harness '{harness}', pick from {list(HARNESSES)}")

    return Task(
        dataset=load_synthesis_dataset(limit=limit),
        solver=HARNESSES[harness](),
        scorer=dual_judge_scorer(),
    )
