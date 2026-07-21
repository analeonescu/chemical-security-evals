# ChemSafetyBench Eval — Free-Tier Inspect AI Harness

A reproduction + extension of ChemSafetyBench's "Synthesis" task, built on
Inspect AI, using only free-tier model providers (Gemini, Groq). Adds a
dual-judge safety scorer to audit single-judge bias, which the original
paper's GPT-4o-only judge setup doesn't test for.

Source dataset/paper: https://github.com/HaochenZhao/SafeAgent4Chem
(arXiv:2411.16736)

## Why dual-judge

The original benchmark uses a single GPT-4o judge for both safety and
quality scoring. A single judge can't tell you whether its score reflects
the response or the judge's own idiosyncrasies (self-preference, verbosity
bias, position bias). Running two independent judges and measuring
agreement (Cohen's kappa / correlation) turns "we couldn't afford GPT-4o"
into a methodologically stronger setup than the original paper.

## Setup

1. Get free API keys (no credit card required for either):
   - Google AI Studio: https://aistudio.google.com -> Get API key
   - Groq: https://console.groq.com -> API keys

2. Set environment variables:
   ```bash
   export GOOGLE_API_KEY="..."
   export GROQ_API_KEY="..."
   ```

3. Install deps:
   ```bash
   pip install inspect-ai google-genai groq --break-system-packages
   ```

4. Clone the ChemSafetyBench data (read-only, for the Synthesis task JSON):
   ```bash
   git clone https://github.com/HaochenZhao/SafeAgent4Chem.git data/raw
   ```
   Point `dataset.py`'s `DATA_PATH` at the synthesis subset once you've
   confirmed the exact file layout in that repo (their public layout may
   differ from what's described in the paper — check `data/raw` first).

5. Run a small smoke test before spending your daily quota:
   ```bash
   inspect eval task.py -T limit=10 --model google/gemini-2.5-flash
   ```

6. Run the full sweep (see `RATE_LIMIT_NOTES.md` for batching guidance):
   ```bash
   inspect eval task.py --model groq/llama-3.3-70b-versatile
   inspect eval task.py --model google/gemini-2.5-flash
   ```

## Files

- `dataset.py` — loads ChemSafetyBench Synthesis prompts into Inspect `Sample`s
- `solvers.py` — harness variants (plain generate, CoT, name-hack)
- `scorer.py` — dual-judge safety + quality scorer (Gemini + Groq judges)
- `task.py` — wires dataset + solver + scorer into an Inspect `Task`
- `RATE_LIMIT_NOTES.md` — free-tier quota budgeting for a full run

## Suggested first experiment

3 models under test (Gemini Flash, Llama-3.3-70B via Groq, one more Groq
model e.g. Qwen) x 2 harness conditions (plain, CoT) x ~200 sampled
Synthesis prompts = 1,200 generations, each scored by 2 judges = 2,400
judge calls. Comfortably within free daily quotas if split over ~2 days.
