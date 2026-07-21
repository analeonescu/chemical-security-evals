# Free-tier rate-limit budgeting

Limits change often — re-check provider dashboards before a big run. As of
mid-2026, rough free-tier shapes to plan around:

- **Google AI Studio (Gemini 2.5 Flash)**: request-per-minute limits are the
  binding constraint, not daily volume. Space judge calls out or use
  Inspect AI's built-in concurrency controls (`--max-connections`) rather
  than firing everything at once.
- **Groq**: generous daily request quota, high throughput, but per-minute
  caps on the largest models. Good for the "models under test" side of the
  experiment since you'll want many generations quickly.

## Budgeting the suggested first experiment

3 models x 2 harnesses x 200 prompts = 1,200 generations
Each generation scored by 2 judges = 2,400 judge calls
Total API calls ≈ 3,600

Split across:
- Day 1: run all generations (1,200 calls, spread across Groq + Gemini
  models under test)
- Day 2: run judge scoring (2,400 calls, spread across the two judge
  models)

Use `inspect eval --max-connections 5` (or lower) to stay under per-minute
caps rather than maxing out concurrency and hitting 429s.

## Cheap smoke tests first

Always run with `-T limit=10` before a full sweep — confirms the dataset
loader, solver, and scorer parse correctly without burning meaningful
quota. A single malformed judge prompt template can silently produce
`parse_error` scores across an entire run if you don't catch it early.
