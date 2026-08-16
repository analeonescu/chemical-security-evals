# Free-tier rate-limit budgeting

Limits change often — re-check provider dashboards before a big run. As of
mid-2026, rough free-tier shapes to plan around:

- **Google AI Studio (Gemini 3.5 Flash)**: request-per-minute limits are the
  binding constraint, not daily volume. Space judge calls out or use
  Inspect AI's built-in concurrency controls (`--max-connections`) rather
  than firing everything at once.
- **Groq**: generous daily request quota, high throughput, but per-minute
  caps on the largest models. Good for the "models under test" side of the
  experiment.
