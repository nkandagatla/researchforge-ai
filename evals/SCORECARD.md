# Eval scorecard (live): not generated yet

This file is written by the eval runner against the real services. It has **not** been generated in this repository yet:
the build environment had no OpenRouter key, and we do not publish numbers we did not measure.

To produce it (about 5 to 8 minutes and a few cents with the default `openai/gpt-4o-mini` routing):

```powershell
copy .env.example .env          # add OPENROUTER_API_KEY (TAVILY_API_KEY recommended)
python -m evals.run             # overwrites this file and evals/results.json
```

The runner sends the 7 questions in [`golden.json`](golden.json) through the full pipeline on a fresh, temporary memory, in order,
so the reworded repeat of the first question must be answered from the semantic cache. For each case it records:

| Metric | Where it comes from | Pass threshold |
|---|---|---|
| Faithfulness | Trust Judge: share of findings entailed by their own quotes | ≥ 70% |
| Citation accuracy | Verifier: share of quotes found verbatim in the source they cite | ≥ 70% |
| Answer relevance | Trust Judge: how well the findings answer the question (0-1) | ≥ 60% |
| False-premise detection | Report Builder's premise verdict is `unsupported` or `false` exactly when the case has a false premise | must match |
| Semantic cache | cache hit exactly when the case expects one | must match |
| Findings and keywords | at least `min_findings` verified findings, and every `must_mention` word in the answer | must match |
| Latency | full run ≤ 90 s, cache hit ≤ 5 s | must match |
| Cost | live OpenRouter prices × tokens used (0 on a cache hit) | reported |

The quality thresholds apply only to cases that are expected to produce findings (not to the false-premise questions).

`python -m evals.run --offline` runs the same harness with scripted LLM, search and embeddings. CI runs it on every push to
test the harness itself; its output goes to `runs/evals-offline/` and is clearly marked as an offline smoke test.
