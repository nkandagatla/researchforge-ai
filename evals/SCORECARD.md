# Eval scorecard (live)

Generated 2026-10-10T23:41:51 · commit `a55a61e` · 7 cases from `evals/golden.json` · command: `python -m evals.run`

| Metric | Value |
|---|---|
| Cases passed | 3 / 7 (errors: 0) |
| Faithfulness (mean over answered cases) | 93% |
| Citation accuracy (mean over answered cases) | 88% |
| Answer relevance (mean over answered cases) | 67% |
| False-premise detection | 50% |
| False-premise false alarms | 0% |
| Semantic-cache hit rate (cases expecting a hit) | 100% |
| Median / max full run | 105.71 s / 143.93 s |
| Max cache-hit latency | 0.74 s |
| Total cost | $0.1707 |

| Case | Pass | Findings | Faithfulness | Citation acc. | Relevance | Premise | Cache | Seconds | Cost | Failed checks |
|---|---|---|---|---|---|---|---|---|---|---|
| `quantum-llm` | ✅ | 30 | 97% | 91% | 82% | supported | miss | 77.12 | $0.0285 | — |
| `quantum-llm-again` | ✅ | 30 | 97% | 91% | 82% | supported | hit | 0.74 | $0.0000 | — |
| `gpu-false-premise` | ❌ | 30 | 97% | 93% | 15% | supported | miss | 76.18 | $0.0301 | premise |
| `transformer-false-premise` | ❌ | 29 | 90% | 92% | 15% | false | miss | 143.93 | $0.0239 | latency |
| `rag-hallucination` | ✅ | 29 | 90% | 80% | 92% | supported | miss | 74.93 | $0.0291 | — |
| `training-energy` | ❌ | 28 | 89% | 86% | 92% | supported | miss | 116.2 | $0.0297 | latency |
| `agent-security` | ❌ | 28 | 89% | 83% | 92% | supported | miss | 105.71 | $0.0293 | latency |
