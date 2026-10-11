# Eval scorecard (live)

Generated 2026-10-11T00:38:22 · commit `5af266b` · 10 cases from `evals/golden.json` · command: `python -m evals.run`

| Metric | Value |
|---|---|
| Cases passed | 9 / 10 (errors: 0) |
| Faithfulness (mean over answered cases) | 85% |
| Citation accuracy (mean over answered cases) | 89% |
| Answer relevance (mean over answered cases) | 73% |
| False-premise detection | 50% |
| False-premise false alarms | 0% |
| Semantic-cache hit rate (cases expecting a hit) | 100% |
| Median / max full run | 61.11 s / 100.16 s |
| Max cache-hit latency | 0.59 s |
| Total cost | $0.2465 |

| Case | Pass | Findings | Faithfulness | Citation acc. | Relevance | Premise | Cache | Seconds | Cost | Failed checks |
|---|---|---|---|---|---|---|---|---|---|---|
| `quantum-llm` | ✅ | 28 | 93% | 84% | 78% | supported | miss | 53.06 | $0.0271 | — |
| `quantum-llm-again` | ✅ | 28 | 93% | 84% | 78% | supported | hit | 0.59 | $0.0000 | — |
| `gpu-false-premise` | ❌ | 29 | 72% | 95% | 15% | supported | miss | 55.34 | $0.0281 | premise |
| `transformer-false-premise` | ✅ | 28 | 86% | 82% | 15% | false | miss | 57.46 | $0.0278 | — |
| `rag-hallucination` | ✅ | 30 | 87% | 92% | 85% | supported | miss | 61.11 | $0.0278 | — |
| `training-energy` | ✅ | 30 | 87% | 93% | 85% | supported | miss | 61.9 | $0.0309 | — |
| `agent-security` | ✅ | 30 | 87% | 97% | 92% | supported | miss | 64.56 | $0.0255 | — |
| `short-ambiguous-rag` | ✅ | 29 | 79% | 86% | 92% | supported | miss | 64.01 | $0.0227 | — |
| `prompt-injection-in-question` | ✅ | 30 | 77% | 97% | 92% | supported | miss | 100.16 | $0.0270 | — |
| `numeric-recent-fact` | ✅ | 30 | 90% | 85% | 95% | supported | miss | 52.73 | $0.0297 | — |
