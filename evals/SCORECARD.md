# Eval scorecard (live)

Generated 2026-10-10T23:57:08 · commit `d0729f0` · 7 cases from `evals/golden.json` · command: `python -m evals.run`

| Metric | Value |
|---|---|
| Cases passed | 6 / 7 (errors: 0) |
| Faithfulness (mean over answered cases) | 81% |
| Citation accuracy (mean over answered cases) | 81% |
| Answer relevance (mean over answered cases) | 61% |
| False-premise detection | 100% |
| False-premise false alarms | 0% |
| Semantic-cache hit rate (cases expecting a hit) | 100% |
| Median / max full run | 79.31 s / 150.0 s |
| Max cache-hit latency | 0.57 s |
| Total cost | $0.1737 |

| Case | Pass | Findings | Faithfulness | Citation acc. | Relevance | Premise | Cache | Seconds | Cost | Failed checks |
|---|---|---|---|---|---|---|---|---|---|---|
| `quantum-llm` | ✅ | 28 | 89% | 77% | 72% | supported | miss | 79.31 | $0.0300 | — |
| `quantum-llm-again` | ✅ | 28 | 89% | 77% | 72% | supported | hit | 0.57 | $0.0000 | — |
| `gpu-false-premise` | ✅ | 28 | 68% | 90% | 15% | false | miss | 77.84 | $0.0284 | — |
| `transformer-false-premise` | ✅ | 29 | 86% | 92% | 15% | unsupported | miss | 150.0 | $0.0261 | — |
| `rag-hallucination` | ❌ | 29 | 69% | 75% | 82% | supported | miss | 81.7 | $0.0302 | faithfulness |
| `training-energy` | ✅ | 28 | 75% | 78% | 78% | supported | miss | 77.76 | $0.0304 | — |
| `agent-security` | ✅ | 28 | 89% | 77% | 92% | supported | miss | 74.65 | $0.0286 | — |
