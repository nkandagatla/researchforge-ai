# STATS (stack: claude)

## Lines per module

| Module | Lines | Code lines (no blanks/comments; docstrings counted) |
|---|---|---|
| `app.py` (entry point) | 33 | 25 |
| `ui/` (9 modules: session, sidebar, research, answer, activity, tabs, charts, history, common) | 991 | 789 |
| `agents.py` | 559 | 475 |
| `tools.py` | 352 | 282 |
| `models.py` | 278 | 198 |
| `evidence.py` | 404 | 317 |
| `memory.py` | 258 | 199 |
| `jobs.py` | 112 | 90 |
| `log.py` | 71 | 53 |
| `evals/` (run, offline) | 463 | 358 |
| `run_cli.py` | 78 | 66 |

`run_cli.py` is the shared headless-runner contract. Largest UI module: `ui/charts.py` (289 lines).

## Quality gates (last run)

| Check | Result |
|---|---|
| `ruff check .` | All checks passed! |
| `mypy` (22 files: app, ui, evals, log and the core modules) | Success: no issues found in 22 source files |
| `pytest` (offline) | 71 passed, 3 deselected (Python 3.13 locally; CI runs 3.11 and 3.12) |
| Tests collected | 74 total: 71 offline unit/graph/app/eval/logging tests + 3 `@pytest.mark.integration` live tests |
| Coverage (offline) | 96% across app.py, ui/, evals/ and the core modules |
| Cyclomatic complexity (radon) | average A (3.7); no D/E/F functions; 3 functions at C (11-12): `evidence.verify`, `run_cli.contract_metrics`, `evals.run.aggregate` |
| `python -m evals.run --offline` | runs all 10 golden cases end to end with scripted services (harness check, not a product score) |
| Live eval scorecard | Committed in [`evals/SCORECARD.md`](evals/SCORECARD.md): a live run of the first 7 cases (6/7 passed); regenerate all 10 with `python -m evals.run` |
| Live integration tests | Need keys and network: `pytest -m integration -s` |

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) repeats these checks on Python 3.11 and 3.12 for every push and pull request.

Offline coverage by area: quote matching; dedup and stable ids; BM25, RRF, MMR and near-duplicate removal; hybrid select and rerank fallback; fencing and sanitizing; consensus; credibility and trust; report rendering; chunking; embedding cache; LanceDB **and** SQLite memory backends (semantic cache, claims, evidence, upsert); router rules; the full graph (loop, cache hit, refresh, related-question memory, interrupt and edited resume, session follow-up, false premise, loop limit and budget, empty retrieval, parallel timing, search queries and result previews in the trace, all charts); the CLI contract keys and second-run cache hit; tools (MCP discovery, search, fetch and degradation, JSON fallback, catalogue, PDF pages); the UI helpers (agent story lines, execution summary rows, finding badges and linked quotes, Markdown escaping, chart matrices, cost and session tiles); the eval harness (scoring rules incl. forbidden phrases, aggregation with no answered case, scorecard with error rows, a case that raises mid-run, a full offline golden-set run, the missing-key message); logging (key redaction in records, tracebacks and handler output, retry/timeout lines, a failing or empty retrieval provider that is logged and traced without failing the run, queued-run slot waits); and Streamlit (chat bubbles, the live story, finding cards, execution summary, plan approval, all tabs, fabricated citation, cache banner, Library, false-premise demo, background runs and double clicks, friendly errors).

## Dependencies

All resolve to binary wheels for **Windows / CPython 3.14** (checked with `pip install --dry-run --only-binary=:all: --platform win_amd64 --python-version 3.14`).

- `streamlit>=1.50`
- `langchain-core>=0.3`
- `langchain-openai>=0.3`
- `langgraph>=1.0`
- `langgraph-checkpoint-sqlite>=2.0`
- `aiosqlite>=0.20`
- `mcp>=1.9,<2`
- `arxiv>=2.1`
- `pymupdf>=1.24`
- `pydantic>=2.7`
- `httpx>=0.27`
- `python-dotenv>=1.0`
- `plotly>=5.20`
- `numpy>=2.1`
- `fastembed>=0.6`
- `lancedb>=0.20`
- `langsmith>=0.3`
- `langfuse>=3.0`
- `langchain>=1.0`
- `pytest>=8`
- `pytest-asyncio>=0.24`
- `pytest-cov>=5`
- `ruff>=0.6`
- `mypy>=1.11`

Cross-check on a second machine: your computer's Linux sandbox (Python 3.10), after `pip install -r requirements.txt` from PyPI, passes every test except the two plan-approval ones. Those are skipped below Python 3.11 because LangGraph's `interrupt()` in async graphs needs 3.11+.

Agent Activity Recharts charts (latency bars and the citation-accuracy trend line; Recharts 2.15.4 + React 18.3.1 UMD, pinned on jsDelivr): rendered in headless Chromium in light and dark mode from the same pinned builds served locally. The bars, bottleneck label, trend line, average reference line and hover tooltips all appeared with no script errors, and the offline fallback message shows when the CDN is unreachable.
