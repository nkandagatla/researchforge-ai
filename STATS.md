# STATS (stack: claude)

## Lines per module

| Module | Lines | Code lines (no blanks/comments) |
|---|---|---|
| `app.py` | 526 | 455 |
| `agents.py` | 408 | 346 |
| `tools.py` | 263 | 205 |
| `models.py` | 219 | 165 |
| `evidence.py` | 365 | 292 |
| `memory.py` | 197 | 148 |
| `run_cli.py` | 68 | 58 |
| **Six app modules total** | **1978** | **1611** |

`run_cli.py` is the shared headless-runner contract and is not counted as an app module.

## Quality gates (last run)

| Check | Result |
|---|---|
| `ruff check .` | All checks passed! |
| `mypy` (7 files) | Success: no issues found in 7 source files |
| `pytest` (offline) | 42 passed, 3 deselected in 7.50s |
| Tests collected | 45 total: 42 offline unit/graph/app tests + 3 `@pytest.mark.integration` live tests |
| Coverage (offline) | 96% across the six modules and `run_cli.py` |
| Live integration tests | **Not run in the build environment** (its network blocks OpenRouter, Parallel, Tavily and arXiv). Run `pytest -m integration -s` |

Offline coverage by area: quote matching; dedup and stable ids; BM25, RRF, MMR and near-duplicate removal; hybrid select and rerank fallback; fencing and sanitizing; consensus; credibility and trust; report rendering; chunking; embedding cache; LanceDB **and** SQLite memory backends (semantic cache, claims, evidence, upsert); router rules; the full graph (loop, cache hit, refresh, related-question memory, interrupt and edited resume, session follow-up, false premise, loop limit and budget, empty retrieval, parallel timing, all charts); the CLI contract keys and second-run cache hit; tools (MCP discovery, search, fetch and degradation, JSON fallback, catalogue, PDF pages); and Streamlit (plan approval, all tabs, fabricated citation, cache banner, Library, false-premise demo).

Also checked by hand in the build environment, with the **real** ONNX models (`bge-small-en-v1.5`, `ms-marco-MiniLM-L-6-v2`) and scripted LLM and search:
- a full graph run took 6.8 s, of which the analyst's embed, rerank and MMR took about 5 s;
- a reworded question was a cache hit in 0.14 s;
- `streamlit run app.py` booted with no errors in the log.

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
