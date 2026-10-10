"""The LangGraph workflow end to end, fully offline: loops, memory, interrupts, consensus, trust, charts and the CLI contract."""

import asyncio
import sys
import json
import time
from typing import Any

import pytest
from conftest import CATALOGUE

import agents
import app
import memory
import run_cli
from agents import Keys, research
from models import ToolCall

Q = "How is quantum computing going to change LLMs and agentic implementations?"
KEYS = Keys("or-key", "tv-key", "pl-key")


def go(mem: memory.Memory, thread: str = "t", resume: Any = None, **inputs: Any) -> dict[str, Any]:
    return asyncio.run(research(KEYS, CATALOGUE, mem, ({"query": Q} | inputs) if resume is None else None, thread, resume))


@pytest.fixture
def mem(tmp_path) -> memory.Memory:  # type: ignore[no-untyped-def]
    return memory.Memory(tmp_path)


def test_full_run_loops_verifies_judges_and_remembers(offline: dict[str, list[str]], mem: memory.Memory) -> None:
    res = go(mem)
    answer = [c for c in res["claims"] if c.status != "not_established"]
    assert res["interrupt"] is None and not res["cache_hit"] and res["loops"] == 1
    assert len(answer) >= 8 and len({s for c in answer for s in c.citations}) >= 6
    assert [c.issues for c in res["claims"] if c.status == "not_established"] == [["invalid reference S99"]]
    assert {s.provider for s in res["ranked"]} == {"parallel", "tavily", "arxiv"} and res["duplicates_removed"] >= 6
    assert all(c.support == 1.0 for c in answer) and res["trust"].citation_accuracy < 1 and res["trust"].answer_relevance == 0.8
    assert "Report Builder:anthropic/claude-haiku-4.5" in offline["llm"] and "Critical Analyst:openai/gpt-4o-mini" in offline["llm"]
    agents_seen = {e.agent for e in res["trace"]}
    assert agents_seen >= {"Memory", "Topic Router", "Planner", "Contextual Retriever", "Critical Analyst", "Verifier", "Supervisor",
                           "Consensus Panel", "Insight Generator", "Trust Judge", "Report Builder"}
    assert len(mem.library()) == len(answer)
    edges = set(res["edges"])
    assert {("recall", "__end__"), ("supervisor", "gap_retriever"), ("gap_retriever", "analyst"), ("trust", "report")} <= edges


def test_semantic_cache_answers_instantly_and_refresh_bypasses_it(offline: dict[str, list[str]], mem: memory.Memory) -> None:
    first = go(mem, "a")
    n_llm = len(offline["llm"])
    start = time.perf_counter()
    cached = go(mem, "b", query=Q.lower().rstrip("?"))
    assert cached["cache_hit"] and time.perf_counter() - start < 2 and len(offline["llm"]) == n_llm and cached["usage"] == []
    assert cached["report_md"] == first["report_md"] and len(cached["claims"]) == len(first["claims"])
    fresh = go(mem, "c", force_refresh=True)
    assert not fresh["cache_hit"] and len(offline["llm"]) > n_llm
    related = go(mem, "d", query="Finding: energy cost of training large language models")
    assert not related["cache_hit"] and related["memory_claims"] and any(s.from_memory for s in related["ranked"])


NEEDS_311 = pytest.mark.skipif(sys.version_info < (3, 11), reason="LangGraph interrupt() in async graphs needs Python 3.11+")


@NEEDS_311
def test_plan_approval_interrupt_and_edited_resume(offline: dict[str, list[str]], mem: memory.Memory) -> None:
    paused = go(mem, "hitl", auto_approve=False)
    assert paused["interrupt"]["plan"]["subquestions"] == ["training", "agents", "hardware"] and offline["web"] == []
    done = go(mem, "hitl", resume={"subquestions": ["only this"], "academic_queries": ["q1"], "premise": ""})
    assert done["interrupt"] is None and done["plan"].subquestions == ["only this"]
    assert offline["web"][:2] == ["parallel:only this", "tavily:only this"]
    followup = go(mem, "hitl", query="Follow-up: what about quantum error correction?", auto_approve=True, max_loops=0)
    assert len(followup["sources"]) > len(done["sources"])  # the session's earlier evidence is reused


def test_false_premise_is_corrected(offline: dict[str, list[str]], mem: memory.Memory) -> None:
    res = go(mem, query="Why have quantum computers already replaced GPUs for training LLMs?", max_loops=0)
    assert res["plan"].premise and res["summary"].premise_verdict == "false" and "Premise check: false" in res["report_md"]


def test_loops_respect_limit_and_budget(offline: dict[str, list[str]], mem: memory.Memory, monkeypatch: pytest.MonkeyPatch) -> None:
    assert go(mem, "x", max_loops=0)["loops"] == 0
    monkeypatch.setenv("RF_BUDGET_USD", "0")
    res = go(mem, "y", max_loops=2, force_refresh=True)
    assert res["loops"] == 0 and any(e.detail == "budget reached" for e in res["trace"])


def test_llm_rerank_fallback_and_failures(offline: dict[str, list[str]], mem: memory.Memory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "rerank", lambda q, t: None)
    res = go(mem, "r", max_loops=0)
    assert any(x.startswith("Reranker (LLM fallback)") for x in offline["llm"]) and res["claims"]

    async def nothing(*a: Any) -> Any:
        return [], [ToolCall(provider="parallel", tool="session", ok=False, ms=1, error="down")]

    async def no_papers(*a: Any) -> Any:
        return [], ToolCall(provider="arxiv", tool="arxiv.Search", ok=False, ms=1, error="down")

    monkeypatch.setattr(agents, "mcp_search", nothing)
    monkeypatch.setattr(agents, "arxiv_search", no_papers)
    empty = go(memory.Memory(mem.root / "other"), "e", max_loops=0, query="An unrelated question about soil chemistry")
    assert empty["claims"] == [] and "No claim survived" in empty["report_md"]


def test_retrievers_run_in_parallel(offline: dict[str, list[str]], mem: memory.Memory, monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(prov: Any, query: str, limit: int) -> Any:
        await asyncio.sleep(0.3)
        return [], [ToolCall(provider=prov.name, tool="search", ok=True, ms=300)]

    monkeypatch.setattr(agents, "mcp_search", slow)
    start = time.perf_counter()
    go(mem, max_loops=0)
    assert time.perf_counter() - start < 1.5  # 6 x 0.3 s MCP calls overlap


def test_every_chart_builds(offline: dict[str, list[str]], mem: memory.Memory) -> None:
    res = go(mem)
    claims, ranked = res["claims"], res["ranked"]
    figs = [app.fig_workflow(res["edges"], res), app.fig_evidence_graph(claims, ranked), app.fig_source_mix(ranked), app.fig_confidence(claims),
            app.fig_gantt(res), app.fig_funnel(res), app.fig_dates(ranked), app.fig_heatmap(res["votes"])]
    assert all(f.data for f in figs)
    labels = {a.text for a in figs[0].layout.annotations if a.text}
    assert "5 branches" in labels and "1 loop(s)" in labels and any(t.startswith("faithfulness") for t in labels)
    cit = app.fig_citations(claims, ranked)
    assert len(cit.data[0].z) == len(claims) and any(y.endswith("✗") for y in cit.data[0].y) and {1} <= {v for row in cit.data[0].z for v in row}
    latency = memory_latency = app.ev.agent_latency(res["trace"])
    html = app.latency_chart_html(latency, {"Analyst": 1000.0}, 4)
    rows, bottleneck = app.latency_rows(memory_latency, {})
    assert bottleneck == max(rows, key=lambda r: r["run"])["agent"] and all(u in html for u in app.RECHARTS)
    assert '"runs": 4' in html and f'"bottleneck": "{bottleneck}"' in html and "__DATA__" not in html
    assert "</script><script>" not in app.latency_chart_html({"</script><script>": {"avg_ms": 1.0, "calls": 1.0, "total_ms": 1.0}}, {}, 1)
    assert len(mem.history()) >= 1 and mem.history()[0]["latency"]["Analyst"] > 0  # research() logs every finished run
    assert mem.history()[0]["citation_accuracy"] == round(res["trust"].citation_accuracy, 3)
    hist = [{"date": f"2026-10-0{d}T10:00:00", "question": f"q{d}", "cache_hit": d == 3, "findings": 9, "citation_accuracy": a}
            for d, a in [(5, 0.9), (4, 0.75), (3, 0.5), (2, None), (1, 0.6)]]  # newest first, as Memory.history() returns
    rows = app.accuracy_rows(hist)
    assert [r["accuracy"] for r in rows] == [60.0, 75.0, 90.0] and rows[0]["label"] == "Oct 01 10:00"  # oldest first, no cache hit/None
    page = app.accuracy_chart_html(hist)
    assert '"mean": 75.0' in page and "latest 90%" in page and "R.LineChart" in page and all(u in page for u in app.RECHARTS)
    assert "No research sessions yet" in app.accuracy_chart_html([]) and "R.BarChart" in app.latency_chart_html({}, {}, 0)


def test_cli_contract_keys_and_cache(offline: dict[str, list[str]], mem: memory.Memory, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    keys = ["stack", "question", "seconds", "cache_hit", "findings", "contested", "not_established", "sources_retrieved", "sources_cited",
            "duplicates_removed", "consensus_score", "faithfulness", "answer_relevance", "citation_accuracy", "input_tokens",
            "output_tokens", "cost_usd", "models_used", "agents", "loops"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("RF_MEMORY_DIR", str(mem.root))
    monkeypatch.setattr(run_cli, "openrouter_models", lambda: CATALOGUE)
    for n in (1, 2):
        monkeypatch.setattr("sys.argv", ["run_cli.py", "--question", Q, "--out", str(tmp_path / f"run{n}")])
        run_cli.main()
    first, second = (json.loads((tmp_path / f"run{n}" / "metrics.json").read_text(encoding="utf-8")) for n in (1, 2))
    assert list(first) == keys == list(second) and first["stack"] == "claude"
    assert not first["cache_hit"] and first["findings"] >= 6 and first["cost_usd"] > 0 and first["agents"] >= 11 and first["loops"] == 1
    assert second["cache_hit"] and second["findings"] == first["findings"] and second["cost_usd"] == 0 and second["input_tokens"] == 0
    assert (tmp_path / "run2" / "report.md").read_text(encoding="utf-8").startswith("# How is quantum")


def test_tracing_is_off_by_default_and_never_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    for key in ("RF_TRACING", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):  # a developer .env must not leak in
        monkeypatch.delenv(key, raising=False)
    callbacks, links = agents.tracing("run-1")
    assert callbacks == [] and links() == [] and agents.os.environ["LANGSMITH_TRACING"] == "false"
    monkeypatch.setenv("RF_TRACING", "true")
    monkeypatch.setattr("langsmith.Client", lambda: (_ for _ in ()).throw(OSError("offline")))
    callbacks, links = agents.tracing("run-2")
    assert agents.os.environ["LANGSMITH_TRACING"] == "true" and links() == []  # a failing backend yields no link, no crash
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_one_failing_step_never_sinks_the_run(offline: dict[str, list[str]], mem: memory.Memory, monkeypatch: pytest.MonkeyPatch) -> None:
    real = agents.ask

    async def flaky(key: str, model: str, schema: type, system: str, user: str, agent: str) -> Any:
        if (schema.__name__ == "ClaimSet" and user.startswith("Question: agents")) or agent in ("Insight Generator", "Trust Judge") \
                or "llama" in model or "gemini" in model:
            raise RuntimeError("provider exploded with sk-or-v1-abcdefghijklmnop")
        return await real(key, model, schema, system, user, agent)

    monkeypatch.setattr(agents, "ask", flaky)
    monkeypatch.setattr(agents, "search_web", lambda *a: (_ for _ in ()).throw(OSError("dns down")))  # every web branch dies
    res = go(mem, "ft", max_loops=0)
    failed = [e for e in res["trace"] if e.action.startswith("failed")]
    assert res["interrupt"] is None and res["report_md"] and any(c.status != "not_established" for c in res["claims"])
    assert {e.agent for e in failed} >= {"Critical Analyst", "Contextual Retriever", "Insight Generator", "Trust Judge", "Consensus Panel"}
    assert res["hypotheses"] == [] and not any("abcdefghijklmnop" in e.detail for e in res["trace"])  # keys redacted in activity


def test_nothing_usable_fails_with_a_clear_message(offline: dict[str, list[str]], mem: memory.Memory, monkeypatch: pytest.MonkeyPatch) -> None:
    real = agents.ask

    async def broken(key: str, model: str, schema: type, system: str, user: str, agent: str) -> Any:
        if schema.__name__ == "ClaimSet":
            raise RuntimeError("model down")
        return await real(key, model, schema, system, user, agent)

    monkeypatch.setattr(agents, "ask", broken)
    with pytest.raises(agents.ResearchFailed, match="Every analyst worker failed"):
        go(mem, "nf", max_loops=0)


def test_vector_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "VECTOR_CACHE_SIZE", 5)
    memory.embed([f"text number {i}" for i in range(12)])
    assert len(memory._VECTORS) == 5 and memory.embed(["text number 11"]).shape == (1, memory.DIM)
