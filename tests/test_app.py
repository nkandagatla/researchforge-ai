"""Streamlit smoke tests (scripted services): every tab and chart renders, plan approval works, demos are caught,
runs happen in a background job (no dropped clicks), and failures show a friendly message instead of crashing the app."""

import sys
from typing import Any

import pytest
import streamlit as st
from conftest import CATALOGUE
from streamlit.testing.v1 import AppTest

import agents
import jobs
import tools


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(tools, "openrouter_models", lambda: CATALOGUE)
    monkeypatch.setenv("RF_MEMORY_DIR", str(tmp_path))
    for key in ("TAVILY_API_KEY", "PARALLEL_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.setenv(key, "")
    st.cache_resource.clear()
    st.cache_data.clear()


def start(monkeypatch: pytest.MonkeyPatch, key: str = "sk-test", auto_approve: bool = False) -> AppTest:
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    if auto_approve:
        at.toggle(key="approve").set_value(True).run()
    return at


def settle(at: AppTest) -> AppTest:
    """Let the background research job finish, then rerun the page so it collects the result (the fragment poll does this live)."""
    assert jobs.wait_all(30)
    return at.run()


def click(at: AppTest, key: str, wait: bool = True) -> AppTest:
    at = next(b for b in at.button if b.key == key).click().run()
    return settle(at) if wait else at


def button(at: AppTest, key: str) -> Any:
    return next(b for b in at.button if b.key == key)


def test_renders_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    at = start(monkeypatch, key="")
    assert not at.exception and any("OPENROUTER_API_KEY" in w.value for w in at.sidebar.warning)


NEEDS_311 = pytest.mark.skipif(sys.version_info < (3, 11), reason="LangGraph interrupt() in async graphs needs Python 3.11+")


@NEEDS_311
def test_plan_approval_answer_tabs_demos_cache_and_library(offline: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    at = start(monkeypatch)
    click(at, "run")
    assert not at.exception and at.session_state.res["interrupt"] and offline["web"] == []  # paused before spending on search
    at.text_area(key="plan_subs").set_value("training\nagents")
    click(at, "plan_go")
    assert not at.exception and not at.error
    res = at.session_state.res
    assert res["plan"].subquestions == ["training", "agents"] and len(at.tabs) == 9  # Research/History + 7 detail tabs
    assert any("Bottom line" in s.value for s in at.success) and "sk-test" not in res["report_md"]
    click(at, "fabricated", wait=False)
    errors = [e.value for e in at.error]
    assert any("invalid reference S99" in e for e in errors) and any("quote not found" in e for e in errors)
    click(at, "run")  # same question again: semantic cache, no plan approval, instant
    assert at.session_state.res["cache_hit"] and any("second brain" in i.value for i in at.info)
    assert any(b.key == "refresh" for b in at.button)
    history = next(d.value for d in at.dataframe if "cache_hit" in d.value.columns)  # default view: Deep Research Sessions
    assert list(history["cache_hit"])[:2] == [True, False] and "citation_accuracy" in history.columns  # newest first
    assert any(m.label == "From memory" and m.value == "1" for m in at.metric)
    assert any(s.value == "Deep Research Sessions" for s in at.subheader) and not any("quote" in d.value.columns for d in at.dataframe)
    at.toggle(key="history_view").set_value(True).run()
    assert not at.exception and any(s.value == "Verified Claims Library" for s in at.subheader)
    library = next(d.value for d in at.dataframe if "quote" in d.value.columns)
    assert len(library) >= 6 and {"text", "quote", "support", "date"} <= set(library.columns)
    assert not any("cache_hit" in d.value.columns for d in at.dataframe)  # sessions view is hidden
    at.text_input(key="library_q").set_value("energy cost of training").run()
    assert not at.exception


def test_auto_approve_runs_end_to_end_in_one_click(offline: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    at = click(start(monkeypatch, auto_approve=True), "run")
    assert not at.exception and not at.session_state.res["interrupt"] and any("Bottom line" in s.value for s in at.success)
    assert offline["web"] and at.status and "Done in" in at.status[0].label
    assert [c.name for c in at.chat_message] == ["user", "assistant"] and at.chat_message[0].markdown[0].value.startswith("How is quantum")
    story = [m.value for m in at.status[0].markdown]
    assert any(line.startswith("🔎 **Contextual Retriever** searched Parallel MCP for *“") for line in story)
    assert any("📄 Parallel · [" in c.value for c in at.status[0].caption)  # result previews under each search
    cards = [m.value for m in at.markdown if "-badge[" in m.value]
    assert cards and all("Established" in c or "Contested" in c for c in cards)
    assert any(m.value.startswith("> “") and "](https://ex.com/" in m.value for m in at.markdown)  # quotes link to their source
    summary = next(t.value for t in at.table if "what it did" in t.value.columns)
    assert {"🔎 Contextual Retriever", "📝 Report Builder"} <= set(summary["agent"]) and "4,000" in set(summary["tokens in"])


def test_run_is_background_and_double_clicks_are_ignored(offline: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    gate = __import__("threading").Event()
    real = agents.research

    async def slow(*a: Any, **kw: Any) -> Any:
        await __import__("asyncio").to_thread(gate.wait, 20)
        return await real(*a, **kw)

    monkeypatch.setattr(agents, "research", slow)
    at = click(start(monkeypatch, auto_approve=True), "run", wait=False)
    assert not at.exception and button(at, "run").disabled and button(at, "false_premise").disabled  # page is back immediately
    assert any(s.label.startswith("Agents working") for s in at.status)
    first = at.session_state.job
    for key in ("run", "false_premise"):  # a stale click from a page rendered before the buttons were disabled
        stale = button(at, key)
        stale.proto.disabled = False
        at = stale.click().run()
    assert at.session_state.job == first and len([j for j in jobs._JOBS.values() if j.running]) == 1
    gate.set()
    at = settle(at)
    assert not at.exception and "job" not in at.session_state and not button(at, "run").disabled
    assert len([x for x in offline["llm"] if x.startswith("Planner")]) == 1  # exactly one run was paid for


def test_false_premise_demo_is_corrected(offline: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    at = start(monkeypatch)
    click(at, "false_premise")
    assert not at.exception and any("Premise check: false" in e.value for e in at.error)


def test_failed_run_shows_details_and_the_app_keeps_working(offline: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    real = agents.research

    async def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("upstream 500 for key sk-test and sk-or-v1-abcdefghijklmnop")

    monkeypatch.setattr(agents, "research", boom)
    at = click(start(monkeypatch, auto_approve=True), "run")
    assert not at.exception and any("hit a problem" in e.value for e in at.error)
    details = next(e for e in at.expander if e.label == "Error details")
    shown = " ".join(c.value for c in details.code)
    assert "RuntimeError: upstream 500" in shown and "sk-test" not in shown and "abcdefghijklmnop" not in shown
    assert at.status[0].label == "Research failed"
    monkeypatch.setattr(agents, "research", real)  # the next click (e.g. the demo) works normally
    click(at, "false_premise")
    assert not at.exception and any("Premise check: false" in e.value for e in at.error)
    assert not any("hit a problem" in e.value for e in at.error)
