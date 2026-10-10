"""Streamlit smoke tests (scripted services): every tab and chart renders, plan approval works, demos are caught."""

import sys
from typing import Any

import pytest
import streamlit as st
from conftest import CATALOGUE
from streamlit.testing.v1 import AppTest

import tools


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(tools, "openrouter_models", lambda: CATALOGUE)
    monkeypatch.setenv("RF_MEMORY_DIR", str(tmp_path))
    for key in ("TAVILY_API_KEY", "PARALLEL_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.setenv(key, "")
    st.cache_resource.clear()
    st.cache_data.clear()


def start(monkeypatch: pytest.MonkeyPatch, key: str = "sk-test") -> AppTest:
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    return AppTest.from_file("../app.py", default_timeout=60).run()


def click(at: AppTest, key: str) -> AppTest:
    return next(b for b in at.button if b.key == key).click().run()


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
    click(at, "fabricated")
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


def test_false_premise_demo_is_corrected(offline: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    at = start(monkeypatch)
    click(at, "false_premise")
    assert not at.exception and any("Premise check: false" in e.value for e in at.error)
