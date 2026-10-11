"""Streaming report generation: the accumulator, the fallback to the normal call, the live 'Writing the answer…' block,
and that a streamed run ends with exactly the same structured result as a non-streamed one."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from itertools import pairwise
from typing import Any

import httpx
import pytest
import streamlit as st
from conftest import CATALOGUE
from langchain_core.messages import AIMessageChunk
from streamlit.testing.v1 import AppTest

import agents
import jobs
import memory
import tools
from agents import Keys, research
from evals.offline import FakeServices, fake_chunks
from models import AnswerDraft, Summary, TokenUsage, TraceEvent
from ui import activity

SUMMARY = Summary(bottom_line="Quantum may help LLMs, mostly later.", executive_summary="Costs $5 and *maybe* more [S1].")
RAW = SUMMARY.model_dump_json()
KEY = "sk-or-v1-0123456789abcdefSECRET"


# ---------------------------------------------------------------- accumulator
def test_partial_fields_and_draft_follow_a_growing_json_object() -> None:
    drafts = [agents.summary_draft(RAW[:n]) for n in range(len(RAW) + 1)]
    assert drafts[0] == "" and drafts[5] == "" and drafts[-1] == f"{SUMMARY.bottom_line}\n\n{SUMMARY.executive_summary}"
    grown = [d for d in drafts if d]
    assert all(b.startswith(a.split("\n\n")[0][:-1]) for a, b in pairwise(grown))  # it only ever grows
    assert tools.partial_fields('```json\n{"bottom_line": "Quan') == {"bottom_line": "Quan"}
    assert tools.partial_fields("not json") == {} and tools.partial_fields('["a"') == {}


def test_accumulate_streams_growing_text_then_validates_with_usage() -> None:
    seen: list[str] = []
    parsed, usage = asyncio.run(tools.accumulate(fake_chunks(RAW, size=7), Summary, seen.append))
    assert parsed == SUMMARY and usage["input_tokens"] == 1000 and usage["output_tokens"] == 100
    assert len(seen) == -(-len(RAW) // 7) and seen[-1] == RAW and all(b.startswith(a) for a, b in pairwise(seen))


def test_accumulate_ignores_a_failing_callback_but_rejects_bad_json() -> None:
    def boom(_: str) -> None:
        raise RuntimeError("ui bug")

    assert asyncio.run(tools.accumulate(fake_chunks(RAW), Summary, boom))[0] == SUMMARY  # the UI is a nicety
    with pytest.raises(ValueError):
        asyncio.run(tools.accumulate(fake_chunks('{"bottom_line": "cut off'), Summary, lambda _: None))
    with pytest.raises(ValueError, match="without any text"):
        asyncio.run(tools.accumulate(fake_chunks(""), Summary, lambda _: None))


def test_drafts_to_sends_only_when_the_readable_text_grew() -> None:
    assert agents.drafts_to(None) is None
    got: list[Any] = []
    on_text = agents.drafts_to(got.append)
    assert on_text is not None
    for n in range(len(RAW) + 1):
        on_text(RAW[:n])
    texts = [d.text for d in got]
    assert all(isinstance(d, AnswerDraft) for d in got) and len(texts) == len(set(texts)) and texts[-1].endswith("[S1].")


# ---------------------------------------------------------------- ask_stream: success, cache and fallback
class FakeLLM:
    """make_llm() stand-in: astream yields the given chunks (or raises), ainvoke records that the fallback was used."""

    def __init__(self, chunks: Any) -> None:
        self.chunks, self.streams = chunks, 0

    def astream(self, msgs: Any, **kw: Any) -> AsyncIterator[AIMessageChunk]:
        self.streams += 1
        assert kw == {"stream_usage": True} and "JSON schema" in msgs[-1].content
        if isinstance(self.chunks, Exception):
            raise self.chunks
        return self.chunks


@pytest.fixture
def fallback(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the non-streaming ask() and record each time ask_stream falls back to it."""
    calls: list[str] = []

    async def ask(key: str, model: str, schema: type, system: str, user: str, agent: str) -> Any:
        calls.append(agent)
        return SUMMARY, TokenUsage(agent=agent, model=model, input_tokens=7, output_tokens=3)

    monkeypatch.setattr(tools, "ask", ask)
    return calls


def stream_with(monkeypatch: pytest.MonkeyPatch, chunks: Any, on_text: Any = None) -> tuple[Any, FakeLLM]:
    llm = FakeLLM(chunks)
    monkeypatch.setattr(tools, "make_llm", lambda key, model: llm)
    out = asyncio.run(tools.ask_stream(KEY, "m/x", Summary, "sys", "user", "Report Builder", on_text if on_text else lambda _: None))
    return out, llm


def test_ask_stream_is_one_streamed_call_and_cached(monkeypatch: pytest.MonkeyPatch, fallback: list[str]) -> None:
    seen: list[str] = []
    (parsed, usage), llm = stream_with(monkeypatch, fake_chunks(RAW), seen.append)
    assert parsed == SUMMARY and llm.streams == 1 and not fallback and seen[-1] == RAW
    assert (usage.agent, usage.input_tokens, usage.output_tokens, usage.cached) == ("Report Builder", 1000, 100, False)
    (again, usage), llm = stream_with(monkeypatch, fake_chunks(RAW))
    assert again == SUMMARY and usage.cached and llm.streams == 0  # same cache as ask()


def test_ask_stream_without_a_callback_is_plain_ask(monkeypatch: pytest.MonkeyPatch, fallback: list[str]) -> None:
    monkeypatch.setattr(tools, "make_llm", lambda key, model: pytest.fail("no stream without a callback"))
    assert asyncio.run(tools.ask_stream(KEY, "m/x", Summary, "s", "u", "Report Builder", None))[0] == SUMMARY and fallback == ["Report Builder"]


async def dies_midway() -> AsyncIterator[AIMessageChunk]:
    yield AIMessageChunk(content=RAW[:20])
    raise ConnectionError(f"stream reset (Authorization: Bearer {KEY})")


async def stalls() -> AsyncIterator[AIMessageChunk]:
    yield AIMessageChunk(content=RAW[:20])
    await asyncio.sleep(5)


@pytest.mark.parametrize("chunks", [NotImplementedError("streaming unsupported"), dies_midway, stalls,
                                    lambda: fake_chunks('{"bottom_line": 1')], ids=["unsupported", "mid-stream error", "stall", "bad json"])
def test_streaming_problems_fall_back_to_the_normal_call(chunks: Any, monkeypatch: pytest.MonkeyPatch, fallback: list[str],
                                                         caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(tools, "STREAM_TIMEOUT_S", 0.2)
    with caplog.at_level(logging.WARNING, logger="researchforge"):
        (parsed, usage), llm = stream_with(monkeypatch, chunks if isinstance(chunks, Exception) else chunks())
    assert parsed == SUMMARY and usage.input_tokens == 7 and fallback == ["Report Builder"] and llm.streams == 1
    assert "streaming failed" in caplog.text and "falling back to a non-streaming call" in caplog.text
    assert "0123456789abcdef" not in caplog.text  # keys stay redacted in the warning


# ---------------------------------------------------------------- jobs and the graph
def test_job_emit_routes_drafts_to_partial_and_steps_to_events() -> None:
    job = jobs.Job("j")
    job.emit("queued notice")
    job.emit(AnswerDraft(text="Bottom"))
    job.emit(TraceEvent(agent="Planner", action="plan"))
    job.emit(AnswerDraft(text="Bottom line grows"))
    assert job.partial == "Bottom line grows" and [type(e).__name__ for e in job.events] == ["str", "TraceEvent"]


def shape(res: dict[str, Any]) -> dict[str, Any]:
    """What a run produced, minus timings and ids that differ between any two runs."""
    return {"keys": sorted(k for k in res if k not in ("wall_s", "thread")), "summary": res["summary"], "report_md": res["report_md"],
            "claims": [c.model_dump() for c in res["claims"]], "trust": res["trust"],
            "usage": [u.model_dump() for u in res["usage"]], "steps": [(e.agent, e.action, e.detail) for e in res["trace"]]}


def run_offline(monkeypatch: pytest.MonkeyPatch, root: Any, on_event: Any = None) -> tuple[dict[str, Any], FakeServices]:
    """One offline research run with fresh fakes (their scripted pages depend on the call count) and an empty cache."""
    fake = FakeServices()
    for name in ("ask", "ask_stream", "mcp_search", "arxiv_search"):
        monkeypatch.setattr(agents, name, getattr(fake, name))
    tools._CACHE.clear()
    memory._VECTORS.clear()
    res = asyncio.run(research(Keys("or-key", "tv-key", "pl-key"), CATALOGUE, memory.Memory(root), {"query": "How will quantum change LLMs?"},
                               "t", on_event=on_event))
    return res, fake


def test_streamed_run_gives_the_same_result_as_before(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    plain, fake = run_offline(monkeypatch, tmp_path / "a")
    assert "streamed" not in fake.calls  # CLI / evals path: no callback, no streaming
    got: list[Any] = []
    streamed, fake = run_offline(monkeypatch, tmp_path / "b", got.append)
    assert fake.calls["streamed"] == ["Report Builder"]
    assert shape(streamed) == shape(plain)
    drafts = [g.text for g in got if isinstance(g, AnswerDraft)]
    assert len(drafts) > 5 and drafts[-1] == f"{plain['summary'].bottom_line}\n\n{plain['summary'].executive_summary}"
    assert [g for g in got if isinstance(g, TraceEvent)] == streamed["trace"]  # activity events are unchanged


# ---------------------------------------------------------------- UI
def test_draft_md_is_escaped_with_a_bold_bottom_line_and_cursor() -> None:
    assert activity.draft_md("") == []
    assert activity.draft_md("Costs $5 *now*\n\nMore [S1]") == [r"**Bottom line:** Costs \$5 \*now\*", r"More \[S1\] ▌"]


def test_running_job_shows_the_answer_being_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(tools, "openrouter_models", lambda: CATALOGUE)
    monkeypatch.setenv("RF_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    st.cache_resource.clear()
    st.cache_data.clear()
    job = jobs.Job("fake-running-job", events=[TraceEvent(agent="Trust Judge", action="faithfulness + relevance", ms=900)])
    job.emit(AnswerDraft(text="Quantum may help LLMs, *mostly* later.\n\nEarly evidence only: costs $5"))
    jobs._JOBS[job.id] = job
    try:
        at = AppTest.from_file("../app.py", default_timeout=60)
        at.session_state["job"] = job.id
        at.session_state["asked"] = "How will quantum change LLMs?"
        at.run()
        assert not at.exception and at.status[0].label.startswith("Writing the answer...") and "1 steps so far" in at.status[0].label
        assert any(c.value.startswith("📝 Writing the answer…") for c in at.caption)
        text = [m.value for m in at.markdown]
        assert r"**Bottom line:** Quantum may help LLMs, \*mostly\* later." in text and r"Early evidence only: costs \$5 ▌" in text
        assert at.button(key="run").disabled  # still busy: the run buttons stay disabled
        job.emit(AnswerDraft(text="Quantum may help LLMs, mostly later.\n\nEarly evidence only: costs $5 and more."))
        at.run()
        assert r"Early evidence only: costs \$5 and more. ▌" in [m.value for m in at.markdown]  # it grows on the next poll
    finally:
        jobs._JOBS.pop(job.id, None)


def test_real_openrouter_client_streams_over_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ChatOpenAI + openai SDK path against a scripted OpenRouter SSE response (no network, no key)."""
    sent: list[dict[str, Any]] = []

    def sse(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        pieces = [RAW[i:i + 9] for i in range(0, len(RAW), 9)]
        frames = [{"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m/x",
                   "choices": [{"index": 0, "delta": {"role": "assistant", "content": p}, "finish_reason": None}]} for p in pieces]
        frames.append({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m/x", "choices": [],
                       "usage": {"prompt_tokens": 321, "completion_tokens": 54, "total_tokens": 375}})
        body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    real_init = tools._RunClients.__init__

    def scripted_http(self: Any) -> None:
        real_init(self)
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(sse))

    monkeypatch.setattr(tools._RunClients, "__init__", scripted_http)
    seen: list[str] = []

    async def go() -> Any:
        async with tools.llm_scope():
            return await tools.ask_stream(KEY, "m/x", Summary, "sys", "user", "Report Builder", seen.append)

    parsed, usage = asyncio.run(go())
    assert parsed == SUMMARY and (usage.input_tokens, usage.output_tokens) == (321, 54) and len(seen) > 5 and seen[-1] == RAW
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True} and len(sent) == 1
