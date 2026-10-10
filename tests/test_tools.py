import asyncio
from types import SimpleNamespace
from typing import Any

import pymupdf
import pytest

import tools
from models import ResearchPlan

PARALLEL_SCHEMA = {"properties": {"objective": {}, "search_queries": {}}, "required": ["objective"]}
FETCH_SCHEMA = {"properties": {"urls": {}}, "required": ["urls"]}


def test_build_args_and_payload_parsing() -> None:
    assert tools.build_args(PARALLEL_SCHEMA, "q", [], 5) == {"objective": "q", "search_queries": ["q"]}
    assert tools.build_args({"properties": {"query": {}, "max_results": {}}}, "q", [], 5) == {"query": "q", "max_results": 5}
    assert tools.build_args(FETCH_SCHEMA, "q", [], 5) is None
    recs = tools.payload_records('{"results": [{"url": "https://a", "title": "A", "excerpts": ["one", "two"]}]}')
    assert tools.record_to_source(recs[0], "parallel").text == "one two"
    text = "Title: T1\nURL: https://t1\nContent: body one\n\nTitle: T2\nURL: https://t2\nContent: body two"
    assert [r["url"] for r in tools.payload_records(text)] == ["https://t1", "https://t2"] and tools.payload_records("none") == []


def test_providers_and_redaction() -> None:
    assert [p.name for p in tools.providers()] == ["parallel"] and tools.providers()[0].headers == {}
    assert tools.providers("tv")[1].headers == {"Authorization": "Bearer tv"}
    assert tools.redact("sk-1 Bearer abc ?tavilyApiKey=xyz", ["sk-1"]) == "*** Bearer *** ?tavilyApiKey=***"


async def test_with_retry() -> None:
    attempts: list[int] = []

    async def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("boom")
        return "ok"

    assert await tools.with_retry(flaky, retries=1, timeout=1) == "ok"
    with pytest.raises(asyncio.TimeoutError):
        await tools.with_retry(lambda: asyncio.sleep(1), retries=0, timeout=0.05)


class FakeLLM:
    def __init__(self, structured_fails: bool) -> None:
        self.structured_fails, self.calls = structured_fails, 0

    def with_structured_output(self, schema: type, include_raw: bool = False) -> Any:
        outer = self

        class R:
            async def ainvoke(self, msgs: Any) -> Any:
                outer.calls += 1
                if outer.structured_fails:
                    raise ValueError("tool calling unsupported")
                return {"parsed": schema(subquestions=["a"], academic_queries=["b"]),
                        "raw": SimpleNamespace(usage_metadata={"input_tokens": 7, "output_tokens": 3})}

        return R()

    async def ainvoke(self, msgs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(content='Sure! {"subquestions": ["x"], "academic_queries": ["y"]} done', usage_metadata={"input_tokens": 9})


@pytest.mark.parametrize("fails", [False, True])
async def test_ask_structured_json_fallback_and_cache(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    fake = FakeLLM(fails)
    monkeypatch.setattr(tools, "make_llm", lambda key, model: fake)
    plan, usage = await tools.ask("k", "m", ResearchPlan, "sys", "user", "Planner")
    assert plan.subquestions == (["x"] if fails else ["a"]) and usage.input_tokens == (9 if fails else 7) and usage.model == "m"
    again, cached = await tools.ask("k", "m", ResearchPlan, "sys", "user", "Planner")
    assert again == plan and cached.cached and fake.calls == (2 if fails else 1)


def test_openrouter_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"data": [{"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]}
    monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: SimpleNamespace(json=lambda: data))
    assert tools.openrouter_models() == {"openai/gpt-4o-mini": (1.5e-7, 6e-7)}
    monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert tools.openrouter_models() == {}


async def test_mcp_session_discovers_searches_fetches_and_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    search = SimpleNamespace(name="web_search_preview", inputSchema=PARALLEL_SCHEMA)
    fetch = SimpleNamespace(name="web_fetch", inputSchema=FETCH_SCHEMA)
    invoked: list[str] = []

    class Ctx:
        def __init__(self, *a: Any, **k: Any) -> None: ...
        async def __aenter__(self) -> Any:
            return self if isinstance(self, Session) else (None, None, None)
        async def __aexit__(self, *a: Any) -> None: ...

    class Session(Ctx):
        async def initialize(self) -> None: ...
        async def list_tools(self) -> Any:
            return SimpleNamespace(tools=[search, fetch])
        async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
            invoked.append(name)
            body = "short" if name == search.name else "full page text"
            return SimpleNamespace(isError=False, structuredContent={"results": [{"url": "https://a", "title": "A", "content": body}]})

    monkeypatch.setattr(tools, "streamablehttp_client", Ctx)
    monkeypatch.setattr(tools, "ClientSession", Session)
    found, calls = await tools.mcp_search(tools.providers()[0], "q", 3)
    assert invoked == ["web_search_preview", "web_fetch"] and found[0].text == "full page text" and all(c.ok for c in calls)
    assert await tools.discover(tools.providers()[0]) == ["web_search_preview", "web_fetch"]

    def broken(*a: Any, **k: Any) -> Any:
        raise ConnectionError("provider down")

    monkeypatch.setattr(tools, "streamablehttp_client", broken)
    found, calls = await tools.mcp_search(tools.providers()[0], "q", 3)
    assert found == [] and not calls[0].ok and "provider down" in calls[0].error


async def test_arxiv_failure_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise OSError("arxiv down")

    monkeypatch.setattr(tools.arxiv, "Client", boom)
    found, call = await tools.arxiv_search("q", 2)
    assert found == [] and not call.ok and "arxiv down" in call.error


def test_pdf_parsing_keeps_pages() -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Transformers use attention.")
    doc.new_page().insert_text((72, 72), "Page two text.")
    pages = tools.pdf_sources("paper.pdf", doc.tobytes())
    assert [p.page for p in pages] == [1, 2] and "attention" in pages[0].text
    with pytest.raises(ValueError, match="Could not parse"):
        tools.pdf_sources("bad.pdf", b"not a pdf")
    blank = pymupdf.open()
    blank.new_page()
    with pytest.raises(ValueError, match="no extractable text"):
        tools.pdf_sources("scan.pdf", blank.tobytes())


def test_llm_clients_are_per_run_and_survive_a_new_event_loop() -> None:
    async def one_run() -> tuple[object, object]:
        async with tools.llm_scope():
            a, b = tools.make_llm("k", "openai/gpt-4o-mini"), tools.make_llm("k", "openai/gpt-4o-mini")
            assert a is b  # reused within the run
            client = a.http_async_client
        assert client.is_closed  # closed with its loop, never reused by the next asyncio.run()
        return a, client

    first, second = asyncio.run(one_run()), asyncio.run(one_run())
    assert first[0] is not second[0] and first[1] is not second[1]
