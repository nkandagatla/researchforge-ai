"""All external I/O: OpenRouter (models, prices, LLM calls), real MCP search, arXiv, PDFs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, NamedTuple, TypeVar

import arxiv
import httpx
import pymupdf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, SecretStr

from models import Provider, Source, TokenUsage, ToolCall

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
OPENROUTER = "https://openrouter.ai/api/v1"
MCP_URLS: dict[Provider, str] = {"parallel": "https://search.parallel.ai/mcp", "tavily": "https://mcp.tavily.com/mcp/"}
_CACHE: dict[str, BaseModel] = {}


class MCPProvider(NamedTuple):
    name: Provider
    url: str
    headers: dict[str, str]


def providers(tavily_key: str = "", parallel_key: str = "") -> list[MCPProvider]:
    """Keys travel only in Authorization headers, never in URLs, traces or the UI."""
    keys = {"parallel": parallel_key, "tavily": tavily_key}
    return [
        MCPProvider(name, url, {"Authorization": f"Bearer {keys[name]}"} if keys[name] else {})
        for name, url in MCP_URLS.items()
        if keys[name] or name == "parallel"  # Parallel allows keyless access; Tavily needs a key
    ]


def redact(text: str, secrets: list[str]) -> str:
    for secret in filter(None, secrets):
        text = text.replace(secret, "***")
    return re.sub(r"(Bearer\s+|tavilyApiKey=)[^\s&\"']+", r"\1***", text)


def err(exc: BaseException) -> str:
    inner = getattr(exc, "exceptions", None)  # unwrap anyio ExceptionGroups
    return err(inner[0]) if inner else redact(f"{type(exc).__name__}: {exc}", [])[:200]


def ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


async def with_retry(fn: Callable[[], Awaitable[R]], retries: int = 1, timeout: float = 25) -> R:
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(fn(), timeout)
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- OpenRouter
def openrouter_models() -> dict[str, tuple[float, float]]:
    """Live catalogue: model id -> ($ per input token, $ per output token). Empty if unreachable."""
    try:
        data = httpx.get(f"{OPENROUTER}/models", timeout=15).json()["data"]
        return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in data}
    except Exception:
        return {}


@lru_cache(maxsize=32)
def make_llm(api_key: str, model: str, max_tokens: int = 2000) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OPENROUTER, api_key=SecretStr(api_key), model=model, temperature=0, max_completion_tokens=max_tokens,
        timeout=60, max_retries=1,
    )


def _usage(msg: Any) -> dict[str, int]:
    return getattr(msg, "usage_metadata", None) or {}


def parse_json(text: str, schema: type[T]) -> T:
    """Parse model JSON leniently: strip code fences, then retry without trailing commas."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.S)
    body = match.group(0) if match else text
    try:
        return schema.model_validate_json(body)
    except ValueError:
        return schema.model_validate_json(re.sub(r",\s*([}\]])", r"\1", body))


async def ask(key: str, model: str, schema: type[T], system: str, user: str, agent: str) -> tuple[T, TokenUsage]:
    """The single structured-LLM entry point: cache, tool-call output, JSON fallback, token accounting."""
    ckey = hashlib.sha256(f"{model}|{schema.__name__}|{system}|{user}".encode()).hexdigest()
    if ckey in _CACHE:
        return schema.model_validate(_CACHE[ckey]), TokenUsage(agent=agent, model=model, cached=True)
    llm, msgs = make_llm(key, model), [SystemMessage(system), HumanMessage(user)]
    parsed: Any = None
    try:
        out: Any = await llm.with_structured_output(schema, include_raw=True).ainvoke(msgs)
        parsed, meta = out.get("parsed"), _usage(out.get("raw"))
    except Exception:
        meta = {}
    if parsed is None:  # some models ignore tool calling: ask for plain JSON instead
        hint = f"Reply with ONLY a JSON object valid for this JSON schema:\n{json.dumps(schema.model_json_schema())}"
        raw = await llm.ainvoke([*msgs, HumanMessage(hint)])
        meta = _usage(raw)
        try:
            parsed = parse_json(str(raw.content), schema)
        except ValueError as err:  # one repair round before giving up
            fix = await llm.ainvoke([*msgs, HumanMessage(hint), raw,
                                     HumanMessage(f"That JSON was invalid ({str(err)[:200]}). Reply with ONLY the corrected JSON object.")])
            parsed, meta = parse_json(str(fix.content), schema), _usage(fix)
    _CACHE[ckey] = parsed
    usage = TokenUsage(agent=agent, model=model, input_tokens=meta.get("input_tokens", 0), output_tokens=meta.get("output_tokens", 0))
    return parsed, usage


# ---------------------------------------------------------------- MCP
def build_args(schema: dict[str, Any], query: str, urls: list[str], limit: int) -> dict[str, Any] | None:
    """Fill an advertised tool's input schema from our query; None when it can't be satisfied."""
    values: dict[str, Any] = {
        "query": query, "objective": query, "q": query, "search_query": query, "search_queries": [query], "queries": [query],
        "max_results": limit, "num_results": limit, "limit": limit,
    }
    if urls:
        values |= {"urls": urls, "url": urls[0]}
    args = {name: values[name] for name in schema.get("properties", {}) if name in values}
    return args if all(r in args for r in schema.get("required", [])) else None


def pick_tool(tools: list[Any], words: tuple[str, ...]) -> Any | None:
    return next((t for t in tools if any(w in t.name.lower() for w in words)), None)


_TEXT_KEYS = ("content", "excerpts", "snippet", "text", "raw_content", "description", "summary")


def _text_records(text: str) -> list[dict[str, Any]]:
    found = []
    for block in re.split(r"\n(?=Title:)", text):
        if url := re.search(r"URL:\s*(\S+)", block):
            title = re.search(r"Title:\s*(.+)", block)
            body = re.search(r"(?:Raw Content|Content):\s*(.+)", block, re.S)
            found.append({"url": url.group(1), "title": (title.group(1) if title else url.group(1)).strip(),
                          "content": body.group(1).strip() if body else ""})
    return found


def payload_records(payload: Any) -> list[dict[str, Any]]:
    """Result-like dicts (anything with a url) from JSON, nested JSON, or 'Title:/URL:/Content:' text."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return _text_records(payload)
    if isinstance(payload, dict):
        own = [payload] if isinstance(payload.get("url"), str) else []
        return own + [r for v in payload.values() for r in payload_records(v)]
    if isinstance(payload, list):
        return [r for item in payload for r in payload_records(item)]
    return []


def record_to_source(rec: dict[str, Any], provider: Provider) -> Source:
    raw = next((rec[k] for k in _TEXT_KEYS if rec.get(k)), "")
    return Source(
        title=str(rec.get("title") or rec["url"]), url=rec["url"], provider=provider,
        published=rec.get("published_date") or rec.get("publish_date"),
        text=" ".join(map(str, raw)) if isinstance(raw, list) else str(raw),
    )


def _payload(result: Any) -> Any:
    return result.structuredContent or "\n".join(getattr(c, "text", "") for c in result.content)


async def _call(session: ClientSession, prov: MCPProvider, tool: Any, args: dict[str, Any]) -> tuple[list[Source], ToolCall]:
    start = time.perf_counter()
    try:
        result = await with_retry(lambda: session.call_tool(tool.name, args))
        if result.isError:
            raise RuntimeError(str(_payload(result))[:200])
        found = [record_to_source(r, prov.name) for r in payload_records(_payload(result))]
        return found, ToolCall(provider=prov.name, tool=tool.name, ok=True, ms=ms_since(start), results=len(found))
    except Exception as exc:
        return [], ToolCall(provider=prov.name, tool=tool.name, ok=False, ms=ms_since(start), error=err(exc))


@asynccontextmanager
async def mcp_session(prov: MCPProvider) -> AsyncIterator[tuple[ClientSession, list[Any]]]:
    """Short-lived Streamable HTTP session: initialize, then discover the advertised tools."""
    async with (
        streamablehttp_client(prov.url, headers=prov.headers, timeout=25) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await with_retry(session.initialize)
        yield session, (await session.list_tools()).tools


async def mcp_search(prov: MCPProvider, query: str, limit: int) -> tuple[list[Source], list[ToolCall]]:
    """Discover tools -> search -> fetch thin results. A failing provider degrades to an error record."""
    start = time.perf_counter()
    try:
        async with mcp_session(prov) as (session, tools):
            search = pick_tool(tools, ("search",))
            args = build_args(search.inputSchema, query, [], limit) if search else None
            if search is None or args is None:
                raise RuntimeError(f"no usable search tool among {[t.name for t in tools]}")
            found, call = await _call(session, prov, search, args)
            calls, thin = [call], [s.url for s in found if len(s.text) < 300][:2]
            fetch = pick_tool(tools, ("fetch", "extract"))
            fetch_args = build_args(fetch.inputSchema, query, thin, limit) if fetch and thin else None
            if fetch and fetch_args:
                fetched, fcall = await _call(session, prov, fetch, fetch_args)
                calls.append(fcall)
                full = {s.url: s.text for s in fetched if s.text}
                found = [s.model_copy(update={"text": full.get(s.url, s.text)}) for s in found]
            return found[:limit], calls
    except Exception as exc:  # provider down: degrade gracefully, keep the trace honest
        return [], [ToolCall(provider=prov.name, tool="session", ok=False, ms=ms_since(start), error=err(exc))]


async def discover(prov: MCPProvider) -> list[str]:
    async with mcp_session(prov) as (_, tools):
        return [t.name for t in tools]


# ---------------------------------------------------------------- arXiv + PDF
async def arxiv_search(query: str, limit: int) -> tuple[list[Source], ToolCall]:
    start = time.perf_counter()

    def run() -> list[Source]:
        results = arxiv.Client(num_retries=1, delay_seconds=1).results(arxiv.Search(query=query, max_results=limit))
        return [
            Source(title=r.title, url=r.entry_id, authors=[a.name for a in r.authors][:6], doi=r.doi,
                   arxiv_id=r.get_short_id().split("v")[0], published=r.published.date().isoformat(),
                   text=r.summary.replace("\n", " "), provider="arxiv")
            for r in results
        ]

    try:
        found = await with_retry(lambda: asyncio.to_thread(run), timeout=30)
        return found, ToolCall(provider="arxiv", tool="arxiv.Search", ok=True, ms=ms_since(start), results=len(found))
    except Exception as exc:
        return [], ToolCall(provider="arxiv", tool="arxiv.Search", ok=False, ms=ms_since(start), error=err(exc))


def pdf_sources(name: str, data: bytes) -> list[Source]:
    """Text-based PDF -> one source per page (page numbers survive chunking). ValueError if nothing is extractable."""
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = [(i + 1, page.get_text().strip()) for i, page in enumerate(doc)]
    except Exception as exc:
        raise ValueError(f"Could not parse PDF '{name}': {exc}") from exc
    found = [Source(title=f"{name} p.{n}", url=f"pdf://{name}#page={n}", text=t, provider="pdf", page=n) for n, t in pages if t]
    if not found:
        raise ValueError(f"'{name}' has no extractable text (scanned PDF?)")
    return found
