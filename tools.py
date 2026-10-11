"""All external I/O: OpenRouter (models, prices, LLM calls), real MCP search, arXiv, PDFs."""

from __future__ import annotations

import asyncio
import os
import contextvars
import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any, NamedTuple, TypeVar

import arxiv
import httpx
import pymupdf
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.utils.json import parse_partial_json
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, SecretStr

from log import get_logger
from models import Provider, Source, TokenUsage, ToolCall

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
OPENROUTER = "https://openrouter.ai/api/v1"
MCP_URLS: dict[Provider, str] = {"parallel": "https://search.parallel.ai/mcp", "tavily": "https://mcp.tavily.com/mcp/"}
_CACHE: OrderedDict[str, BaseModel] = OrderedDict()  # LLM answer cache, LRU-bounded so a long-lived server stays flat
CACHE_SIZE = 256
STREAM_TIMEOUT_S = 60.0  # a stalled stream falls back to the normal call instead of hanging the run
logger = get_logger(__name__)


class MCPProvider(NamedTuple):
    """One MCP search server: its name, Streamable HTTP endpoint and auth headers (the key never goes in the URL)."""

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
    """Replace the given secret values and anything shaped like a known API key or bearer token with ``***``."""
    for secret in filter(None, secrets):
        text = text.replace(secret, "***")
    text = re.sub(r"(Bearer\s+|tavilyApiKey=)[^\s&\"']+", r"\1***", text)
    return re.sub(r"\b(sk-or-v1-|sk-lf-|pk-lf-|tvly-|lsv2_)[\w-]{8,}", r"\1***", text)  # well-known key shapes, even unlisted


def err(exc: BaseException) -> str:
    """A short (<= 200 chars), redacted 'Type: message' for traces and the UI; unwraps the first exception of a group."""
    inner = getattr(exc, "exceptions", None)  # unwrap anyio ExceptionGroups
    return err(inner[0]) if inner else redact(f"{type(exc).__name__}: {exc}", [])[:200]


def ms_since(start: float) -> int:
    """Milliseconds elapsed since a ``time.perf_counter()`` reading."""
    return int((time.perf_counter() - start) * 1000)


async def with_retry(fn: Callable[[], Awaitable[R]], retries: int = 1, timeout: float = 25, what: str = "call") -> R:
    """Await fn() with a per-attempt timeout, retrying with a short backoff; the last error is re-raised to the caller.

    ``what`` names the provider and operation in the retry log lines (e.g. "tavily web_search")."""
    for attempt in range(1, retries + 2):
        try:
            return await asyncio.wait_for(fn(), timeout)
        except Exception as exc:
            kind = f"timed out after {timeout:g}s" if isinstance(exc, TimeoutError) else f"failed: {err(exc)}"
            if attempt > retries:
                logger.debug("%s attempt %d/%d %s; giving up", what, attempt, retries + 1, kind)
                raise
            logger.warning("%s attempt %d/%d %s; retrying", what, attempt, retries + 1, kind)
            await asyncio.sleep(0.5 * attempt)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- OpenRouter
def openrouter_models() -> dict[str, tuple[float, float]]:
    """Live catalogue: model id -> ($ per input token, $ per output token). Empty if unreachable."""
    try:
        data = httpx.get(f"{OPENROUTER}/models", timeout=15).json()["data"]
        return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in data}
    except Exception as exc:  # offline or an unexpected payload shape: logged, and routing uses the default model
        logger.warning("openrouter model catalogue unavailable (%s): routing falls back to the default model", err(exc))
        return {}


class _RunClients:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=60)
        self.llms: dict[tuple[str, str, int], ChatOpenAI] = {}


_RUN: contextvars.ContextVar[_RunClients | None] = contextvars.ContextVar("rf_run_clients", default=None)


@asynccontextmanager
async def llm_scope() -> AsyncIterator[None]:
    """LLM clients for ONE research run, bound to that run's event loop and closed with it.

    Every run is its own asyncio.run() (Streamlit reruns, CLI). langchain-openai otherwise shares one process-wide async
    httpx pool, created on the first run's loop: the next run's calls then failed with 'RuntimeError: Event loop is closed'.
    """
    clients = _RunClients()
    token = _RUN.set(clients)
    try:
        yield
    finally:
        _RUN.reset(token)
        with suppress(Exception):
            await clients.http.aclose()


def make_llm(api_key: str, model: str, max_tokens: int = 2000) -> ChatOpenAI:
    """An OpenRouter chat client at temperature 0; inside llm_scope() it is reused for the run and shares its httpx pool."""
    scope = _RUN.get()
    if scope is not None and (api_key, model, max_tokens) in scope.llms:
        return scope.llms[(api_key, model, max_tokens)]
    llm = ChatOpenAI(
        base_url=OPENROUTER, api_key=SecretStr(api_key), model=model, temperature=0, max_completion_tokens=max_tokens,
        timeout=60, max_retries=1, http_async_client=scope.http if scope else httpx.AsyncClient(timeout=60),
    )
    if scope is not None:
        scope.llms[(api_key, model, max_tokens)] = llm
    return llm


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


def _cache_key(model: str, schema: type[BaseModel], system: str, user: str) -> str:
    return hashlib.sha256(f"{model}|{schema.__name__}|{system}|{user}".encode()).hexdigest()


def _remember(ckey: str, parsed: BaseModel) -> None:
    _CACHE[ckey] = parsed
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)


def json_hint(schema: type[BaseModel]) -> str:
    """The plain-JSON instruction used when a model can't (or, while streaming, shouldn't) use structured output."""
    return f"Reply with ONLY a JSON object valid for this JSON schema:\n{json.dumps(schema.model_json_schema())}"


async def ask(key: str, model: str, schema: type[T], system: str, user: str, agent: str) -> tuple[T, TokenUsage]:
    """The single structured-LLM entry point: cache, tool-call output, JSON fallback, token accounting."""
    ckey = _cache_key(model, schema, system, user)
    if (hit := _CACHE.get(ckey)) is not None:
        _CACHE.move_to_end(ckey)
        return schema.model_validate(hit), TokenUsage(agent=agent, model=model, cached=True)
    llm, msgs = make_llm(key, model), [SystemMessage(system), HumanMessage(user)]
    parsed: Any = None
    try:
        out: Any = await llm.with_structured_output(schema, include_raw=True).ainvoke(msgs)
        parsed, meta = out.get("parsed"), _usage(out.get("raw"))
    except Exception as exc:  # tool calling unsupported or malformed: the plain-JSON path below still gets an answer
        logger.info("openrouter %s (%s): structured output failed (%s); asking for plain JSON", model, agent, err(exc))
        meta = {}
    if parsed is None:  # some models ignore tool calling: ask for plain JSON instead
        hint = json_hint(schema)
        raw = await llm.ainvoke([*msgs, HumanMessage(hint)])
        meta = _usage(raw)
        try:
            parsed = parse_json(str(raw.content), schema)
        except ValueError as bad:  # one repair round before giving up
            logger.info("openrouter %s (%s): invalid JSON, one repair round", model, agent)
            fix = await llm.ainvoke([*msgs, HumanMessage(hint), raw,
                                     HumanMessage(f"That JSON was invalid ({str(bad)[:200]}). Reply with ONLY the corrected JSON object.")])
            parsed, meta = parse_json(str(fix.content), schema), _usage(fix)
    _remember(ckey, parsed)
    usage = TokenUsage(agent=agent, model=model, input_tokens=meta.get("input_tokens", 0), output_tokens=meta.get("output_tokens", 0))
    return parsed, usage


def partial_fields(raw: str) -> dict[str, Any]:
    """The fields of a JSON object that is still being streamed ('{"bottom_line": "Quantum may he' -> {'bottom_line': ...}).

    Code fences are ignored; anything not yet parseable (or not an object) gives {}."""
    text = re.sub(r"^```(?:json)?\s*", "", raw.lstrip())
    start = text.find("{")
    if start < 0:
        return {}
    try:
        out = parse_partial_json(text[start:])
    except Exception:  # half a token of JSON: wait for the next chunk
        return {}
    return out if isinstance(out, dict) else {}


async def accumulate(chunks: AsyncIterator[Any], schema: type[T], on_text: Callable[[str], None]) -> tuple[T, dict[str, int]]:
    """Stream accumulator: join the chunks' text, hand the growing text to on_text, then parse + validate the whole reply.

    Returns the validated object and the token usage (OpenRouter sends it in the last chunk). A callback that raises is
    logged and ignored (the UI is a nicety); a reply that is not valid JSON for ``schema`` raises ValueError."""
    message: AIMessageChunk | None = None
    async for chunk in chunks:
        message = chunk if message is None else message + chunk
        if chunk.text:
            try:
                on_text(message.text)
            except Exception as exc:
                logger.debug("stream callback failed (ignored): %s", err(exc))
    if message is None or not message.text.strip():
        raise ValueError("the stream ended without any text")
    return parse_json(message.text, schema), _usage(message)


async def ask_stream(key: str, model: str, schema: type[T], system: str, user: str, agent: str,
                     on_text: Callable[[str], None] | None) -> tuple[T, TokenUsage]:
    """ask(), but the reply is streamed and on_text sees the raw JSON grow token by token (for the live answer preview).

    Same cache, same validated schema and same token accounting as ask(), and still ONE LLM call: the model streams the
    JSON object instead of returning it in one piece. Any streaming problem (provider without streaming, an error or stall
    mid-stream, invalid JSON) logs a warning and falls back to the normal ask(); without on_text this simply is ask()."""
    if on_text is None:
        return await ask(key, model, schema, system, user, agent)
    ckey = _cache_key(model, schema, system, user)
    if (hit := _CACHE.get(ckey)) is not None:
        _CACHE.move_to_end(ckey)
        return schema.model_validate(hit), TokenUsage(agent=agent, model=model, cached=True)
    msgs = [SystemMessage(system), HumanMessage(user), HumanMessage(json_hint(schema))]
    stream: Any = None
    try:
        stream = make_llm(key, model).astream(msgs, stream_usage=True)
        parsed, meta = await asyncio.wait_for(accumulate(stream, schema, on_text), STREAM_TIMEOUT_S)
    except Exception as exc:  # never crash a run over streaming: the non-streaming path still answers
        logger.warning("openrouter %s (%s): streaming failed (%s); falling back to a non-streaming call", model, agent, err(exc))
        return await ask(key, model, schema, system, user, agent)
    finally:  # close the HTTP stream now (also after a timeout), not whenever the generator is garbage-collected
        with suppress(Exception):
            await stream.aclose()
    _remember(ckey, parsed)
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
    """The first advertised MCP tool whose name contains any of the words (e.g. "search", "fetch")."""
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
    """Map one provider result record onto a Source, taking the first non-empty text field (lists are joined)."""
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
        result = await with_retry(lambda: session.call_tool(tool.name, args), what=f"{prov.name} {tool.name}")
        if result.isError:
            raise RuntimeError(str(_payload(result))[:200])
        found = [record_to_source(r, prov.name) for r in payload_records(_payload(result))]
    except Exception as exc:  # timeouts, HTTP errors and tool errors become a failed ToolCall, never a crashed run
        logger.warning("%s %s failed after %d ms: %s", prov.name, tool.name, ms_since(start), err(exc))
        return [], ToolCall(provider=prov.name, tool=tool.name, ok=False, ms=ms_since(start), error=err(exc))
    if not found:
        logger.warning("%s %s returned no results", prov.name, tool.name)
    return found, ToolCall(provider=prov.name, tool=tool.name, ok=True, ms=ms_since(start), results=len(found))


@asynccontextmanager
async def mcp_session(prov: MCPProvider) -> AsyncIterator[tuple[ClientSession, list[Any]]]:
    """Short-lived Streamable HTTP session: initialize, then discover the advertised tools."""
    async with (
        streamablehttp_client(prov.url, headers=prov.headers, timeout=25) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await with_retry(session.initialize, what=f"{prov.name} initialize")
        yield session, (await session.list_tools()).tools


async def _fill_thin(session: ClientSession, prov: MCPProvider, tools: list[Any], query: str, found: list[Source],
                     limit: int) -> tuple[list[Source], list[ToolCall]]:
    """Results with only a snippet (< 300 chars) get their full page from the provider's fetch/extract tool, if it has one."""
    thin = [s.url for s in found if len(s.text) < 300][:2]
    fetch = pick_tool(tools, ("fetch", "extract"))
    fetch_args = build_args(fetch.inputSchema, query, thin, limit) if fetch and thin else None
    if not (fetch and fetch_args):
        return found, []
    fetched, call = await _call(session, prov, fetch, fetch_args)
    full = {s.url: s.text for s in fetched if s.text}
    return [s.model_copy(update={"text": full.get(s.url, s.text)}) for s in found], [call]


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
            found, fetch_calls = await _fill_thin(session, prov, tools, query, found, limit)
            return found[:limit], [call, *fetch_calls]
    except Exception as exc:  # provider down: degrade gracefully, keep the trace honest
        logger.warning("%s MCP session failed after %d ms: %s", prov.name, ms_since(start), err(exc))
        return [], [ToolCall(provider=prov.name, tool="session", ok=False, ms=ms_since(start), error=err(exc))]


async def discover(prov: MCPProvider) -> list[str]:
    """Names of the tools a provider advertises (used by the integration tests and for debugging)."""
    async with mcp_session(prov) as (_, tools):
        return [t.name for t in tools]


# ---------------------------------------------------------------- arXiv + PDF
async def arxiv_search(query: str, limit: int) -> tuple[list[Source], ToolCall]:
    """Up to ``limit`` arXiv papers (abstract as text) in a worker thread; failures and timeouts become a failed ToolCall."""
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
        # arXiv often stalls from cloud IPs: fail fast, the web search still covers the subquestion
        found = await with_retry(lambda: asyncio.to_thread(run), retries=0, timeout=float(os.getenv("RF_ARXIV_TIMEOUT", "15")),
                                 what="arxiv search")
    except Exception as exc:
        kind = "timed out" if isinstance(exc, TimeoutError) else "failed"
        logger.warning("arxiv search %s after %d ms: %s", kind, ms_since(start), err(exc))
        return [], ToolCall(provider="arxiv", tool="arxiv.Search", ok=False, ms=ms_since(start), error=err(exc))
    if not found:
        logger.warning("arxiv search returned no results")
    return found, ToolCall(provider="arxiv", tool="arxiv.Search", ok=True, ms=ms_since(start), results=len(found))


def pdf_sources(name: str, data: bytes) -> list[Source]:
    """Text-based PDF -> one source per page (page numbers survive chunking). ValueError if nothing is extractable."""
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = [(i + 1, page.get_text().strip()) for i, page in enumerate(doc)]
    except Exception as exc:
        logger.warning("pdf %r (%d bytes) could not be parsed: %s", name, len(data), err(exc))
        raise ValueError(f"Could not parse PDF '{name}': {exc}") from exc
    found = [Source(title=f"{name} p.{n}", url=f"pdf://{name}#page={n}", text=t, provider="pdf", page=n) for n, t in pages if t]
    if not found:
        logger.warning("pdf %r has no extractable text (%d pages)", name, len(pages))
        raise ValueError(f"'{name}' has no extractable text (scanned PDF?)")
    return found
