"""Structured logging: secrets never reach a log line, and a failing retrieval provider is logged without failing the run."""

import asyncio
import io
import logging
import time
from typing import Any

import pytest
from conftest import CATALOGUE

import agents
import jobs
import log
import memory
import tools
from agents import Keys, research
from models import ToolCall

KEY = "sk-or-v1-0123456789abcdefSECRET"


def test_log_records_are_redacted_at_the_source(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "plain-env-secret-value")
    logger = log.get_logger("tools")
    assert logger.name == "researchforge.tools" and log.get_logger("researchforge.x").name == "researchforge.x"
    with caplog.at_level(logging.WARNING, logger="researchforge"):
        logger.warning("call failed: Authorization: Bearer %s, env %s", KEY, "plain-env-secret-value")
        try:
            raise RuntimeError(f"upstream echoed tavilyApiKey=tvly-abcdefghijkl and {KEY}")
        except RuntimeError:
            logger.warning("with traceback", exc_info=True)
    assert "0123456789abcdef" not in caplog.text and "plain-env-secret-value" not in caplog.text and "abcdefghijkl" not in caplog.text
    assert "Bearer ***" in caplog.text and "RuntimeError" in caplog.text  # the traceback survives, only the secrets go


def test_keys_registered_for_a_run_are_redacted(caplog: pytest.LogCaptureFixture) -> None:
    log.add_secrets("typed-into-the-sidebar-key", "short")  # too-short values are ignored (they'd blank out ordinary words)
    with caplog.at_level(logging.INFO, logger="researchforge"):
        log.get_logger("agents").info("using typed-into-the-sidebar-key; short words stay")
    assert "typed-into-the-sidebar-key" not in caplog.text and "short words stay" in caplog.text


def test_configure_is_idempotent_and_honours_rf_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger(log.ROOT)
    monkeypatch.setattr(root, "handlers", [h for h in root.handlers if not getattr(h, "_rf", False)])
    monkeypatch.setattr(root, "level", root.level)
    monkeypatch.setenv("RF_LOG_LEVEL", "debug")
    out = io.StringIO()
    log.configure(stream=out)
    log.configure(stream=io.StringIO())  # a second call must not add a second handler
    assert root.level == logging.DEBUG and sum(getattr(h, "_rf", False) for h in root.handlers) == 1
    log.get_logger("jobs").debug("job abc finished with %s", KEY)
    line = out.getvalue()
    assert "DEBUG" in line and "researchforge.jobs: job abc finished with sk-or-v1-***" in line


async def test_retry_and_timeout_are_logged_with_provider_and_attempt(caplog: pytest.LogCaptureFixture) -> None:
    async def slow() -> None:
        await asyncio.sleep(1)

    with caplog.at_level(logging.WARNING, logger="researchforge"), pytest.raises(TimeoutError):
        await tools.with_retry(slow, retries=1, timeout=0.01, what="tavily web_search")
    assert "tavily web_search attempt 1/2 timed out after 0.01s; retrying" in caplog.text


def test_failing_provider_is_logged_traced_and_does_not_fail_the_run(offline: dict[str, list[str]], tmp_path: Any,
                                                                     monkeypatch: pytest.MonkeyPatch,
                                                                     caplog: pytest.LogCaptureFixture) -> None:
    real = agents.mcp_search

    async def flaky(prov: Any, query: str, limit: int) -> Any:
        if prov.name == "tavily":  # an HTTP error escaping the client, with the key in its message
            raise ConnectionError(f"502 Bad Gateway for Authorization: Bearer {KEY}")
        if prov.name == "parallel" and query == "agents":  # a provider that answers but finds nothing
            return [], [ToolCall(provider="parallel", tool="web_search", ok=True, ms=3, results=0)]
        return await real(prov, query, limit)

    async def arxiv_down(query: str, limit: int) -> Any:
        return [], ToolCall(provider="arxiv", tool="arxiv.Search", ok=False, ms=15000, error="TimeoutError: ")

    monkeypatch.setattr(agents, "mcp_search", flaky)
    monkeypatch.setattr(agents, "arxiv_search", arxiv_down)
    with caplog.at_level(logging.INFO, logger="researchforge"):
        res = asyncio.run(research(Keys("or-key", "tv-key", "pl-key"), CATALOGUE, memory.Memory(tmp_path), {"query": "q", "max_loops": 0}, "pf"))
    assert res["report_md"] and any(c.status != "not_established" for c in res["claims"])  # parallel's results carried the run
    failed = [c for c in res["tool_calls"] if not c.ok]
    assert {c.provider for c in failed} == {"tavily", "arxiv"} and all("0123456789abcdef" not in c.error for c in failed)
    notes = [e.action for e in res["trace"] if e.agent == "Contextual Retriever"]
    assert "failed: tavily search" in notes and "failed: arxiv arxiv.Search" in notes and "no results" in notes
    assert "tavily search raised: ConnectionError" in caplog.text and "0123456789abcdef" not in caplog.text
    assert "run " in caplog.text and "finished: thread=pf" in caplog.text and "cost=$" in caplog.text


def test_job_slot_wait_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    async def make(emit: Any) -> dict[str, Any]:
        return {"ok": True}

    with caplog.at_level(logging.INFO, logger="researchforge"):
        assert jobs._SLOTS.acquire(blocking=False)  # another run holds the only slot
        job = jobs.start(make, lambda s: s)
        for _ in range(200):
            if job.events:
                break
            time.sleep(0.01)
        jobs._SLOTS.release()
        assert jobs.wait_all(10)
    assert job.result == {"ok": True} and job.events
    assert f"job {job.id[:8]} queued" in caplog.text and "got a slot after waiting" in caplog.text and "finished after" in caplog.text
