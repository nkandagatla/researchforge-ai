"""Offline fakes: LLM, MCP, arXiv and the ONNX models are all scripted, so unit tests never touch the network."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents
import jobs
import memory
import tools
from evals.offline import CATALOGUE, FakeCrossEncoder, FakeEmbedder, FakeServices, page_text, src

__all__ = ["CATALOGUE", "page_text", "src"]  # re-exported for the test modules


@pytest.fixture(autouse=True)
def offline_models(monkeypatch: pytest.MonkeyPatch) -> Any:
    tools._CACHE.clear()
    memory._VECTORS.clear()
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "RF_TRACING"):  # app.py loads the developer's .env on import
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(memory, "_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(memory, "_cross_encoder", lambda: FakeCrossEncoder())
    yield
    assert jobs.wait_all(30), "a background research job outlived its test"


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Patch every network edge of the graph and record the calls."""
    fake = FakeServices()
    monkeypatch.setattr(agents, "ask", fake.ask)
    monkeypatch.setattr(agents, "ask_stream", fake.ask_stream)
    monkeypatch.setattr(agents, "mcp_search", fake.mcp_search)
    monkeypatch.setattr(agents, "arxiv_search", fake.arxiv_search)
    return fake.calls
