"""Offline fakes: LLM, MCP, arXiv and the ONNX models are all scripted, so unit tests never touch the network."""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents
import jobs
import memory
import tools
from models import (ClaimDraft, ClaimSet, Entailment, Insights, JudgeReport, JudgeVerdict, Quote, ResearchPlan, Source,
                    Summary, TokenUsage, ToolCall, TopicProfile, TrustJudgement)

CATALOGUE = {m: (1.5e-7, 6e-7) for m in ["openai/gpt-4o-mini", "google/gemini-2.5-flash", "anthropic/claude-haiku-4.5",
                                         "openai/gpt-4.1-mini", "meta-llama/llama-3.3-70b-instruct"]}
TOPICS = ["energy cost of training large language models", "quantum optimisation for agent planning",
          "variational circuits designed by language model agents", "error correction limits for near term hardware",
          "quantum kernels for small classification datasets", "hybrid quantum classical inference pipelines",
          "post quantum security for agentic systems", "tensor network compression of transformer weights"]


def page_text(i: int, provider: str) -> str:
    topic = TOPICS[i % len(TOPICS)]
    return (f"Researchers report that {topic} could change how practitioners build systems in the next decade. "
            f"A {provider} survey found that {topic} remains experimental and mostly simulated today. "
            f"Critics argue that claims about {topic} are overstated because hardware is still noisy.")


def src(title: str, text: str = "", provider: str = "parallel", **kw: Any) -> Source:
    return Source(title=title, text=text or page_text(0, provider), provider=provider, url=kw.pop("url", f"https://ex.com/{title}"), **kw)


class FakeEmbedder:
    """Hashed bag of words: similar texts get similar vectors, identical texts identical ones."""

    def embed(self, texts: list[str], **kw: Any) -> Any:
        for t in texts:
            v = np.zeros(memory.DIM, dtype=np.float32)
            for w in re.findall(r"[a-z]{3,}", t.lower()):
                v[zlib.crc32(w.encode()) % memory.DIM] += 1.0
            yield v + 1e-3


class FakeCrossEncoder:
    def rerank(self, query: str, texts: list[str], **kw: Any) -> Any:
        q = set(re.findall(r"[a-z]{3,}", query.lower()))
        return [len(q & set(re.findall(r"[a-z]{3,}", t.lower()))) for t in texts]


def scripted(schema: type, user: str, model: str) -> Any:
    name = schema.__name__
    if name == "TopicProfile":
        return TopicProfile(domain="quantum AI", complexity=4, technical_depth=4, needs_recency=True, rationale="deep and new")
    if name == "ResearchPlan":
        premise = "Quantum computers have replaced GPUs for LLM training" if "replaced" in user else ""
        return ResearchPlan(subquestions=["training", "agents", "hardware", "extra"], academic_queries=["quantum llm", "quantum agents"],
                            premise=premise)
    if name == "ClaimSet":
        blocks = re.findall(r'<untrusted_source id="(S\d+)"[^>]*>\n(.*?)\n</untrusted_source>', user, re.S)
        drafts = [ClaimDraft(text=f"Finding from source{sid} on {body.split(' that ')[1][:40]}",
                             evidence=[Quote(source_id=sid, quote=body.split(". ")[0] + ".")]) for sid, body in blocks]
        drafts.append(ClaimDraft(text="Fabricated", evidence=[Quote(source_id="S99", quote="this quote does not exist anywhere at all")]))
        gaps = [] if user.startswith("Question: gap") else ["gap: cost data for quantum training"]
        return ClaimSet(claims=drafts, gaps=gaps)
    if name == "JudgeReport":
        idx = [int(line[1:].split(":")[0]) for line in user.splitlines() if line.startswith("#")]
        verdict = "contradicted" if "llama" in model else "supported"
        return JudgeReport(verdicts=[JudgeVerdict(index=i, verdict=verdict) for i in idx])
    if name == "TrustJudgement":
        idx = [int(line[1:].split(":")[0]) for line in user.splitlines() if line.startswith("#")]
        return TrustJudgement(entailment=[Entailment(index=i, entailed=i % 4 != 0) for i in idx], answer_relevance=0.8)
    if name == "Insights":
        return Insights(hypotheses=[{"statement": "Hybrid quantum-classical training emerges", "kind": "hypothesis", "based_on": [0]}])
    if name == "Summary":
        if "replaced GPUs" in user:
            return Summary(bottom_line="No: GPUs still train LLMs.", executive_summary="Quantum is experimental.", premise_verdict="false",
                           premise_note="Quantum computers have not replaced GPUs for LLM training.")
        return Summary(bottom_line="Quantum may help LLMs, mostly later.", executive_summary="Early evidence only.")
    if name == "Ranking":
        return schema(order=list(range(30))[::-1])
    raise AssertionError(name)


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
    calls: dict[str, list[str]] = {"llm": [], "web": [], "arxiv": []}

    async def fake_ask(key: str, model: str, schema: type, system: str, user: str, agent: str) -> Any:
        calls["llm"].append(f"{agent}:{model}")
        return scripted(schema, user, model), TokenUsage(agent=agent, model=model, input_tokens=1000, output_tokens=100)

    async def fake_mcp(prov: Any, query: str, limit: int) -> Any:
        calls["web"].append(f"{prov.name}:{query}")
        n = len(calls["web"])
        found = [src(f"{prov.name} {query} page {n}", page_text(n, prov.name), prov.name)]
        found.append(found[0].model_copy(update={"url": found[0].url + "?utm_source=x"}))  # tracking-param duplicate
        return found, [ToolCall(provider=prov.name, tool="web_search", ok=True, ms=5, results=2)]

    async def fake_arxiv(query: str, limit: int) -> Any:
        calls["arxiv"].append(query)
        n = len(calls["arxiv"])
        paper = src(f"paper on {query} {n}", page_text(n + 3, "arxiv"), "arxiv", arxiv_id=f"2401.0000{n}", published="2025-03-01")
        return [paper], ToolCall(provider="arxiv", tool="arxiv.Search", ok=True, ms=5, results=1)

    monkeypatch.setattr(agents, "ask", fake_ask)
    monkeypatch.setattr(agents, "mcp_search", fake_mcp)
    monkeypatch.setattr(agents, "arxiv_search", fake_arxiv)
    return calls
