"""Scripted stand-ins for every network edge (LLM, MCP search, arXiv) and the local ONNX models.

Shared by the offline test suite (tests/conftest.py) and the offline eval mode (python -m evals.run --offline), so the
pipeline and the eval harness run end to end with no keys and no network. The numbers they produce measure the harness,
not the product."""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np

import agents
import memory
import tools
from models import (ClaimDraft, ClaimSet, Entailment, Insights, JudgeReport, JudgeVerdict, Quote, ResearchPlan, Source, Summary,
                    TokenUsage, ToolCall, TopicProfile, TrustJudgement)

CATALOGUE = {m: (1.5e-7, 6e-7) for m in ["openai/gpt-4o-mini", "google/gemini-2.5-flash", "anthropic/claude-haiku-4.5",
                                         "openai/gpt-4.1-mini", "meta-llama/llama-3.3-70b-instruct"]}
TOPICS = ["energy cost of training large language models", "quantum optimisation for agent planning",
          "variational circuits designed by language model agents", "error correction limits for near term hardware",
          "quantum kernels for small classification datasets", "hybrid quantum classical inference pipelines",
          "post quantum security for agentic systems", "tensor network compression of transformer weights"]
FALSE_PREMISE_MARKERS = ("replaced", "abandoned")  # the scripted planner "notices" a premise in these questions


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


def _indexes(user: str) -> list[int]:
    return [int(line[1:].split(":")[0]) for line in user.splitlines() if line.startswith("#")]


def _plan(user: str) -> ResearchPlan:
    premise = f"Taken for granted: {user}" if any(m in user for m in FALSE_PREMISE_MARKERS) else ""
    return ResearchPlan(subquestions=["training", "agents", "hardware", "extra"], academic_queries=["quantum llm", "quantum agents"],
                        premise=premise)


def _claims(user: str) -> ClaimSet:
    blocks = re.findall(r'<untrusted_source id="(S\d+)"[^>]*>\n(.*?)\n</untrusted_source>', user, re.S)
    drafts = [ClaimDraft(text=f"Finding from source{sid} on {body.split(' that ')[1][:40]}",
                         evidence=[Quote(source_id=sid, quote=body.split(". ")[0] + ".")]) for sid, body in blocks]
    drafts.append(ClaimDraft(text="Fabricated", evidence=[Quote(source_id="S99", quote="this quote does not exist anywhere at all")]))
    return ClaimSet(claims=drafts, gaps=[] if user.startswith("Question: gap") else ["gap: cost data for quantum training"])


def _summary(user: str) -> Summary:
    if "Premise: none" not in user:  # the planner found a premise: the scripted report corrects it
        return Summary(bottom_line="No: GPUs still train LLMs.", executive_summary="Quantum is experimental.", premise_verdict="false",
                       premise_note="Quantum computers have not replaced GPUs for LLM training.")
    return Summary(bottom_line="Quantum may help LLMs, mostly later.", executive_summary="Early evidence only.")


def scripted(schema: type, user: str, model: str) -> Any:
    """The structured answer each agent's schema asks for, derived from the prompt so the graph's logic is exercised."""
    builders: dict[str, Any] = {
        "TopicProfile": lambda: TopicProfile(domain="quantum AI", complexity=4, technical_depth=4, needs_recency=True, rationale="deep and new"),
        "ResearchPlan": lambda: _plan(user),
        "ClaimSet": lambda: _claims(user),
        "JudgeReport": lambda: JudgeReport(verdicts=[JudgeVerdict(index=i, verdict="contradicted" if "llama" in model else "supported")
                                                     for i in _indexes(user)]),
        "TrustJudgement": lambda: TrustJudgement(entailment=[Entailment(index=i, entailed=i % 4 != 0) for i in _indexes(user)],
                                                 answer_relevance=0.8),
        "Insights": lambda: Insights(hypotheses=[{"statement": "Hybrid quantum-classical training emerges", "kind": "hypothesis", "based_on": [0]}]),
        "Summary": lambda: _summary(user),
        "Ranking": lambda: schema(order=list(range(30))[::-1]),
    }
    if schema.__name__ not in builders:
        raise AssertionError(f"no scripted answer for {schema.__name__}")
    return builders[schema.__name__]()


class FakeServices:
    """Drop-in replacements for agents.ask / mcp_search / arxiv_search that record every call."""

    def __init__(self) -> None:
        self.calls: dict[str, list[str]] = {"llm": [], "web": [], "arxiv": []}

    async def ask(self, key: str, model: str, schema: type, system: str, user: str, agent: str) -> Any:
        self.calls["llm"].append(f"{agent}:{model}")
        return scripted(schema, user, model), TokenUsage(agent=agent, model=model, input_tokens=1000, output_tokens=100)

    async def mcp_search(self, prov: Any, query: str, limit: int) -> Any:
        self.calls["web"].append(f"{prov.name}:{query}")
        n = len(self.calls["web"])
        found = [src(f"{prov.name} {query} page {n}", page_text(n, prov.name), prov.name)]
        found.append(found[0].model_copy(update={"url": found[0].url + "?utm_source=x"}))  # tracking-param duplicate
        return found, [ToolCall(provider=prov.name, tool="web_search", ok=True, ms=5, results=2)]

    async def arxiv_search(self, query: str, limit: int) -> Any:
        self.calls["arxiv"].append(query)
        n = len(self.calls["arxiv"])
        paper = src(f"paper on {query} {n}", page_text(n + 3, "arxiv"), "arxiv", arxiv_id=f"2401.0000{n}", published="2025-03-01")
        return [paper], ToolCall(provider="arxiv", tool="arxiv.Search", ok=True, ms=5, results=1)


@contextmanager
def offline_services() -> Iterator[FakeServices]:
    """Patch the network edges and the ONNX models for the duration of the block (used by the offline eval mode)."""
    fake = FakeServices()
    tools._CACHE.clear()
    memory._VECTORS.clear()
    with ExitStack() as stack:
        for target, name, value in [(agents, "ask", fake.ask), (agents, "mcp_search", fake.mcp_search), (agents, "arxiv_search", fake.arxiv_search),
                                    (memory, "_embedder", FakeEmbedder), (memory, "_cross_encoder", FakeCrossEncoder),
                                    (tools, "openrouter_models", lambda: CATALOGUE)]:
            stack.enter_context(patch.object(target, name, value))
        yield fake
    memory._VECTORS.clear()
