"""Headless runner (shared contract): python run_cli.py --question "<text>" --out <dir>"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import evidence as ev
import log
import memory
from agents import Keys, research
from models import Trust
from tools import openrouter_models


def trust_metrics(trust: Trust | None) -> dict[str, float]:
    """The trust panel rounded for the contract (zeros when the run produced no trust scores)."""
    t = trust or Trust()
    return {"consensus_score": round(t.consensus, 3), "faithfulness": round(t.faithfulness, 3), "answer_relevance": round(t.answer_relevance, 3),
            "citation_accuracy": round(t.citation_accuracy, 3)}


def contract_metrics(res: dict[str, Any], question: str, seconds: float, prices: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """The shared metrics.json contract for one run (a cache hit reports zero tokens and cost)."""
    claims, usage = res.get("claims", []), [] if res.get("cache_hit") else res.get("usage", [])
    answer = ev.answer_claims(claims)
    return {
        "stack": "claude", "question": question, "seconds": round(seconds, 2), "cache_hit": bool(res.get("cache_hit")),
        "findings": len(answer), "contested": sum(c.status == "contested" for c in claims),
        "not_established": sum(c.status == "not_established" for c in claims), "sources_retrieved": len(res.get("sources", [])),
        "sources_cited": len({s for c in answer for s in c.citations}), "duplicates_removed": int(res.get("duplicates_removed", 0)),
        **trust_metrics(res.get("trust")),
        "input_tokens": sum(u.input_tokens for u in usage), "output_tokens": sum(u.output_tokens for u in usage),
        "cost_usd": round(sum(ev.run_cost(usage, prices).values()), 6), "models_used": sorted({u.model for u in usage if u.model}),
        "agents": len({e.agent for e in res.get("trace", [])}), "loops": int(res.get("loops", 0)),
    }


def main() -> None:
    """Parse the CLI flags, run one research turn on a fresh thread, and write report.md + metrics.json to --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-loops", type=int, default=1)
    parser.add_argument("--refresh", action="store_true", help="ignore the semantic cache")
    parser.add_argument("--warmup", action="store_true", help="download/load the local ONNX models first (not timed)")
    parser.add_argument("--trace", action="store_true", help="trace to LangSmith and/or Langfuse if their keys are in .env")
    args = parser.parse_args()
    load_dotenv()
    log.configure()  # stderr only: stdout stays the metrics JSON line of the contract
    os.environ["RF_TRACING"] = str(args.trace).lower()
    if args.warmup:
        memory.warmup()
    keys = Keys(os.environ["OPENROUTER_API_KEY"], os.getenv("TAVILY_API_KEY", ""), os.getenv("PARALLEL_API_KEY", ""))
    prices, mem = openrouter_models(), memory.Memory()
    start = time.perf_counter()
    inputs = {"query": args.question, "auto_approve": True, "max_loops": args.max_loops, "force_refresh": args.refresh,
              "default_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")}
    res = asyncio.run(research(keys, prices, mem, inputs, f"cli-{uuid.uuid4()}"))
    metrics = contract_metrics(res, args.question, time.perf_counter() - start, prices)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(res.get("report_md", ""), encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))
    for url in res.get("trace_urls", []):
        print("trace:", url)


if __name__ == "__main__":
    main()
