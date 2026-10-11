"""Run the golden set through the full research pipeline and score it.

    python -m evals.run                 # live: needs OPENROUTER_API_KEY (Tavily/Parallel optional); writes evals/SCORECARD.md + results.json
    python -m evals.run --offline       # scripted LLM/search/models (no keys, no network): tests the harness itself, as CI does

Per case it scores faithfulness, citation accuracy and answer relevance (the app's trust metrics), false-premise detection,
semantic-cache behaviour, a keyword check on the answer, latency and cost, then aggregates them into a scorecard.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

import evidence as ev
import memory
import run_cli
import tools
from agents import Keys, research

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden.json"
QUALITY = ("faithfulness", "citation_accuracy", "answer_relevance")


class Case(BaseModel):
    id: str
    question: str
    false_premise: bool = False
    expect_cache_hit: bool = False
    min_findings: int = 0
    must_mention: list[str] = Field(default_factory=list)
    note: str = ""


class Thresholds(BaseModel):
    faithfulness: float = 0.7
    citation_accuracy: float = 0.7
    answer_relevance: float = 0.6
    full_run_seconds: float = 90
    cache_hit_seconds: float = 5


class Golden(BaseModel):
    description: str = ""
    thresholds: Thresholds = Thresholds()
    cases: list[Case]


def load(path: Path = GOLDEN) -> Golden:
    return Golden.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- scoring (pure, unit-tested)
def answer_text(res: dict[str, Any]) -> str:
    """Everything the user reads as the answer: bottom line, summary and the verified findings."""
    summary = res.get("summary")
    parts = [summary.bottom_line, summary.executive_summary] if summary else []
    return " ".join(parts + [c.text for c in ev.answer_claims(res.get("claims", []))])


def checks(case: Case, metrics: dict[str, Any], premise_flagged: bool, text: str, limits: Thresholds) -> dict[str, bool]:
    """Pass/fail per criterion. Quality thresholds only apply to cases that are expected to produce findings."""
    out = {"premise": premise_flagged == case.false_premise, "cache": metrics["cache_hit"] == case.expect_cache_hit,
           "findings": metrics["findings"] >= case.min_findings,
           "mentions": all(word.lower() in text.lower() for word in case.must_mention),
           "latency": metrics["seconds"] <= (limits.cache_hit_seconds if metrics["cache_hit"] else limits.full_run_seconds)}
    if case.min_findings:
        out |= {k: metrics[k] >= getattr(limits, k) for k in QUALITY}
    return out


def score(case: Case, res: dict[str, Any], seconds: float, prices: dict[str, tuple[float, float]], limits: Thresholds) -> dict[str, Any]:
    """One case's row in results.json: the run_cli contract metrics plus the premise verdict and the checks."""
    metrics = run_cli.contract_metrics(res, case.question, seconds, prices)
    summary = res.get("summary")
    verdict = summary.premise_verdict if summary else "none"
    flagged = verdict in ("unsupported", "false")
    result = checks(case, metrics, flagged, answer_text(res), limits)
    return {"id": case.id, "question": case.question, "false_premise": case.false_premise, "expect_cache_hit": case.expect_cache_hit,
            "premise_verdict": verdict, "error": "", "checks": result, "passed": all(result.values()), **metrics}


def failed_case(case: Case, seconds: float, error: str) -> dict[str, Any]:
    return {"id": case.id, "question": case.question, "false_premise": case.false_premise, "expect_cache_hit": case.expect_cache_hit,
            "premise_verdict": "none", "error": error,
            "checks": {"run": False}, "passed": False, "seconds": round(seconds, 2), "cache_hit": False, "findings": 0, "cost_usd": 0.0,
            **dict.fromkeys(QUALITY, 0.0)}


def rate(hits: list[bool]) -> float | None:
    return round(sum(hits) / len(hits), 3) if hits else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Scorecard headline: quality means over cases with findings, premise detection, cache hit rate, latency and cost."""
    ok = [r for r in rows if not r["error"]]
    answered = [r for r in ok if r["findings"] > 0]
    full, cached = [r["seconds"] for r in ok if not r["cache_hit"]], [r["seconds"] for r in ok if r["cache_hit"]]
    return {
        "cases": len(rows), "passed": sum(r["passed"] for r in rows), "errors": len(rows) - len(ok),
        **{k: round(ev.mean([r[k] for r in answered]), 3) if answered else None for k in QUALITY},
        "false_premise_detection": rate([r["premise_verdict"] in ("unsupported", "false") for r in ok if r["false_premise"]]),
        "false_premise_false_alarms": rate([r["premise_verdict"] in ("unsupported", "false") for r in ok if not r["false_premise"]]),
        "cache_hit_rate": rate([r["cache_hit"] for r in rows if r["expect_cache_hit"]]),
        "median_full_run_seconds": round(sorted(full)[len(full) // 2], 2) if full else None,
        "max_full_run_seconds": round(max(full), 2) if full else None,
        "max_cache_hit_seconds": round(max(cached), 2) if cached else None,
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
    }


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def scorecard(meta: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """SCORECARD.md: headline metrics, then one row per case with its failed checks."""
    banner = ("> **OFFLINE SMOKE TEST.** LLM, search and embeddings were scripted (`--offline`): these numbers test the eval harness, "
              "not the product. Run `python -m evals.run` with an OpenRouter key for the real scorecard.\n\n" if meta["mode"] == "offline" else "")
    head = [f"# Eval scorecard ({meta['mode']})", "", banner + f"Generated {meta['date']} · commit `{meta['commit']}` · {summary['cases']} cases "
            f"from `evals/golden.json` · command: `{meta['command']}`", "",
            "| Metric | Value |", "|---|---|",
            f"| Cases passed | {summary['passed']} / {summary['cases']} (errors: {summary['errors']}) |",
            *(f"| {k.replace('_', ' ').capitalize()} (mean over answered cases) | {pct(summary[k])} |" for k in QUALITY),
            f"| False-premise detection | {pct(summary['false_premise_detection'])} |",
            f"| False-premise false alarms | {pct(summary['false_premise_false_alarms'])} |",
            f"| Semantic-cache hit rate (cases expecting a hit) | {pct(summary['cache_hit_rate'])} |",
            f"| Median / max full run | {summary['median_full_run_seconds']} s / {summary['max_full_run_seconds']} s |",
            f"| Max cache-hit latency | {summary['max_cache_hit_seconds']} s |",
            f"| Total cost | ${summary['total_cost_usd']:.4f} |", "",
            "| Case | Pass | Findings | Faithfulness | Citation acc. | Relevance | Premise | Cache | Seconds | Cost | Failed checks |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    return "\n".join([*head, *map(case_row, rows)]) + "\n"


def case_row(r: dict[str, Any]) -> str:
    failed = ", ".join(k for k, v in r["checks"].items() if not v) or "—"
    cells = [f"`{r['id']}`", "✅" if r["passed"] else "❌", str(r["findings"]), *(pct(r[k]) for k in QUALITY), r["premise_verdict"],
             "hit" if r["cache_hit"] else "miss", str(r["seconds"]), f"${r['cost_usd']:.4f}", failed + (f" · {r['error']}" if r["error"] else "")]
    return "| " + " | ".join(cells) + " |"


# ---------------------------------------------------------------- running
def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@contextlib.contextmanager
def services(offline: bool) -> Iterator[tuple[Keys, dict[str, tuple[float, float]]]]:
    """Keys and live prices, or scripted services that never touch the network."""
    if offline:
        from evals.offline import offline_services

        with offline_services():
            yield Keys("offline", "offline-tavily", ""), tools.openrouter_models()
        return
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set: add it to .env, or run `python -m evals.run --offline` to test the harness.")
    yield Keys(key, os.getenv("TAVILY_API_KEY", ""), os.getenv("PARALLEL_API_KEY", "")), tools.openrouter_models()


async def run_case(case: Case, keys: Keys, prices: dict[str, tuple[float, float]], mem: memory.Memory, limits: Thresholds,
                   max_loops: int) -> dict[str, Any]:
    start = time.perf_counter()
    inputs = {"query": case.question, "auto_approve": True, "max_loops": max_loops,
              "default_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")}
    try:
        res = await research(keys, prices, mem, inputs, f"eval-{case.id}")
    except Exception as exc:  # one broken case is recorded as failed; the others still run
        return failed_case(case, time.perf_counter() - start, tools.err(exc))
    return score(case, res, time.perf_counter() - start, prices, limits)


async def run_all(golden: Golden, keys: Keys, prices: dict[str, tuple[float, float]], mem: memory.Memory, max_loops: int,
                  log: Any = print) -> list[dict[str, Any]]:
    """Cases run one after another on one memory (so the cache case can hit), each on its own session thread."""
    rows = []
    for case in golden.cases:
        row = await run_case(case, keys, prices, mem, golden.thresholds, max_loops)
        log(f"{'PASS' if row['passed'] else 'FAIL'} {case.id}: {row['seconds']}s, {row['findings']} findings"
            + (f", error: {row['error']}" if row["error"] else ""))
        rows.append(row)
    return rows


def write(out: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = aggregate(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    (out / "SCORECARD.md").write_text(scorecard(meta, summary, rows), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="scripted LLM, search and embeddings (no keys, no network)")
    parser.add_argument("--out", type=Path, help="output folder (default: evals/ live, runs/evals-offline offline)")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--max-loops", type=int, default=1)
    parser.add_argument("--memory-dir", type=Path, help="memory to evaluate against (default: a fresh temporary one)")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any case fails a check")
    args = parser.parse_args(argv)
    load_dotenv()
    os.environ["RF_TRACING"] = "false"
    golden = load(args.golden)
    if args.only:
        golden.cases = [c for c in golden.cases if c.id in set(args.only.split(","))]
    out = args.out or (HERE.parent / "runs" / "evals-offline" if args.offline else HERE)
    mode = "offline" if args.offline else "live"
    command = "python -m evals.run" + (" --offline" if args.offline else "")
    with tempfile.TemporaryDirectory(prefix="rf-evals-") as tmp, services(args.offline) as (keys, prices):
        mem = memory.Memory(args.memory_dir or tmp)
        rows = asyncio.run(run_all(golden, keys, prices, mem, args.max_loops))
    meta = {"mode": mode, "date": datetime.now().isoformat(timespec="seconds"), "commit": commit(), "command": command,
            "default_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"), "max_loops": args.max_loops}
    summary = write(out, meta, rows)
    print(json.dumps(summary))
    print(f"wrote {out / 'SCORECARD.md'} and {out / 'results.json'}")
    return 1 if summary["errors"] or (args.strict and summary["passed"] < summary["cases"]) else 0


if __name__ == "__main__":
    sys.exit(main())
