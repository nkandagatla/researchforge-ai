"""Run the golden set through the full research pipeline and score it.

    python -m evals.run                 # live: needs OPENROUTER_API_KEY (Tavily/Parallel optional); writes evals/SCORECARD.md + results.json
    python -m evals.run --offline       # scripted LLM/search/models (no keys, no network): tests the harness itself, as CI does

Per case it scores faithfulness, citation accuracy and answer relevance (the app's trust metrics), false-premise detection,
semantic-cache behaviour, keyword checks on the answer (required and forbidden phrases), latency and cost, then aggregates
them into a scorecard.
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
import log as rflog
import memory
import run_cli
import tools
from agents import Keys, research

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden.json"
QUALITY = ("faithfulness", "citation_accuracy", "answer_relevance")


class Case(BaseModel):
    """One golden question and what a good answer must (and must not) do."""

    id: str
    question: str
    false_premise: bool = False
    expect_cache_hit: bool = False
    repeat_of: str = ""  # id of an earlier case whose memory this one reuses (the semantic-cache check)
    min_findings: int = 0
    must_mention: list[str] = Field(default_factory=list)  # every word must appear in the answer (case-insensitive)
    must_not_mention: list[str] = Field(default_factory=list)  # none may appear: e.g. complying with an injected instruction
    note: str = ""


class Thresholds(BaseModel):
    """Pass bars for the quality metrics and the latency limits (full run vs semantic-cache hit)."""

    faithfulness: float = 0.7
    citation_accuracy: float = 0.7
    answer_relevance: float = 0.6
    full_run_seconds: float = 150
    cache_hit_seconds: float = 5


class Golden(BaseModel):
    """The golden set file: thresholds plus the ordered cases."""

    description: str = ""
    thresholds: Thresholds = Thresholds()
    cases: list[Case]


def load(path: Path = GOLDEN) -> Golden:
    """Read and validate a golden set (evals/golden.json by default)."""
    return Golden.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- scoring (pure, unit-tested)
def answer_text(res: dict[str, Any]) -> str:
    """Everything the user reads as the answer: bottom line, summary and the verified findings."""
    summary = res.get("summary")
    parts = [summary.bottom_line, summary.executive_summary] if summary else []
    return " ".join(parts + [c.text for c in ev.answer_claims(res.get("claims", []))])


def mentions(text: str, phrases: list[str]) -> list[str]:
    """The phrases that occur in text, case-insensitively."""
    low = text.lower()
    return [p for p in phrases if p.lower() in low]


def checks(case: Case, metrics: dict[str, Any], premise_flagged: bool, text: str, limits: Thresholds) -> dict[str, bool]:
    """Pass/fail per criterion. Quality thresholds only apply to cases that are expected to produce findings, and the
    "avoids" check only to cases with must_not_mention."""
    out = {"premise": premise_flagged == case.false_premise, "cache": metrics["cache_hit"] == case.expect_cache_hit,
           "findings": metrics["findings"] >= case.min_findings,
           "mentions": len(mentions(text, case.must_mention)) == len(case.must_mention),
           "latency": metrics["seconds"] <= (limits.cache_hit_seconds if metrics["cache_hit"] else limits.full_run_seconds)}
    if case.must_not_mention:
        out["avoids"] = not mentions(text, case.must_not_mention)
    if case.min_findings:
        out |= {k: metrics[k] >= getattr(limits, k) for k in QUALITY}
    return out


def score(case: Case, res: dict[str, Any], seconds: float, prices: dict[str, tuple[float, float]], limits: Thresholds) -> dict[str, Any]:
    """One case's row in results.json: the run_cli contract metrics plus the premise verdict and the checks."""
    metrics = run_cli.contract_metrics(res, case.question, seconds, prices)
    summary = res.get("summary")
    verdict = summary.premise_verdict if summary else "none"
    result = checks(case, metrics, verdict in ("unsupported", "false"), answer_text(res), limits)
    return {"id": case.id, "question": case.question, "false_premise": case.false_premise, "expect_cache_hit": case.expect_cache_hit,
            "premise_verdict": verdict, "error": "", "checks": result, "passed": all(result.values()), **metrics}


def failed_case(case: Case, seconds: float, error: str) -> dict[str, Any]:
    """The results row for a case whose run raised: every check fails and the redacted error is kept."""
    return {"id": case.id, "question": case.question, "false_premise": case.false_premise, "expect_cache_hit": case.expect_cache_hit,
            "premise_verdict": "none", "error": error,
            "checks": {"run": False}, "passed": False, "seconds": round(seconds, 2), "cache_hit": False, "findings": 0, "cost_usd": 0.0,
            **dict.fromkeys(QUALITY, 0.0)}


def rate(hits: list[bool]) -> float | None:
    """Share of True values, rounded; None when there is nothing to rate."""
    return round(sum(hits) / len(hits), 3) if hits else None


def flagged(row: dict[str, Any]) -> bool:
    """True when the report called the question's premise unsupported or false."""
    return row["premise_verdict"] in ("unsupported", "false")


def median(values: list[float]) -> float | None:
    """Upper median, rounded; None for no values."""
    return round(sorted(values)[len(values) // 2], 2) if values else None


def latency_summary(ok: list[dict[str, Any]]) -> dict[str, float | None]:
    """Median and max full-run seconds and the slowest cache hit, over cases that ran."""
    full, cached = [r["seconds"] for r in ok if not r["cache_hit"]], [r["seconds"] for r in ok if r["cache_hit"]]
    return {"median_full_run_seconds": median(full), "max_full_run_seconds": max(full, default=None),
            "max_cache_hit_seconds": max(cached, default=None)}


def premise_summary(ok: list[dict[str, Any]]) -> dict[str, float | None]:
    """False-premise detection rate and false-alarm rate on normal questions."""
    return {"false_premise_detection": rate([flagged(r) for r in ok if r["false_premise"]]),
            "false_premise_false_alarms": rate([flagged(r) for r in ok if not r["false_premise"]])}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Scorecard headline: quality means over cases with findings, premise detection, cache hit rate, latency and cost."""
    ok = [r for r in rows if not r["error"]]
    answered = [r for r in ok if r["findings"] > 0]
    quality = {k: round(ev.mean([r[k] for r in answered]), 3) if answered else None for k in QUALITY}
    return {
        "cases": len(rows), "passed": sum(r["passed"] for r in rows), "errors": len(rows) - len(ok), **quality, **premise_summary(ok),
        "cache_hit_rate": rate([r["cache_hit"] for r in rows if r["expect_cache_hit"]]),
        **latency_summary(ok), "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
    }


def pct(v: float | None) -> str:
    """A rate as a whole percentage, 'n/a' for None."""
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
    """One Markdown table row of the scorecard, with the failed checks and any error."""
    failed = ", ".join(k for k, v in r["checks"].items() if not v) or "—"
    cells = [f"`{r['id']}`", "✅" if r["passed"] else "❌", str(r["findings"]), *(pct(r[k]) for k in QUALITY), r["premise_verdict"],
             "hit" if r["cache_hit"] else "miss", str(r["seconds"]), f"${r['cost_usd']:.4f}", failed + (f" · {r['error']}" if r["error"] else "")]
    return "| " + " | ".join(cells) + " |"


# ---------------------------------------------------------------- running
def commit() -> str:
    """Short git hash of the evaluated code ('unknown' outside a checkout)."""
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
    """Run one case through research() on its own thread and score it; an exception becomes an error row."""
    start = time.perf_counter()
    inputs = {"query": case.question, "auto_approve": True, "max_loops": max_loops,
              "default_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")}
    try:
        res = await research(keys, prices, mem, inputs, f"eval-{case.id}")
    except Exception as exc:  # one broken case is recorded as failed; the others still run
        return failed_case(case, time.perf_counter() - start, tools.err(exc))
    return score(case, res, time.perf_counter() - start, prices, limits)


def case_memory(case: Case, root: Path, shared: memory.Memory | None, seen: dict[str, memory.Memory]) -> memory.Memory:
    """Each case gets a fresh memory so earlier answers can't leak in as evidence, except a repeat, which reuses the
    memory of the case it repeats (that is what the semantic-cache check measures). --memory-dir shares one memory."""
    if shared is not None:
        return shared
    mem = seen[case.repeat_of] if case.repeat_of in seen else memory.Memory(root / case.id)
    seen[case.id] = mem
    return mem


async def run_all(golden: Golden, keys: Keys, prices: dict[str, tuple[float, float]], root: Path, max_loops: int,
                  log: Any = print, shared: memory.Memory | None = None) -> list[dict[str, Any]]:
    """Cases run one after another, each on its own session thread and (unless shared) its own fresh memory."""
    rows: list[dict[str, Any]] = []
    seen: dict[str, memory.Memory] = {}
    for case in golden.cases:
        mem = case_memory(case, root, shared, seen)
        row = await run_case(case, keys, prices, mem, golden.thresholds, max_loops)
        log(f"{'PASS' if row['passed'] else 'FAIL'} {case.id}: {row['seconds']}s, {row['findings']} findings"
            + (f", error: {row['error']}" if row["error"] else ""))
        rows.append(row)
    return rows


def write(out: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the rows and write results.json and SCORECARD.md to out; returns the summary."""
    summary = aggregate(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    (out / "SCORECARD.md").write_text(scorecard(meta, summary, rows), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """CLI flags for python -m evals.run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="scripted LLM, search and embeddings (no keys, no network)")
    parser.add_argument("--out", type=Path, help="output folder (default: evals/ live, runs/evals-offline offline)")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--max-loops", type=int, default=1)
    parser.add_argument("--memory-dir", type=Path, help="memory to evaluate against (default: a fresh temporary one)")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any case fails a check")
    return parser.parse_args(argv)


def evaluate(golden: Golden, offline: bool, max_loops: int, memory_dir: Path | None) -> list[dict[str, Any]]:
    """Run every case on a fresh temporary memory of its own, or all on memory_dir when it is given."""
    with tempfile.TemporaryDirectory(prefix="rf-evals-") as tmp, services(offline) as (keys, prices):
        shared = memory.Memory(memory_dir) if memory_dir else None
        return asyncio.run(run_all(golden, keys, prices, Path(tmp), max_loops, shared=shared))


def run_meta(offline: bool, max_loops: int) -> dict[str, Any]:
    """Run metadata for the scorecard header: mode, date, commit, command, model and loop limit."""
    return {"mode": "offline" if offline else "live", "date": datetime.now().isoformat(timespec="seconds"), "commit": commit(),
            "command": "python -m evals.run" + (" --offline" if offline else ""),
            "default_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"), "max_loops": max_loops}


def main(argv: list[str] | None = None) -> int:
    """Run the golden set and write the scorecard; exit 1 on any errored case (or any failed case with --strict)."""
    args = parse_args(argv)
    load_dotenv()
    rflog.configure()  # RF_LOG_LEVEL (default INFO) to stderr; the per-case PASS/FAIL lines stay on stdout
    os.environ["RF_TRACING"] = "false"
    golden = load(args.golden)
    if args.only:
        golden.cases = [c for c in golden.cases if c.id in set(args.only.split(","))]
    rows = evaluate(golden, args.offline, args.max_loops, args.memory_dir)
    out = args.out or (HERE.parent / "runs" / "evals-offline" if args.offline else HERE)
    summary = write(out, run_meta(args.offline, args.max_loops), rows)
    print(json.dumps(summary))
    print(f"wrote {out / 'SCORECARD.md'} and {out / 'results.json'}")
    return 1 if summary["errors"] or (args.strict and summary["passed"] < summary["cases"]) else 0


if __name__ == "__main__":
    sys.exit(main())
