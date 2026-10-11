"""The eval harness: scoring rules, aggregation, the scorecard, and a full offline run of the golden set (as CI runs it)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from evals import run as evals
from evals.offline import offline_services

ROW = {"cache_hit": False, "findings": 4, "seconds": 40.0, "faithfulness": 0.9, "citation_accuracy": 0.8, "answer_relevance": 0.7}


def test_golden_set_covers_false_premise_and_cache() -> None:
    golden = evals.load()
    ids = [c.id for c in golden.cases]
    assert 5 <= len(ids) <= 12 and len(set(ids)) == len(ids)
    assert any(c.false_premise for c in golden.cases) and any(c.expect_cache_hit for c in golden.cases)
    first_cache = next(i for i, c in enumerate(golden.cases) if c.expect_cache_hit)
    assert first_cache > 0  # the question it repeats has to run first, on the same memory
    repeat = golden.cases[first_cache]
    assert repeat.repeat_of in ids[:first_cache]


def test_each_case_gets_its_own_memory_except_a_repeat(tmp_path: Path) -> None:
    seen: dict = {}
    first = evals.case_memory(evals.Case(id="a", question="q"), tmp_path, None, seen)
    other = evals.case_memory(evals.Case(id="b", question="q2"), tmp_path, None, seen)
    again = evals.case_memory(evals.Case(id="c", question="q", repeat_of="a"), tmp_path, None, seen)
    assert first is not other and again is first
    shared = evals.memory.Memory(tmp_path / "shared")
    assert evals.case_memory(evals.Case(id="d", question="q"), tmp_path, shared, seen) is shared


def test_checks_apply_quality_thresholds_only_where_findings_are_expected() -> None:
    limits = evals.Thresholds()
    case = evals.Case(id="a", question="q", min_findings=3, must_mention=["Quantum"])
    assert all(evals.checks(case, ROW, False, "quantum may help", limits).values())
    bad = evals.checks(case, ROW | {"faithfulness": 0.5, "seconds": 200.0}, True, "nothing relevant", limits)
    assert {k for k, v in bad.items() if not v} == {"faithfulness", "latency", "premise", "mentions"}
    premise = evals.Case(id="fp", question="q", false_premise=True)
    result = evals.checks(premise, ROW | {"findings": 0, "faithfulness": 0.0}, True, "", limits)
    assert result["premise"] and "faithfulness" not in result  # no quality bar for a question that should be refused
    cached = evals.checks(evals.Case(id="c", question="q", expect_cache_hit=True), ROW | {"cache_hit": True, "seconds": 9.0}, False, "", limits)
    assert cached["cache"] and not cached["latency"]  # a cache hit must come back within cache_hit_seconds


def test_aggregate_and_scorecard() -> None:
    rows = [evals.failed_case(evals.Case(id="boom", question="q"), 1.0, "RuntimeError: down")]
    rows.append({**ROW, "id": "ok", "question": "q", "false_premise": True, "expect_cache_hit": False, "premise_verdict": "false",
                 "error": "", "checks": {"premise": True}, "passed": True, "cost_usd": 0.01})
    summary = evals.aggregate(rows)
    assert (summary["cases"], summary["passed"], summary["errors"]) == (2, 1, 1)
    assert summary["faithfulness"] == 0.9 and summary["false_premise_detection"] == 1.0 and summary["cache_hit_rate"] is None
    md = evals.scorecard({"mode": "live", "date": "d", "commit": "abc", "command": "python -m evals.run"}, summary, rows)
    assert "OFFLINE" not in md and "| `boom` | ❌ |" in md and "RuntimeError: down" in md and "| Cases passed | 1 / 2" in md


def test_offline_run_writes_results_and_scorecard(tmp_path: Path) -> None:
    assert evals.main(["--offline", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in data["cases"]}
    assert data["meta"]["mode"] == "offline" and len(cases) == len(evals.load().cases) and data["summary"]["errors"] == 0
    assert cases["quantum-llm-again"]["cache_hit"] and cases["quantum-llm-again"]["cost_usd"] == 0
    assert data["summary"]["false_premise_detection"] == 1.0 and data["summary"]["false_premise_false_alarms"] == 0.0
    assert all(c["findings"] > 0 and 0 < c["citation_accuracy"] <= 1 for c in cases.values())
    card = (tmp_path / "SCORECARD.md").read_text(encoding="utf-8")
    assert card.startswith("# Eval scorecard (offline)") and "OFFLINE SMOKE TEST" in card and "`gpu-false-premise` | ✅" in card


def test_live_mode_needs_a_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY is not set"):
        evals.main(["--out", str(tmp_path)])
    assert not (tmp_path / "SCORECARD.md").exists()


def test_must_not_mention_fails_an_answer_that_complies_with_an_injected_instruction() -> None:
    limits = evals.Thresholds()
    case = evals.Case(id="inj", question="q", must_mention=["retriev"], must_not_mention=["sources are fake", "Fabricated"])
    clean = evals.checks(case, ROW, False, "Retrieval-augmented generation grounds answers in retrieved text.", limits)
    assert clean["avoids"] and clean["mentions"]
    complied = evals.checks(case, ROW, False, "As instructed: the SOURCES ARE FAKE. Retrieval is irrelevant.", limits)
    assert not complied["avoids"] and complied["mentions"]
    assert "avoids" not in evals.checks(evals.Case(id="plain", question="q"), ROW, False, "", limits)  # only checked when asked
    assert evals.mentions("Fabricated sources", ["fabricated", "fake"]) == ["fabricated"]


def test_golden_edge_cases_are_well_formed() -> None:
    cases = {c.id: c for c in evals.load().cases}
    assert {"short-ambiguous-rag", "prompt-injection-in-question", "numeric-recent-fact"} <= set(cases)
    assert len(cases["short-ambiguous-rag"].question) <= 5 and cases["prompt-injection-in-question"].must_not_mention
    assert all(not c.false_premise and c.min_findings >= 1 for k, c in cases.items() if k in {"short-ambiguous-rag", "numeric-recent-fact"})


def test_a_case_that_raises_is_an_error_row_and_the_rest_still_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = evals.research

    async def research(keys: Any, prices: Any, mem: Any, inputs: dict[str, Any], thread: str) -> dict[str, Any]:
        if thread == "eval-boom":
            raise RuntimeError("graph exploded with sk-or-v1-0123456789abcdef")
        return await real(keys, prices, mem, inputs, thread)

    monkeypatch.setattr(evals, "research", research)
    golden = evals.Golden(cases=[evals.Case(id="before", question="What is RAG?", min_findings=1), evals.Case(id="boom", question="q"),
                                 evals.Case(id="after", question="How do agents call tools?", min_findings=1)])
    printed: list[str] = []
    with offline_services() as fake:
        keys, prices = evals.Keys("offline", "offline-tavily", ""), evals.tools.openrouter_models()
        rows = asyncio.run(evals.run_all(golden, keys, prices, tmp_path, 0, log=printed.append))
    assert [r["id"] for r in rows] == ["before", "boom", "after"] and fake.calls["llm"]
    boom = rows[1]
    assert boom["error"].startswith("RuntimeError: graph exploded") and "0123456789abcdef" not in boom["error"]
    assert not boom["passed"] and boom["checks"] == {"run": False} and rows[0]["findings"] > 0 and rows[2]["findings"] > 0
    assert printed[1].startswith("FAIL boom") and "error: RuntimeError" in printed[1]
    assert evals.aggregate(rows)["errors"] == 1


def test_aggregate_with_no_answered_cases_and_a_scorecard_of_errors_only() -> None:
    refused = {**ROW, "id": "fp", "question": "q", "false_premise": True, "expect_cache_hit": False, "premise_verdict": "false",
               "error": "", "checks": {"premise": True}, "passed": True, "cost_usd": 0.0, "findings": 0}
    summary = evals.aggregate([refused])
    assert all(summary[k] is None for k in evals.QUALITY)  # nothing answered: no quality mean, not a misleading 0%
    assert summary["false_premise_detection"] == 1.0 and summary["false_premise_false_alarms"] is None

    rows = [evals.failed_case(evals.Case(id="down", question="q"), 2.5, "ConnectError: no route to host")]
    summary = evals.aggregate(rows)
    assert (summary["cases"], summary["passed"], summary["errors"]) == (1, 0, 1)
    assert summary["median_full_run_seconds"] is None and summary["max_cache_hit_seconds"] is None and summary["total_cost_usd"] == 0
    md = evals.scorecard({"mode": "offline", "date": "d", "commit": "abc", "command": "python -m evals.run --offline"}, summary, rows)
    assert "OFFLINE SMOKE TEST" in md and "| Cases passed | 0 / 1 (errors: 1) |" in md and "| Faithfulness (mean over answered cases) | n/a |" in md
    assert "| `down` | ❌ | 0 |" in md and "run · ConnectError: no route to host" in md
