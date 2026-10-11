"""The eval harness: scoring rules, aggregation, the scorecard, and a full offline run of the golden set (as CI runs it)."""

import json
from pathlib import Path

import pytest

from evals import run as evals

ROW = {"cache_hit": False, "findings": 4, "seconds": 40.0, "faithfulness": 0.9, "citation_accuracy": 0.8, "answer_relevance": 0.7}


def test_golden_set_covers_false_premise_and_cache() -> None:
    golden = evals.load()
    ids = [c.id for c in golden.cases]
    assert 5 <= len(ids) <= 8 and len(set(ids)) == len(ids)
    assert any(c.false_premise for c in golden.cases) and any(c.expect_cache_hit for c in golden.cases)
    first_cache = next(i for i, c in enumerate(golden.cases) if c.expect_cache_hit)
    assert first_cache > 0  # the question it repeats has to run first, on the same memory


def test_checks_apply_quality_thresholds_only_where_findings_are_expected() -> None:
    limits = evals.Thresholds()
    case = evals.Case(id="a", question="q", min_findings=3, must_mention=["Quantum"])
    assert all(evals.checks(case, ROW, False, "quantum may help", limits).values())
    bad = evals.checks(case, ROW | {"faithfulness": 0.5, "seconds": 120.0}, True, "nothing relevant", limits)
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
