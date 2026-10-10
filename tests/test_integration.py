"""LIVE acceptance tests (network + .env keys; downloads the ONNX models once): pytest -m integration -s"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

import tools

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
QUESTION = "How is quantum computing going to change the current AI LLM/models and agentic implementations?"
pytestmark = pytest.mark.integration


def cli(question: str, out: Path, *extra: str) -> dict:  # type: ignore[type-arg]
    subprocess.run([sys.executable, str(ROOT / "run_cli.py"), "--question", question, "--out", str(out), *extra], check=True, cwd=ROOT)
    return json.loads((out / "metrics.json").read_text())


async def test_live_mcp_discovery() -> None:
    for prov in tools.providers(os.getenv("TAVILY_API_KEY", ""), os.getenv("PARALLEL_API_KEY", "")):
        names = await tools.discover(prov)
        print(prov.name, names)
        assert any("search" in n for n in names)


def test_live_acceptance_run_then_cache_hit(tmp_path: Path) -> None:
    first = cli(QUESTION, tmp_path / "run1", "--warmup", "--refresh")
    print(first)
    assert first["seconds"] < 90 and first["findings"] >= 8 and first["sources_cited"] >= 6 and not first["cache_hit"]
    second = cli(QUESTION, tmp_path / "run2")
    print(second)
    assert second["cache_hit"] and second["seconds"] < 5


def test_live_false_premise_is_corrected(tmp_path: Path) -> None:
    cli("Why have quantum computers already replaced GPUs for training LLMs?", tmp_path / "fp", "--refresh", "--max-loops", "0")
    report = (tmp_path / "fp" / "report.md").read_text()
    print(report[:1500])
    assert "Premise check" in report
