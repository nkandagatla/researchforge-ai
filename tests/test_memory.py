import numpy as np
import pytest
from conftest import src

import evidence as ev
import memory
from models import Claim, Quote


def test_chunking_size_overlap_and_pages() -> None:
    text = " ".join(f"Sentence number {i} talks about quantum hardware and training." for i in range(120))
    chunks = memory.chunk(src("p", text, "pdf", page=7).model_copy(update={"id": "S1"}))
    assert len(chunks) > 3 and all(len(c.text) <= 2100 and c.page == 7 and c.source_id == "S1" for c in chunks)
    assert chunks[1].text.split(".")[0] in chunks[0].text  # overlap carries context across the boundary


def test_embeddings_are_normalized_and_cached() -> None:
    vecs = memory.embed(["quantum llm", "quantum llm", "gpu"])
    assert vecs.shape == (3, memory.DIM) and np.allclose(np.linalg.norm(vecs, axis=1), 1) and np.allclose(vecs[0], vecs[1])
    assert len(memory._VECTORS) == 2 and memory.embed([]).shape == (0, memory.DIM)


@pytest.mark.parametrize("backend", ["auto", "sqlite"])
def test_second_brain_semantic_cache_claims_and_evidence(tmp_path, backend: str) -> None:  # type: ignore[no-untyped-def]
    mem = memory.Memory(tmp_path, backend=backend)
    assert mem.backend == ("lancedb" if backend == "auto" else "sqlite") and mem.cached_report(memory.embed(["anything"])[0]) is None
    sources, _ = ev.assign_ids([src("Quantum energy paper", "Quantum computing could reduce the energy cost of training models.", "arxiv")], {})
    q = Quote(source_id="S1", quote="Quantum computing could reduce the energy cost of training models.")
    claims = [Claim(text="Quantum computing may cut LLM training energy", evidence=[q], support=1.0),
              Claim(text="Fabricated claim never stored", evidence=[q], status="not_established")]
    question = "How will quantum computing change LLM training?"
    chunks = memory.chunk(sources[0])
    mem.remember(question, memory.embed([question])[0], '{"query": "q"}', claims, sources, chunks, memory.embed([c.text for c in chunks]))
    hit = mem.cached_report(memory.embed(["how will quantum computing change llm training"])[0])
    assert hit is not None and hit[0] >= memory.CACHE_THRESHOLD and hit[1]["state"] == '{"query": "q"}'
    assert mem.cached_report(memory.embed(["best pizza dough recipe"])[0]) is None
    lib = mem.library()
    assert [c.text for c in lib] == ["Quantum computing may cut LLM training energy"] and lib[0].sources[0].startswith("Quantum energy paper")
    assert mem.similar_claims(memory.embed(["quantum computing LLM training energy"])[0])[0].similarity > 0.6
    found = mem.evidence(memory.embed(["quantum computing energy cost training"])[0])
    assert found and found[0].from_memory and found[0].provider == "arxiv" and "energy cost" in found[0].text
    mem.remember(question, memory.embed([question])[0], '{"query": "v2"}', claims, sources, chunks, memory.embed([c.text for c in chunks]))
    assert len(mem.library()) == 1 and mem.cached_report(memory.embed([question])[0])[1]["state"] == '{"query": "v2"}'  # upsert, no dupes


def test_rerank_returns_none_when_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise OSError("no model")

    monkeypatch.setattr(memory, "_cross_encoder", broken)
    assert memory.rerank("q", ["a"]) is None


def test_run_history_is_logged_newest_first(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mem = memory.Memory(tmp_path)
    assert mem.history() == []
    for i in range(3):
        mem.log_run({"question": f"q{i}", "cache_hit": False})
    assert [r["question"] for r in mem.history()] == ["q2", "q1", "q0"] and len(mem.history(limit=2)) == 2
