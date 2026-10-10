"""Second brain: local embeddings + reranker (fastembed, ONNX) and three vector tables
(evidence chunks, verified claims, cached reports) in LanceDB, or numpy + SQLite behind the same interface."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from models import Chunk, Claim, Source, StoredClaim

EMBED_MODEL, RERANK_MODEL, DIM = "BAAI/bge-small-en-v1.5", "Xenova/ms-marco-MiniLM-L-6-v2", 384
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / ".models"
CACHE_THRESHOLD = 0.92  # cosine for "same question" -> instant answer from memory
Hit = tuple[float, dict[str, Any]]


# ---------------------------------------------------------------- local models
def _hf_token() -> None:
    """huggingface_hub reads HF_TOKEN; accept HF_API_KEY from .env too (authenticated downloads are faster)."""
    if os.getenv("HF_API_KEY") and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HF_API_KEY"]


@lru_cache(maxsize=1)
def _embedder() -> Any:
    from fastembed import TextEmbedding

    _hf_token()
    return TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_DIR))


@lru_cache(maxsize=1)
def _cross_encoder() -> Any:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    _hf_token()
    return TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODEL_DIR))


_VECTORS: dict[str, np.ndarray] = {}


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalized bge-small vectors (dot product = cosine), cached per text so archiving never re-embeds."""
    if missing := [t for t in dict.fromkeys(texts) if t not in _VECTORS]:
        vecs = np.asarray(list(_embedder().embed(missing)), dtype=np.float32)
        _VECTORS.update(zip(missing, vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9), strict=True))
    return np.stack([_VECTORS[t] for t in texts]) if texts else np.zeros((0, DIM), dtype=np.float32)


def rerank(query: str, texts: list[str]) -> list[float] | None:
    """Cross-encoder relevance scores; None when the model can't load (callers fall back to an LLM rerank)."""
    try:
        return [float(s) for s in _cross_encoder().rerank(query, texts)]
    except Exception:
        return None


def warmup() -> None:
    """Download and load both ONNX models once (about 150 MB), so the first research run stays fast."""
    embed(["warm up"])
    rerank("warm", ["up"])


def chunk(src: Source, size: int = 2000, overlap: int = 200) -> list[Chunk]:
    """About 500-token, sentence-aligned chunks with a small overlap; PDF page numbers are kept."""
    out, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", src.text.strip()):
        if current and len(current) + len(sentence) > size:
            out.append(current.strip())
            current = current[-overlap:].split(" ", 1)[-1]
        current += sentence + " "
    if current.strip():
        out.append(current.strip())
    return [Chunk(source_id=src.id, text=t, page=src.page) for t in out]


# ---------------------------------------------------------------- storage backends (same interface)
def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class LanceTable:
    def __init__(self, db: Any, name: str) -> None:
        self.db, self.name = db, name

    def _open(self) -> Any:
        return self.db.open_table(self.name) if self.name in self.db.list_tables().tables else None

    def add(self, rows: list[dict[str, Any]]) -> None:
        if rows and (table := self._open()) is not None:
            table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)
        elif rows:
            self.db.create_table(self.name, data=rows)

    def search(self, vec: np.ndarray, k: int) -> list[Hit]:
        if (table := self._open()) is None:
            return []
        rows = table.search(vec.tolist()).distance_type("cosine").limit(k).to_list()
        return [(1 - float(r["_distance"]), json.loads(r["payload"])) for r in rows]

    def all(self) -> list[dict[str, Any]]:
        table = self._open()
        return [json.loads(r["payload"]) for r in table.to_arrow().select(["payload"]).to_pylist()] if table is not None else []


class SqliteTable:
    def __init__(self, conn: sqlite3.Connection, name: str) -> None:
        self.conn, self.name = conn, name
        conn.execute(f"CREATE TABLE IF NOT EXISTS {name} (id TEXT PRIMARY KEY, vector BLOB, payload TEXT)")

    def add(self, rows: list[dict[str, Any]]) -> None:
        self.conn.executemany(f"INSERT OR REPLACE INTO {self.name} VALUES (?, ?, ?)",
                              [(r["id"], np.asarray(r["vector"], dtype=np.float32).tobytes(), r["payload"]) for r in rows])
        self.conn.commit()

    def search(self, vec: np.ndarray, k: int) -> list[Hit]:
        rows = self.conn.execute(f"SELECT vector, payload FROM {self.name}").fetchall()
        if not rows:
            return []
        sims = np.stack([np.frombuffer(v, dtype=np.float32) for v, _ in rows]) @ vec
        return [(float(sims[i]), json.loads(rows[i][1])) for i in np.argsort(-sims)[:k]]

    def all(self) -> list[dict[str, Any]]:
        return [json.loads(p) for (p,) in self.conn.execute(f"SELECT payload FROM {self.name}")]


# ---------------------------------------------------------------- the second brain
class Memory:
    def __init__(self, path: Path | str | None = None, backend: str = "auto") -> None:
        root = Path(path or os.getenv("RF_MEMORY_DIR") or HERE / ".memory")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        try:
            if backend == "sqlite":
                raise ImportError
            import lancedb

            db: Any = lancedb.connect(str(root / "lance"))
            tables: list[Any] = [LanceTable(db, n) for n in ("chunks", "claims", "reports")]
            self.backend = "lancedb"
        except ImportError:  # LanceDB unavailable: numpy + SQLite, same interface
            conn = sqlite3.connect(root / "memory.sqlite", check_same_thread=False)
            tables, self.backend = [SqliteTable(conn, n) for n in ("chunks", "claims", "reports")], "sqlite"
        self.chunks, self.claims, self.reports = tables

    def cached_report(self, qvec: np.ndarray) -> Hit | None:
        """Semantic cache: a near-identical earlier question returns its stored report."""
        hits = self.reports.search(qvec, 1)
        return hits[0] if hits and hits[0][0] >= CACHE_THRESHOLD else None

    def similar_claims(self, qvec: np.ndarray, k: int = 6, floor: float = 0.6) -> list[StoredClaim]:
        return [StoredClaim(**p, similarity=round(s, 3)) for s, p in self.claims.search(qvec, k) if s >= floor]

    def evidence(self, qvec: np.ndarray, k: int = 8, floor: float = 0.55) -> list[Source]:
        """Previously fetched chunks relevant to this question: memory is checked before the web."""
        return [Source(**p, from_memory=True) for s, p in self.chunks.search(qvec, k) if s >= floor]

    def log_run(self, record: dict[str, Any]) -> None:
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def history(self, limit: int = 200) -> list[dict[str, Any]]:
        """Past runs, newest first."""
        path = self.root / "history.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        return [json.loads(line) for line in reversed(lines[-limit:]) if line.strip()]

    def library(self) -> list[StoredClaim]:
        return sorted((StoredClaim(**p) for p in self.claims.all()), key=lambda c: c.date, reverse=True)

    def remember(self, question: str, qvec: np.ndarray, snapshot: str, claims: list[Claim], sources: list[Source],
                 chunks: list[Chunk], chunk_vecs: np.ndarray) -> None:
        """Store evidence chunks, verified claims (never 'not established' ones) and the report for the cache."""
        by_id = {s.id: s for s in sources}
        self.chunks.add([
            {"id": _key(f"{by_id[c.source_id].url}|{c.text}"), "vector": v.tolist(),
             "payload": json.dumps(by_id[c.source_id].model_dump(exclude={"id", "text", "from_memory"}) | {"text": c.text})}
            for c, v in zip(chunks, chunk_vecs, strict=True) if c.source_id in by_id
        ])
        keep = [c for c in claims if c.status != "not_established"]
        stored = [StoredClaim(text=c.text, quotes=[q.quote for q in c.evidence], status=c.status, support=c.support,
                              sources=[f"{by_id[s].title} <{by_id[s].url}>" for s in c.citations if s in by_id],
                              question=question, date=date.today().isoformat()) for c in keep]
        self.claims.add([{"id": _key(s.text), "vector": v.tolist(), "payload": s.model_dump_json(exclude={"similarity"})}
                         for s, v in zip(stored, embed([s.text for s in stored]), strict=True)])
        self.reports.add([{"id": _key(question.strip().lower()), "vector": qvec.tolist(), "payload": json.dumps({"state": snapshot})}])
