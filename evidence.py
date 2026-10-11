"""Deterministic evidence logic (no LLM, no I/O): dedup, stable ids, hybrid retrieval + MMR,
quote verification, consensus, credibility, trust metrics and the Markdown report."""

from __future__ import annotations

import html
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from models import Chunk, Claim, ClaimDraft, Hypothesis, Quote, Snippet, Source, StoredClaim, Summary, TokenUsage, TraceEvent, Trust, Vote

STOP = set(
    "the and for with that this from are was were has have had not but its into than then their there what which when "
    "where who how why does did can could would should will about over under between also more most such these those "
    "they them been being use used using via per our your".split()
)
INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|disregard .{0,30}instructions|you are now|system prompt"
    r"|</?(system|assistant|untrusted_source)[^>]*>)",
    re.I,
)
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")


# ---------------------------------------------------------------- identity, dedup, stable ids
def normalize_url(url: str) -> str:
    p = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")])
    return urlunsplit((p.scheme.lower() or "https", p.netloc.lower().removeprefix("www."), p.path.rstrip("/"), query, ""))


def source_keys(src: Source) -> list[str]:
    """Strongest identity first: DOI, arXiv id (also from abs/pdf/html URLs), normalized title, URL."""
    keys = []
    if src.doi:
        keys.append("doi:" + src.doi.lower().removeprefix("https://doi.org/"))
    match = ARXIV_URL.search(src.url)
    if arxiv_id := src.arxiv_id or (match.group(1) if match else None):
        keys.append("arxiv:" + arxiv_id.split("v")[0])
    title = " ".join(re.findall(r"[a-z0-9]+", src.title.lower().split(" - ")[0].split(" | ")[0]))
    if len(title) > 25:
        keys.append("title:" + title)
    if src.url:
        keys.append("url:" + normalize_url(src.url) + (f"#p{src.page}" if src.page else ""))
    return keys or ["title:" + src.title]


def dedupe(sources: list[Source]) -> list[Source]:
    """A paper and its preprint, or a URL with tracking params, collapse into one record (richest text kept)."""
    kept: list[Source] = []
    index: dict[str, int] = {}
    for src in sources:
        keys = source_keys(src)
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            hit = len(kept)
            kept.append(src)
        elif len(src.text) > len(kept[hit].text):
            kept[hit] = kept[hit].model_copy(update={"text": src.text})
        index |= dict.fromkeys(keys, hit)
    return kept


def assign_ids(sources: list[Source], ids: dict[str, str]) -> tuple[list[Source], dict[str, str]]:
    """Stable S-ids for the whole session, so claims from an earlier loop keep pointing at the same source."""
    ids = dict(ids)
    out = []
    for s in sources:
        keys = source_keys(s)
        sid = next((ids[k] for k in keys if k in ids), None) or f"S{len(set(ids.values())) + 1}"
        ids |= dict.fromkeys(keys, sid)
        out.append(s.model_copy(update={"id": sid}))
    return out, ids


# ---------------------------------------------------------------- ranking + hybrid retrieval
def terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in STOP]


def overlap(needle: set[str], text: str) -> float:
    return len(needle & set(terms(text))) / len(needle) if needle else 0.0


def sanitize(text: str) -> str:
    return INJECTION.sub("[removed]", text).replace("\x00", "")


def rank(sources: list[Source], query: str, limit: int) -> list[Source]:
    """Relevance x credibility with provider round-robin, so no single provider crowds out the others."""
    needle = set(terms(query))
    scored = sorted((s for s in sources if s.text), key=lambda s: overlap(needle, f"{s.title} {s.text}") + 0.2 * credibility(s), reverse=True)
    buckets: dict[str, list[Source]] = {}
    for s in scored:
        buckets.setdefault(s.provider, []).append(s)
    picked: list[Source] = []
    while len(picked) < limit and any(buckets.values()):
        for queue in [b for b in buckets.values() if b]:
            if len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def bm25(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized = [terms(d) for d in docs]
    avg = sum(map(len, tokenized)) / max(len(tokenized), 1) or 1.0
    df = Counter(t for doc in tokenized for t in set(doc))
    scores = np.zeros(len(docs))
    for i, doc in enumerate(tokenized):
        tf = Counter(doc)
        for t in set(terms(query)):
            if t in tf:
                idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                scores[i] += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avg))
    return scores


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion of several best-first index lists."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1 / (k + pos + 1)
    return sorted(fused, key=lambda i: -fused[i])


def mmr(qvec: np.ndarray, vecs: np.ndarray, candidates: list[int], k: int, lam: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance: relevant to the question, but not a near-copy of what's already picked."""
    picked: list[int] = []
    pool = list(candidates)
    while pool and len(picked) < k:
        best = max(pool, key=lambda i: lam * float(vecs[i] @ qvec) - (1 - lam) * max((float(vecs[i] @ vecs[j]) for j in picked), default=0.0))
        picked.append(best)
        pool.remove(best)
    return picked


def near_duplicates(vecs: np.ndarray, threshold: float = 0.95) -> list[int]:
    """Indexes to drop: chunks whose embedding almost equals an earlier one (mirrors, preprint copies)."""
    drop: list[int] = []
    for i in range(len(vecs)):
        if any(float(vecs[i] @ vecs[j]) >= threshold for j in range(i) if j not in drop):
            drop.append(i)
    return drop


def hybrid_select(question: str, qvec: np.ndarray, chunks: list[Chunk], vecs: np.ndarray, k: int,
                  reranker: Callable[[str, list[str]], list[float] | None], pool: int = 24) -> list[int]:
    """BM25 + dense fused by RRF -> rerank the top pool -> MMR down to k."""
    if not chunks:
        return []
    dense = list(np.argsort(-(vecs @ qvec)))
    lexical = list(np.argsort(-bm25(question, [c.text for c in chunks])))
    top = rrf([int(i) for i in dense], [int(i) for i in lexical])[:pool]
    scores = reranker(question, [chunks[i].text for i in top])
    if scores is not None:
        top = [i for _, i in sorted(zip(scores, top, strict=True), key=lambda p: -p[0])]
    return mmr(qvec, vecs, top[: max(k * 2, k)], k)


def shorten(text: str, size: int = 150) -> str:
    """One line, at most size characters, cut at a word boundary."""
    flat = " ".join(text.split())
    return flat if len(flat) <= size else flat[:size].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def previews(sources: list[Source], per_provider: int = 1, size: int = 150) -> list[Snippet]:
    """Short, sanitized previews of the first results of each provider, for the live agent activity."""
    seen: Counter[str] = Counter()
    out: list[Snippet] = []
    for s in sources:
        if s.text and seen[s.provider] < per_provider:
            seen[s.provider] += 1
            out.append(Snippet(provider=s.provider, title=shorten(s.title, 90), url=s.url, text=shorten(sanitize(s.text), size)))
    return out


def build_context(chunks: list[Chunk], by_id: dict[str, Source]) -> str:
    """Web text is untrusted: sanitized and fenced, grouped per source id."""
    grouped: dict[str, list[str]] = {}
    for c in chunks:
        grouped.setdefault(c.source_id, []).append(sanitize(c.text))
    return "\n\n".join(
        f'<untrusted_source id="{sid}" title="{html.escape(by_id[sid].title)}" provider="{by_id[sid].provider}" '
        f'published="{by_id[sid].published or "n.d."}">\n' + "\n...\n".join(texts) + "\n</untrusted_source>"
        for sid, texts in grouped.items()
    )


# ---------------------------------------------------------------- quote verification
def normalize_quote(value: str) -> str:
    """Tolerant of whitespace, HTML entities, markdown escapes, smart quotes and case; never of word changes."""
    value = unicodedata.normalize("NFKC", html.unescape(re.sub(r"\\([\W_])", r"\1", value)))
    return " ".join(re.findall(r"\d+(?:[.,]\d+)*|[^\W\d_]+", value.lower()))


def quote_in_source(quote: str, text: str) -> bool:
    """Every '...'-separated fragment (>= 4 words) must appear verbatim in the source."""
    haystack = f" {normalize_quote(text)} "
    parts = [p for p in (normalize_quote(x) for x in re.split(r"\.\.\.|…", quote)) if p]
    return bool(parts) and all(len(p.split()) >= 4 and f" {p} " in haystack for p in parts)


def verify(drafts: list[ClaimDraft], sources: list[Source]) -> list[Claim]:
    """Keep only quotes found verbatim in a real cited source; contested when a counter-quote verifies too."""
    by_id = {s.id: s for s in sources}

    def check(quotes: list[Quote], issues: list[str]) -> list[Quote]:
        good = []
        for q in quotes:
            if q.source_id not in by_id:
                issues.append(f"invalid reference {q.source_id}")
            elif not quote_in_source(q.quote, by_id[q.source_id].text):
                issues.append(f"quote not found in {q.source_id}")
            else:
                good.append(q)
        return good

    claims, seen = [], set()
    for d in drafts:
        if (key := " ".join(sorted(set(terms(d.text))))) in seen:
            continue
        seen.add(key)
        issues: list[str] = [] if d.evidence else ["no citation"]
        evidence = check(d.evidence, issues)
        counter = [q for q in check(d.counter, []) if q.source_id not in {e.source_id for e in evidence}]
        status = "not_established" if not evidence else "contested" if counter else "established"
        claims.append(Claim(text=d.text, evidence=evidence or d.evidence, counter=counter, status=status, issues=issues))
    return claims


def citation_accuracy(drafts: list[ClaimDraft], sources: list[Source]) -> float:
    """Share of all quotes the analyst produced that exist verbatim in the source they cite."""
    by_id = {s.id: s for s in sources}
    quotes = [q for d in drafts for q in d.evidence]
    ok = sum(q.source_id in by_id and quote_in_source(q.quote, by_id[q.source_id].text) for q in quotes)
    return ok / len(quotes) if quotes else 0.0


# ---------------------------------------------------------------- consensus + trust
def tally(votes: list[Vote], n_claims: int) -> list[tuple[float, str] | None]:
    """Per claim: (share of judges supporting, majority verdict; ties are 'unclear'); None if nobody voted."""
    out: list[tuple[float, str] | None] = []
    for i in range(n_claims):
        verdicts = [v.verdict for v in votes if v.claim == i]
        counts = Counter(verdicts).most_common()
        if not counts:
            out.append(None)
            continue
        majority = "unclear" if len(counts) > 1 and counts[0][1] == counts[1][1] else counts[0][0]
        out.append((verdicts.count("supported") / len(verdicts), majority))
    return out


def apply_consensus(claims: list[Claim], votes: list[Vote]) -> list[Claim]:
    """A claim the judges' majority calls contradicted leaves the answer for 'Not established'."""
    out = []
    for claim, score in zip(claims, tally(votes, len(claims)), strict=True):
        change: dict[str, object] = {} if score is None else {"support": score[0], "majority": score[1]}
        if score and score[1] == "contradicted" and claim.status != "not_established":
            change |= {"status": "not_established", "issues": [*claim.issues, "consensus panel: contradicted"]}
        out.append(claim.model_copy(update=change))
    return out


def answer_claims(claims: list[Claim]) -> list[Claim]:
    return [c for c in claims if c.status != "not_established"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def credibility(s: Source) -> float:
    """0-1: identifier/peer review, recency, provider; .gov/.edu get a small boost."""
    ident = 1.0 if s.doi or s.arxiv_id or s.provider == "arxiv" else 0.6 if s.provider == "pdf" else 0.3
    year = int(s.published[:4]) if s.published and s.published[:4].isdigit() else None
    recency = 0.5 if year is None else max(0.0, 1 - (date.today().year - year) / 6)
    provider = {"arxiv": 0.9, "pdf": 0.8, "tavily": 0.6, "parallel": 0.6}[s.provider]
    bonus = 0.1 if re.search(r"\.(gov|edu|ac\.\w+)(/|$)", urlsplit(s.url).netloc + "/") else 0.0
    return round(min(1.0, 0.45 * ident + 0.25 * recency + 0.3 * provider + bonus), 3)


def trust(claims: list[Claim], drafts: list[ClaimDraft], sources: list[Source], relevance: float) -> Trust:
    answer = answer_claims(claims)
    cited = {sid for c in answer for sid in c.citations}
    return Trust(
        faithfulness=mean([float(c.entailed) for c in answer if c.entailed is not None]),
        answer_relevance=relevance, citation_accuracy=citation_accuracy(drafts, sources),
        consensus=mean([c.support for c in answer if c.support is not None]),
        credibility=mean([credibility(s) for s in sources if s.id in cited]),
    )


# ---------------------------------------------------------------- metrics + report
def usage_cost(u: TokenUsage, prices: dict[str, tuple[float, float]]) -> float:
    """Dollar cost of one LLM call at the live OpenRouter prices (0 for a model without a price)."""
    price_in, price_out = prices.get(u.model, (0.0, 0.0))
    return u.input_tokens * price_in + u.output_tokens * price_out


def run_cost(usage: list[TokenUsage], prices: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Dollar cost per model."""
    cost: dict[str, float] = {}
    for u in usage:
        if u.model in prices:
            cost[u.model] = cost.get(u.model, 0.0) + usage_cost(u, prices)
    return cost


def _summary_lines(summary: Summary | None, scores: Trust | None) -> list[str]:
    out: list[str] = []
    if summary and summary.premise_verdict in ("unsupported", "false"):
        out += [f"> **Premise check: {summary.premise_verdict}.** {summary.premise_note}", ""]
    if summary:
        out += [f"**Bottom line:** {summary.bottom_line}", "", summary.executive_summary, ""]
    if scores:
        out += [" · ".join(f"{k.replace('_', ' ')}: {v:.0%}" for k, v in scores.model_dump().items()), ""]
    return out


def _finding_lines(answer: list[Claim]) -> list[str]:
    out = [f"## Findings ({len(answer)} quoted and verified)", ""]
    for i, c in enumerate(answer, 1):
        agree = f" · judges supporting: {c.support:.0%}" if c.support is not None else ""
        cites = " ".join(f"[{s}]" for s in c.citations)
        out.append(f"{i}. **{'Contested' if c.status == 'contested' else 'Established'}** — {c.text} {cites}{agree}")
        out += [f"   - [{q.source_id}] “{q.quote.strip()}”" for q in c.evidence]
        out += [f"   - Counter [{q.source_id}]: “{q.quote.strip()}”" for q in c.counter]
    return out if answer else [*out, "_No claim survived quote verification._"]


def _section(title: str, lines: list[str]) -> list[str]:
    return ["", f"## {title}", "", *lines] if lines else []


def reference_line(s: Source) -> str:
    """One bibliography entry: authors; date; DOI or arXiv id; page; provider; credibility."""
    ident = f"doi:{s.doi}" if s.doi else f"arXiv:{s.arxiv_id}" if s.arxiv_id else ""
    authors = ", ".join(s.authors[:3]) + (" et al." if len(s.authors) > 3 else "")
    meta = "; ".join(filter(None, [authors, s.published, ident, f"p.{s.page}" if s.page else "", s.provider,
                                   f"credibility {credibility(s):.2f}"]))
    return f"- **[{s.id}]** [{s.title}]({s.url}) — {meta}"


def render_report(query: str, summary: Summary | None, claims: list[Claim], hypotheses: list[Hypothesis], gaps: list[str],
                  sources: list[Source], memory: list[StoredClaim], scores: Trust | None) -> str:
    """Answer first, then quoted evidence. 'Not established' claims are listed but never part of the answer."""
    rejected = [f"- {c.text} — {'; '.join(c.issues)}" for c in claims if c.status == "not_established"]
    out = [f"# {query}", "", *_summary_lines(summary, scores), *_finding_lines(answer_claims(claims))]
    out += _section("Already in your library", [f"- {m.text} ({m.date}, from: {m.question})" for m in memory])
    out += _section("Hypotheses and trends (speculative, not verified findings)", [f"- _{h.kind.title()}:_ {h.statement}" for h in hypotheses])
    out += _section("Not established", rejected + [f"- Open question: {g}" for g in gaps])
    out += ["", "## References", "", *(reference_line(s) for s in sources)]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- latency + run history
LATENCY_AGENTS = {"Planner": "Planner", "Retriever": "Contextual Retriever", "Analyst": "Critical Analyst", "Verifier": "Verifier",
                  "Reporter": "Report Builder"}


def agent_latency(trace: list[TraceEvent]) -> dict[str, dict[str, float]]:
    """Average, count and total milliseconds per research agent (parallel branches count as separate calls)."""
    out = {}
    for label, agent in LATENCY_AGENTS.items():
        ms = [e.ms for e in trace if e.agent == agent]
        out[label] = {"avg_ms": round(sum(ms) / len(ms), 1) if ms else 0.0, "calls": float(len(ms)), "total_ms": float(sum(ms))}
    return out


def history_average(records: list[dict[str, Any]]) -> tuple[dict[str, float], int]:
    """Mean per-agent latency over past full runs (cache hits skipped, they run no agents)."""
    runs = [r["latency"] for r in records if not r.get("cache_hit") and r.get("latency")]
    return {k: round(mean([r[k] for r in runs if r.get(k)]), 1) for k in LATENCY_AGENTS}, len(runs)


def run_record(res: dict[str, Any], prices: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """One row of research history."""
    answer, scores, usage = answer_claims(res.get("claims", [])), res.get("trust"), [] if res.get("cache_hit") else res.get("usage", [])
    return {
        "date": datetime.now().isoformat(timespec="seconds"), "question": res.get("query", ""), "seconds": round(res.get("wall_s", 0.0), 1),
        "cache_hit": bool(res.get("cache_hit")), "findings": len(answer), "sources_cited": len({s for c in answer for s in c.citations}),
        "consensus": round(scores.consensus, 3) if scores else None, "faithfulness": round(scores.faithfulness, 3) if scores else None,
        "citation_accuracy": round(scores.citation_accuracy, 3) if scores and not res.get("cache_hit") else None,
        "cost_usd": round(sum(run_cost(usage, prices).values()), 6), "loops": res.get("loops", 0),
        "latency": {} if res.get("cache_hit") else {k: v["avg_ms"] for k, v in agent_latency(res.get("trace", [])).items()},
    }
