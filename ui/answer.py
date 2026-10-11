"""The answer, chat style: the question as a user bubble, the answer in an assistant bubble with metrics, the trust panel
and one bordered card per quoted finding (support badge, status pill, quotes linked to their sources)."""

from __future__ import annotations

from typing import Any

import streamlit as st

import evidence as ev
from models import Claim, ClaimDraft, Quote, Source, Summary, Trust, Vote
from ui.common import Result, md
from ui.session import catalogue, run, settings

FAKES = [  # the "Inject fabricated citation" demo: one invented source id, one invented quote in a real source
    ("Quantum computers already train GPT-class models 1000x faster.", "S99", "Quantum computers already train GPT models 1000x faster."),
    ("This source proves quantum LLMs run in production today.", "", "Quantum LLMs are deployed in production at every major lab today."),
]


# ---------------------------------------------------------------- pure helpers (unit-tested)
def badges(c: Claim) -> list[str]:
    """Status pill, judges' support and the entailment flag as Streamlit Markdown badges."""
    out = [":orange-badge[Contested]" if c.status == "contested" else ":green-badge[Established]"]
    s = c.support
    if s is None:
        out.append(":gray-badge[not judged]")
    else:
        color = "green" if s >= 0.67 else "orange" if s >= 0.34 else "red"
        out.append(f":{color}-badge[{s:.0%} of judges support]")
    if c.entailed is False:
        out.append(":red-badge[not entailed by its quotes]")
    return out


def quote_md(q: Quote, urls: dict[str, str], counter: bool = False) -> str:
    """A verified quote as a blockquote, ending with its source id linked to the source."""
    sid = f"[{q.source_id}]({urls[q.source_id]})" if urls.get(q.source_id) else f"\\[{q.source_id}\\]"
    return f"> {'**Counter:** ' if counter else ''}“{md(q.quote)}” — {sid}"


def metric_tiles(res: Result, prices: dict[str, tuple[float, float]]) -> list[tuple[str, Any]]:
    answer = ev.answer_claims(res.get("claims", []))
    cost = 0.0 if res.get("cache_hit") else sum(ev.run_cost(res.get("usage", []), prices).values())
    return [("Quoted findings", len(answer)), ("Sources cited", len({s for c in answer for s in c.citations})),
            ("Runtime", f"{res['wall_s']:.1f}s"), ("Cost", f"${cost:.4f}"), ("Re-search loops", res.get("loops", 0))]


# ---------------------------------------------------------------- rendering
def user_bubble(question: str) -> None:
    with st.chat_message("user"):
        st.markdown(md(question))


def cache_banner(cfg: dict[str, Any], res: Result) -> None:
    if not res.get("cache_hit"):
        return
    st.info("Answered instantly from your second brain (semantic cache). Nothing was searched or paid for.")
    if st.button("Refresh with a new search", key="refresh"):
        run(cfg, {**settings(cfg), "query": res["query"], "force_refresh": True})
        st.rerun()


def summary_block(summary: Summary | None) -> None:
    if summary is None:
        return
    if summary.premise_verdict in ("unsupported", "false"):
        st.error(f"**Premise check: {summary.premise_verdict}.** {summary.premise_note}")
    st.success(f"**Bottom line:** {summary.bottom_line}")
    st.write(summary.executive_summary)


def tiles(items: list[tuple[str, Any]]) -> None:
    for col, (label, value) in zip(st.columns(len(items)), items, strict=True):
        col.metric(label, value)


def trust_panel(trust: Trust | None) -> None:
    if trust:
        st.caption("Trust panel")
        tiles([(k.replace("_", " ").title(), f"{v:.0%}") for k, v in trust.model_dump().items()])


def library_matches(res: Result) -> None:
    for url in res.get("trace_urls", []):
        st.markdown(f"[Open the {'Langfuse' if 'langfuse' in url else 'LangSmith'} trace]({url})")
    if mem := res.get("memory_claims"):
        with st.expander(f"Already in your library: {len(mem)} verified claims related to this question"):
            for c in mem:
                st.markdown(f"- {md(c.text)} _(from “{md(c.question)}”, {c.date}, similarity {c.similarity:.2f})_")


def finding_card(i: int, c: Claim, votes: list[Vote], urls: dict[str, str]) -> None:
    """C<i> matches the claim labels in the Evidence and Consensus charts."""
    with st.container(border=True):
        st.markdown(f"**C{i + 1}.** {md(c.text)}")
        st.markdown(" ".join(badges(c)))
        for q in c.evidence:
            st.markdown(quote_md(q, urls))
        for q in c.counter:
            st.markdown(quote_md(q, urls, counter=True))
        if mine := [v for v in votes if v.claim == i]:
            st.caption("Judges: " + " · ".join(f"{v.model.split('/')[-1]}: {v.verdict}" for v in mine))


def findings(res: Result) -> None:
    claims, votes = res.get("claims", []), res.get("votes", [])
    urls = {s.id: s.url for s in res.get("ranked", [])}
    st.markdown(f"##### Findings ({len(ev.answer_claims(claims))} quoted and verified)")
    for i, c in enumerate(claims):
        if c.status != "not_established":
            finding_card(i, c, votes, urls)
    if hyps := res.get("hypotheses"):
        st.markdown("##### Hypotheses and trends")
        st.info("Speculative: generated from the findings above, not verified by any source.")
        for h in hyps:
            st.markdown(f"- _{h.kind.title()}_: {md(h.statement)}")
    if rejected := [c for c in claims if c.status == "not_established"]:
        st.markdown(f"##### Not established ({len(rejected)})")
        for c in rejected:
            st.warning(f"{c.text} ({'; '.join(c.issues)})")


def answer_view(cfg: dict[str, Any], res: Result) -> None:
    """Everything inside the assistant bubble."""
    cache_banner(cfg, res)
    summary_block(res.get("summary"))
    tiles(metric_tiles(res, catalogue()))
    trust_panel(res.get("trust"))
    library_matches(res)
    findings(res)


def fabricated_demo(ranked: list[Source]) -> None:
    """Show the verifier catching an invented source id and an invented quote."""
    if st.button("Inject fabricated citation", key="fabricated"):
        real = ranked[0].id if ranked else "S1"
        fakes = [ClaimDraft(text=text, evidence=[Quote(source_id=sid or real, quote=quote)]) for text, sid, quote in FAKES]
        for c in ev.verify(fakes, ranked):
            st.error(f"Caught by the auditor: {c.status.upper()} — {c.text} ({'; '.join(c.issues)})")
