"""Agent activity as a story: one line per step with the agent's icon, the exact search queries and short result previews,
streamed live while a run works in the background (plus the answer itself while the Report Builder writes it), then kept
as a collapsed log and an end-of-run execution summary."""

from __future__ import annotations

from typing import Any

import streamlit as st

import evidence as ev
from models import TokenUsage, TraceEvent
from ui.common import Result, icon, link, md
from ui.session import current_job

VERBS = {  # what each graph step did, in plain words
    "recall": "checked your second brain", "semantic cache hit": "found this question already answered",
    "route": "picked a model for each step", "plan": "split the question into sub-questions", "search": "searched",
    "hybrid retrieval + quoted claims": "read the evidence and extracted quoted claims", "no evidence": "found no evidence to read",
    "verify quotes + citations": "checked every quote against its cited source", "finish": "closed the research loop",
    "cross-model judging": "cross-examined the findings with independent judges", "nothing to judge": "had no findings to judge",
    "hypotheses": "proposed forward-looking hypotheses", "no findings": "had no findings to build on",
    "faithfulness + relevance": "scored faithfulness and answer relevance", "nothing to score": "had no findings to score",
    "answer, report + archive to memory": "wrote the answer and saved it to memory", "no results": "found nothing for this search",
}


def verb(action: str) -> str:
    """Plain-words description of a graph step's action for the activity story."""
    if action.startswith("loop "):
        return f"sent the retrievers back for the open gaps ({action.split(':')[0]})"
    return VERBS.get(action, action)


def narrate(e: TraceEvent | str) -> tuple[str, list[str]]:
    """One activity step as (Markdown headline, preview lines). Plain strings are notices from the job runner."""
    if isinstance(e, str):
        return f"⏳ {md(e)}", []
    failed = e.action.startswith("failed")
    secs = f" · `{e.ms / 1000:.1f}s`"
    query, tool = getattr(e, "query", ""), getattr(e, "tool", "")  # events from an older build may lack these
    if query:
        head = f"{icon(e.agent)} **{e.agent}** searched {tool} for *“{md(query)}”*: {md(e.detail)}{secs}"
    else:
        what = f"could not finish its {md(e.action.removeprefix('failed: '))}" if failed else verb(e.action)
        head = f"{'⚠️' if failed else icon(e.agent)} **{e.agent}** {what}" + (f": {md(e.detail)}" if e.detail else "") + secs
    previews = [f"📄 {s.provider.title()} · {link(s.title, s.url)}: {md(s.text)}" for s in getattr(e, "snippets", [])]
    return head, previews


def story(events: list[Any]) -> None:
    """Render activity steps as headlines with their result previews."""
    for e in events:
        head, previews = narrate(e)
        st.markdown(head)
        for line in previews:
            st.caption(line)


def draft_md(text: str) -> list[str]:
    """The answer being written, as escaped Markdown paragraphs: the bottom line in bold, a cursor after the last word."""
    paras = [md(p) for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []
    paras[0] = f"**Bottom line:** {paras[0]}"
    paras[-1] += " ▌"
    return paras


def writing_block(text: str) -> None:
    """'Writing the answer…': the Report Builder's streamed text so far (it grows on every poll)."""
    if paras := draft_md(text):
        with st.container(border=True):
            st.caption(f"{icon('Report Builder')} Writing the answer… (the verified findings, report and trust panel follow when it is done)")
            for p in paras:
                st.markdown(p)


@st.fragment(run_every=1.0)
def live_status() -> None:
    """Polls the running job once a second, streams agent activity and the answer as it is written; redraws the whole
    page when the run finishes (the full answer view then replaces the draft)."""
    job = current_job()
    if job is None or not job.running:
        st.rerun()
        return
    events, partial = list(job.events), job.partial
    steps = sum(isinstance(e, TraceEvent) for e in events)
    label = "Writing the answer" if partial else "Agents working"
    with st.status(f"{label}... {job.elapsed:.0f}s · {steps} steps so far", expanded=not partial):
        story(events)
    writing_block(partial)


def last_status() -> None:
    """The finished run's activity, collapsed under its outcome."""
    if last := st.session_state.get("last_run"):
        label, state, events = last
        with st.status(label, state=state, expanded=False):
            story(events)


# ---------------------------------------------------------------- execution summary (pure rows, unit-tested)
def searches(queries: list[str], shown: int = 3) -> str:
    """'N searches: “q1”, “q2”, …' with at most ``shown`` queries listed."""
    listed = ", ".join(f"“{ev.shorten(q, 48)}”" for q in queries[:shown])
    return f"{len(queries)} searches: {listed}" + (f" and {len(queries) - shown} more" if len(queries) > shown else "")


def highlight(events: list[TraceEvent], llm_calls: int) -> str:
    """What one agent did on this run, in a sentence."""
    if queries := [q for e in events if (q := getattr(e, "query", ""))]:
        return searches(queries)
    if not events:
        return f"{llm_calls} LLM call(s)"
    ok = [e for e in events if not e.action.startswith("failed")]
    last = (ok or events)[-1]
    text = ev.shorten(verb(last.action) + (f" ({last.detail})" if last.detail else ""), 140)
    failures = len(events) - len(ok)
    return text + (f"; {failures} step(s) failed and the run continued" if failures else "")


def execution_rows(res: Result, prices: dict[str, tuple[float, float]]) -> list[dict[str, Any]]:
    """Per agent, in the order they first acted: what it did, steps, agent time, tokens and cost (nothing is spent on a cache hit)."""
    trace, usage = res.get("trace", []), [] if res.get("cache_hit") else res.get("usage", [])
    agents = dict.fromkeys([e.agent for e in trace] + [u.agent for u in usage])
    return [agent_row(a, [e for e in trace if e.agent == a], [u for u in usage if u.agent == a], prices) for a in agents]


def agent_row(agent: str, events: list[TraceEvent], calls: list[TokenUsage], prices: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """One execution-summary row: what the agent did, steps, agent time, tokens and cost."""
    return {"agent": f"{icon(agent)} {agent}", "what it did": highlight(events, len(calls)), "steps": len(events),
            "time (s)": round(sum(e.ms for e in events) / 1000, 1), "tokens in": sum(u.input_tokens for u in calls),
            "tokens out": sum(u.output_tokens for u in calls), "cost ($)": round(sum(ev.usage_cost(u, prices) for u in calls), 5)}


def execution_summary(res: Result, prices: dict[str, tuple[float, float]]) -> None:
    """The end-of-run execution summary card: totals plus one row per agent."""
    rows = execution_rows(res, prices)
    tokens, cost = sum(r["tokens in"] + r["tokens out"] for r in rows), sum(r["cost ($)"] for r in rows)
    with st.container(border=True):
        st.markdown("#### Execution summary")
        if res.get("cache_hit"):
            st.caption("Answered from memory: no agent searched, read or spent tokens on this run.")
        st.caption(f"{len(rows)} agents · {sum(r['steps'] for r in rows)} steps · {tokens:,} tokens · ${cost:.4f} · "
                   f"{res.get('wall_s', 0.0):.1f}s wall clock (parallel agents overlap, so agent times add up to more)")
        st.table([r | {"time (s)": f"{r['time (s)']:.1f}", "tokens in": f"{r['tokens in']:,}", "tokens out": f"{r['tokens out']:,}",
                       "cost ($)": f"${r['cost ($)']:.5f}"} for r in rows])  # a table wraps the "what it did" sentences
