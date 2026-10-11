"""The seven detail tabs under an answer: agent activity, evidence, dashboard, consensus, router and cost, sources, report."""

from __future__ import annotations

from typing import Any

import streamlit as st
import streamlit.components.v1 as components

import evidence as ev
from ui import charts
from ui.common import Result
from ui.session import brain, catalogue

TABS = ["Agent Activity", "Evidence", "Dashboard", "Consensus", "Router & cost", "Sources", "Report"]


def embed_html(page: str, height: int) -> None:
    """A self-contained chart page: st.iframe on current Streamlit (components.html is deprecated), components.html before it."""
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:
        components.html(page, height=height)


def activity_tab(res: Result) -> None:
    st.plotly_chart(charts.fig_workflow(res.get("edges", []), res), width="stretch", key="fig_workflow")
    history = brain().history()
    current, (average, runs) = ev.agent_latency(res.get("trace", [])), ev.history_average(history)
    embed_html(charts.latency_chart_html(current, average, runs), 360)
    embed_html(charts.accuracy_chart_html(history), 360)
    rows, bottleneck = charts.latency_rows(current, average)
    if bottleneck:
        slow = next(r for r in rows if r["agent"] == bottleneck)
        st.caption(f"Bottleneck: **{bottleneck}** averages {slow['run']:.1f}s per call over {slow['calls']} call(s) "
                   f"({slow['total']:.1f}s total). Parallel branches overlap, so check the timeline below for what is on the critical path.")
    st.plotly_chart(charts.fig_gantt(res), width="stretch", key="fig_gantt")


def evidence_tab(res: Result) -> None:
    claims, ranked = res.get("claims", []), res.get("ranked", [])
    st.plotly_chart(charts.fig_evidence_graph(claims, ranked), width="stretch", key="fig_graph")
    st.plotly_chart(charts.fig_citations(claims, ranked), width="stretch", key="fig_citations")


def dashboard_tab(res: Result) -> None:
    claims, ranked = res.get("claims", []), res.get("ranked", [])
    left, right = st.columns(2)
    left.plotly_chart(charts.fig_source_mix(ranked), width="stretch", key="fig_mix")
    right.plotly_chart(charts.fig_confidence(claims), width="stretch", key="fig_conf")
    left.plotly_chart(charts.fig_funnel(res), width="stretch", key="fig_funnel")
    right.plotly_chart(charts.fig_dates(ranked), width="stretch", key="fig_dates")
    st.caption(f"Duplicates removed (identity + embedding near-duplicates): {res.get('duplicates_removed', 0)}")


def consensus_tab(res: Result) -> None:
    votes = res.get("votes", [])
    st.plotly_chart(charts.fig_heatmap(votes), width="stretch", key="fig_heat")
    st.dataframe([v.model_dump() | {"claim": f"C{v.claim + 1}"} for v in votes], hide_index=True, key="votes_table")


def cost_rows(res: Result, prices: dict[str, tuple[float, float]]) -> list[dict[str, Any]]:
    """Per model used on this run: live price per million tokens and what this run spent."""
    cost = ev.run_cost(res.get("usage", []), prices)
    return [{"model": mo, "$/1M in": prices[mo][0] * 1e6, "$/1M out": prices[mo][1] * 1e6, "this run $": round(cost.get(mo, 0), 5)}
            for mo in dict.fromkeys(u.model for u in res.get("usage", [])) if mo in prices]


def router_tab(res: Result) -> None:
    route = res["route"]
    judges = ", ".join(f"`{j}`" for j in route.judges)
    st.markdown(f"**Extraction** `{route.extract_model}` · **Synthesis** `{route.synth_model}` · **Judges** {judges}")
    st.markdown("\n".join(f"- {n}" for n in route.notes))
    if res.get("profile"):
        st.json(res["profile"].model_dump())
    st.dataframe(cost_rows(res, catalogue()), hide_index=True, key="cost_table")


def sources_tab(res: Result) -> None:
    st.dataframe([{"id": s.id, "provider": s.provider, "credibility": ev.credibility(s), "memory": s.from_memory, "title": s.title,
                   "published": s.published, "page": s.page, "url": s.url} for s in res.get("ranked", [])], hide_index=True, key="sources_table")
    st.dataframe([c.model_dump() for c in res.get("tool_calls", [])], hide_index=True, key="calls_table")


def detail_tabs(res: Result) -> None:
    renderers = [activity_tab, evidence_tab, dashboard_tab, consensus_tab, router_tab, sources_tab,
                 lambda r: st.markdown(r.get("report_md", ""))]
    for tab, render in zip(st.tabs(TABS), renderers, strict=True):
        with tab:
            render(res)
