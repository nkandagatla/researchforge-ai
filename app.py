"""ResearchForge AI v2: research you can audit, and that remembers."""

from __future__ import annotations

import asyncio
import html
import json
import math
import os
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import evidence as ev
import memory
from agents import JUDGE_POOL, STATE, Keys, research
from models import Claim, ClaimDraft, Quote, Source, Vote
from tools import openrouter_models, pdf_sources, redact

load_dotenv()
QUESTION = "How is quantum computing going to change the current AI LLM/models and agentic implementations?"
FALSE_PREMISE = "Why have quantum computers already replaced GPUs for training LLMs?"
VERDICT_COLOR = {"supported": "#16a34a", "contradicted": "#dc2626", "unclear": "#94a3b8"}
PROVIDER_COLOR = {"parallel": "#6366f1", "tavily": "#f59e0b", "arxiv": "#b91c1c", "pdf": "#0ea5e9"}
NODE_POS = {  # layered layout of the compiled LangGraph
    "__start__": (0, 0), "recall": (1, 0), "router": (2, 1.3), "planner": (2, -1.3), "approve": (3, -1.3), "retriever": (4, -1.3),
    "analyst": (5, 0), "verifier": (6, 0), "supervisor": (7, 0), "gap_retriever": (6, 1.6), "consensus": (8, 1.4),
    "insight": (8, 0), "trust": (8, -1.4), "report": (9, 0), "__end__": (10, 0),
}
AGENTS = {"recall": "Memory", "router": "Topic Router", "planner": "Planner", "approve": "Plan approval", "retriever": "Contextual Retriever",
          "gap_retriever": "Gap retriever", "analyst": "Critical Analyst", "verifier": "Verifier", "supervisor": "Supervisor",
          "consensus": "Consensus Panel", "insight": "Insight Generator", "trust": "Trust Judge", "report": "Report Builder"}
Result = dict[str, Any]
RECHARTS = [  # pinned UMD builds, loaded in order: Recharts needs window.React + window.PropTypes
    "https://cdn.jsdelivr.net/npm/react@18.3.1/umd/react.production.min.js",
    "https://cdn.jsdelivr.net/npm/react-dom@18.3.1/umd/react-dom.production.min.js",
    "https://cdn.jsdelivr.net/npm/prop-types@15.8.1/prop-types.min.js",
    "https://cdn.jsdelivr.net/npm/recharts@2.15.4/umd/Recharts.js",
]
RECHARTS_SHELL = """<!doctype html><html><head><meta charset="utf-8"><style>
.viz-root{color-scheme:light;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--hover:rgba(11,11,11,.05);--s1:#2a78d6;--s2:#eb6834;--ring:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--hover:rgba(255,255,255,.06);--s1:#3987e5;--s2:#d95926;--ring:rgba(255,255,255,.10)}}
body{margin:0;background:transparent;font:13px system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{background:var(--surface);color:var(--text);border:1px solid var(--ring);border-radius:8px;padding:12px 8px 4px}
h4{margin:0 0 2px 8px;font-size:14px}.sub{margin:0 0 6px 8px;color:var(--text2);font-size:12px}
.tip{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:8px 10px;box-shadow:0 2px 8px rgba(0,0,0,.12);max-width:320px}
.muted{color:var(--text2);margin-top:4px}</style>__SCRIPTS__</head><body><div class="viz-root"><h4>__TITLE__</h4>
<p class="sub" id="sub"></p><div id="chart"></div></div><script>
const D = __DATA__;
document.getElementById("sub").textContent = D.subtitle;
const el = document.getElementById("chart");
if (!window.Recharts || !window.ReactDOM) {
  el.textContent = "The chart library could not load (offline?). The same numbers are in the tables on this page.";
} else {
  const R = window.Recharts, h = React.createElement, css = getComputedStyle(el.parentElement), v = n => css.getPropertyValue(n).trim();
  const axis = {tick: {fill: v("--muted"), fontSize: 12}, axisLine: false, tickLine: false};
  ReactDOM.createRoot(el).render(h(R.ResponsiveContainer, {width: "100%", height: 290}, (() => {
__CHART__
  })()));
}
</script></body></html>"""
LATENCY_CHART = """
  const tip = ({active, payload, label}) => !(active && payload && payload.length) ? null : h("div", {className: "tip"},
    h("b", null, label), ...payload.map(p => h("div", {key: p.dataKey}, `${p.name}: ${p.value.toFixed(2)} s`)),
    h("div", {className: "muted"}, `${payload[0].payload.calls} call(s) · ${payload[0].payload.total.toFixed(1)} s total this run`));
  const flag = ({x, y, width, value}) => value !== D.bottleneck ? null :
    h("text", {x: x + width / 2, y: y - 8, textAnchor: "middle", fill: v("--text"), fontSize: 11, fontWeight: 600}, "▲ bottleneck");
  return h(R.BarChart, {data: D.rows, margin: {top: 22, right: 12, left: 0, bottom: 0}, barGap: 2, barCategoryGap: "28%"},
    h(R.CartesianGrid, {vertical: false, stroke: v("--grid")}),
    h(R.XAxis, {...axis, dataKey: "agent", axisLine: {stroke: v("--axis")}}),
    h(R.YAxis, {...axis, unit: " s", width: 48}),
    h(R.Tooltip, {content: tip, cursor: {fill: v("--hover")}}),
    h(R.Legend, {iconType: "circle", wrapperStyle: {fontSize: 12}, formatter: t => h("span", {style: {color: v("--text2")}}, t)}),
    h(R.Bar, {dataKey: "run", name: "This run", fill: v("--s1"), radius: [4, 4, 0, 0], isAnimationActive: false},
      h(R.LabelList, {dataKey: "agent", content: flag})),
    h(R.Bar, {dataKey: "avg", name: `Average over ${D.runs} run(s)`, fill: v("--s2"), radius: [4, 4, 0, 0], isAnimationActive: false}));"""
ACCURACY_CHART = """
  const tip = ({active, payload}) => !(active && payload && payload.length) ? null : (p => h("div", {className: "tip"},
    h("b", null, `${p.accuracy.toFixed(1)}% citation accuracy`), h("div", null, p.when),
    h("div", {className: "muted"}, `${p.findings} findings · ${p.question}`)))(payload[0].payload);
  const dot = ({cx, cy, index}) => h("circle", {key: index, cx, cy, r: index === D.rows.length - 1 ? 6 : 4, fill: v("--s1"),
    stroke: v("--surface"), strokeWidth: 2});
  return h(R.LineChart, {data: D.rows, margin: {top: 16, right: 24, left: 0, bottom: 0}},
    h(R.CartesianGrid, {vertical: false, stroke: v("--grid")}),
    h(R.XAxis, {...axis, dataKey: "label", axisLine: {stroke: v("--axis")}, interval: "preserveStartEnd", minTickGap: 24}),
    h(R.YAxis, {...axis, domain: [0, 100], unit: "%", width: 48}),
    h(R.Tooltip, {content: tip, cursor: {stroke: v("--axis"), strokeDasharray: "3 3"}}),
    D.mean === null ? null : h(R.ReferenceLine, {y: D.mean, stroke: v("--muted"), strokeDasharray: "4 4",
      label: {value: `avg ${D.mean.toFixed(0)}%`, position: "insideTopRight", fill: v("--text2"), fontSize: 11}}),
    h(R.Line, {dataKey: "accuracy", name: "Citation accuracy", stroke: v("--s1"), strokeWidth: 2, dot, activeDot: {r: 7},
      isAnimationActive: false}));"""


# ---------------------------------------------------------------- figures (pure functions, unit-tested)
def _layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    return fig.update_layout(title=title, height=height, margin={"l": 10, "r": 10, "t": 40, "b": 10})


def handoffs(res: Result) -> dict[tuple[str, str], str]:
    """What each agent actually handed to the next on this run."""
    claims, plan, trust = res.get("claims", []), res["plan"], res.get("trust")
    answer = len(ev.answer_claims(claims))
    return {
        ("recall", "router"): f"{len(res.get('memory_claims', []))} remembered claims",
        ("planner", "approve"): f"{len(plan.subquestions)} subquestions",
        ("approve", "retriever"): f"{len(plan.subquestions) + len(plan.academic_queries)} branches",
        ("retriever", "analyst"): f"{len(res.get('sources', []))} results", ("router", "analyst"): res["route"].extract_model.split("/")[-1],
        ("analyst", "verifier"): f"{len(res.get('drafts', []))} quoted drafts", ("verifier", "supervisor"): f"{answer} verified",
        ("supervisor", "gap_retriever"): f"{res.get('loops', 0)} loop(s)", ("consensus", "report"): f"{len(res.get('votes', []))} votes",
        ("insight", "report"): f"{len(res.get('hypotheses', []))} hypotheses",
        ("trust", "report"): f"faithfulness {trust.faithfulness:.0%}" if trust else "",
    }


def fig_workflow(edges: list[tuple[str, str]], res: Result) -> go.Figure:
    """Agent-to-agent workflow drawn from the compiled graph's real edges, labelled with each hand-off."""
    labels, secs = handoffs(res), Counter[str]()
    for e in res.get("trace", []):
        secs[e.agent] += e.ms / 1000
    fig = go.Figure()
    for a, b in edges:
        (x0, y0), (x1, y1) = NODE_POS[a], NODE_POS[b]
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", arrowhead=3, arrowwidth=1.3,
                           arrowcolor="#64748b", standoff=16, startstandoff=16)
        if label := labels.get((a, b)):
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.22, text=label, showarrow=False, font={"size": 9})
    names = list(NODE_POS)
    fig.add_trace(go.Scatter(
        x=[NODE_POS[n][0] for n in names], y=[NODE_POS[n][1] for n in names], mode="markers+text", textposition="bottom center",
        text=[AGENTS.get(n, n.strip("_")) for n in names], hoverinfo="text",
        marker={"size": 26, "color": ["#0f172a" if n in AGENTS else "#cbd5e1" for n in names]},
        hovertext=[f"{AGENTS.get(n, n)}: {secs.get(AGENTS.get(n, ''), 0):.1f}s of work" for n in names]))
    fig.update_xaxes(visible=False, range=[-0.5, 10.5])
    fig.update_yaxes(visible=False, range=[-2, 2.1])
    return _layout(fig, "Agent-to-agent workflow (edge labels = what was handed over)", 420).update_layout(showlegend=False)


def fig_evidence_graph(claims: list[Claim], sources: list[Source]) -> go.Figure:
    """Claims (inner ring) to sources (outer ring); edge colour = consensus majority verdict."""
    by_id = {s.id: s for s in sources}
    used = list(dict.fromkeys(sid for c in claims for sid in c.citations if sid in by_id))
    cnames = [f"C{i + 1}" for i in range(len(claims))]

    def ring(names: list[str], r: float) -> dict[str, tuple[float, float]]:
        step = 2 * math.pi / max(len(names), 1)
        return {n: (r * math.cos(i * step), r * math.sin(i * step)) for i, n in enumerate(names)}

    pos, fig = ring(cnames, 0.45) | ring(used, 1.0), go.Figure()
    for verdict, color in VERDICT_COLOR.items():
        xs: list[float | None] = []
        ys: list[float | None] = []
        for name, c in zip(cnames, claims, strict=True):
            for sid in (s for s in c.citations if s in by_id and (c.majority or "unclear") == verdict):
                xs += [pos[name][0], pos[sid][0], None]
                ys += [pos[name][1], pos[sid][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": 2}, name=verdict, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in cnames], y=[pos[n][1] for n in cnames], mode="markers+text", text=cnames, name="claims",
                             marker={"size": 24, "symbol": "diamond",
                                     "color": ["#dc2626" if c.status == "not_established" else "#0f172a" for c in claims]},
                             textfont={"color": "white", "size": 9}, hovertext=[f"{c.status}: {c.text}" for c in claims], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[pos[s][0] for s in used], y=[pos[s][1] for s in used], mode="markers+text", text=used, name="sources",
                             textposition="top center", marker={"size": 18, "color": [PROVIDER_COLOR[by_id[s].provider] for s in used]},
                             hovertext=[f"{by_id[s].title} ({by_id[s].provider}, credibility {ev.credibility(by_id[s]):.2f})" for s in used],
                             hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return _layout(fig, "Evidence graph (edge colour = judges' majority verdict)", 520)


def fig_source_mix(sources: list[Source]) -> go.Figure:
    mix = Counter(s.provider for s in sources)
    pie = go.Pie(labels=list(mix), values=list(mix.values()), hole=0.5, marker={"colors": [PROVIDER_COLOR[p] for p in mix]})
    return _layout(go.Figure(pie), "Source mix")


def fig_confidence(claims: list[Claim]) -> go.Figure:
    colors = ["#dc2626" if c.status == "not_established" else "#f59e0b" if c.status == "contested" else "#16a34a" for c in claims]
    bar = go.Bar(x=[f"C{i + 1}" for i in range(len(claims))], y=[c.support or 0.0 for c in claims], marker_color=colors,
                 hovertext=[f"{c.status}: {c.text}" for c in claims])
    return _layout(go.Figure(bar).update_yaxes(range=[0, 1], tickformat=".0%"), "Claim confidence (share of judges supporting)")


def fig_gantt(res: Result) -> go.Figure:
    trace = res.get("trace", [])
    bar = go.Bar(y=[e.agent for e in trace], x=[max(e.ms, 30) / 1000 for e in trace], base=[e.started for e in trace], orientation="h",
                 hovertext=[f"{e.action}: {e.detail}" for e in trace], marker_color="#6366f1", opacity=0.7)
    return _layout(go.Figure(bar).update_xaxes(title="seconds since start"), "Agent timeline (overlapping bars = parallel work)", 420)


def fig_funnel(res: Result) -> go.Figure:
    raw, sent = (sum(c[i] for c in res.get("context_chars", [])) for i in (0, 1))
    stages = ["Retrieved results", "Unique ranked sources", "Evidence chars (k)", "Chars sent to the LLM (k)"]
    values = [len(res.get("sources", [])), len(res.get("ranked", [])), round(raw / 1000), round(sent / 1000)]
    return _layout(go.Figure(go.Funnel(y=stages, x=values, textinfo="value")), "Context reduction")


def fig_dates(sources: list[Source]) -> go.Figure:
    dated = [s for s in sources if s.published and s.published[:4].isdigit()]
    dots = go.Scatter(x=[(s.published or "")[:10] for s in dated], y=[s.provider for s in dated], mode="markers", hoverinfo="text",
                      marker={"size": 12, "color": [PROVIDER_COLOR[s.provider] for s in dated]}, hovertext=[f"[{s.id}] {s.title}" for s in dated])
    return _layout(go.Figure(dots), f"Publish dates ({len(dated)} of {len(sources)} sources dated)")


def latency_rows(current: dict[str, dict[str, float]], average: dict[str, float]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = [{"agent": k, "run": round(v["avg_ms"] / 1000, 2), "avg": round(average.get(k, 0.0) / 1000, 2),
                                   "calls": int(v["calls"]), "total": round(v["total_ms"] / 1000, 2)} for k, v in current.items()]
    slowest = max(rows, key=lambda r: r["run"]) if rows else None
    return rows, slowest["agent"] if slowest and slowest["run"] > 0 else ""


def recharts_page(title: str, chart_js: str, data: dict[str, Any]) -> str:
    """A self-contained Recharts chart (React UMD from pinned CDN builds) with light/dark tokens and an offline fallback."""
    scripts = "".join(f'<script src="{url}" crossorigin="anonymous"></script>' for url in RECHARTS)
    payload = json.dumps(data).replace("</", "<\\/")
    return (RECHARTS_SHELL.replace("__SCRIPTS__", scripts).replace("__TITLE__", html.escape(title))
            .replace("__CHART__", chart_js).replace("__DATA__", payload))


def latency_chart_html(current: dict[str, dict[str, float]], average: dict[str, float], runs: int) -> str:
    """Bar chart: this run vs the historical average latency per research agent, bottleneck flagged."""
    rows, bottleneck = latency_rows(current, average)
    subtitle = f"Bottleneck this run: {bottleneck}" if bottleneck else "No agent timings yet"
    return recharts_page("Average latency per research agent", LATENCY_CHART, {"rows": rows, "bottleneck": bottleneck, "runs": runs,
                                                                                "subtitle": subtitle})


def accuracy_rows(history: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    """Citation accuracy per full research session (cache hits re-verify nothing), oldest first."""
    runs = [r for r in history if not r.get("cache_hit") and r.get("citation_accuracy") is not None][:limit][::-1]
    return [{"label": datetime.fromisoformat(r["date"]).strftime("%b %d %H:%M"), "when": r["date"].replace("T", " "),
             "accuracy": round(100 * r["citation_accuracy"], 1), "findings": r.get("findings", 0),
             "question": r["question"][:90] + ("..." if len(r["question"]) > 90 else "")} for r in runs]


def accuracy_chart_html(history: list[dict[str, Any]]) -> str:
    """Line chart: citation accuracy across research sessions, with the running mean as a reference line."""
    rows = accuracy_rows(history)
    mean = round(ev.mean([r["accuracy"] for r in rows]), 1) if rows else None
    subtitle = (f"{len(rows)} research sessions · latest {rows[-1]['accuracy']:.0f}% · share of quotes found verbatim in their source"
                if rows else "No research sessions yet")
    return recharts_page("Citation accuracy over time", ACCURACY_CHART, {"rows": rows, "mean": mean, "subtitle": subtitle})


def fig_citations(claims: list[Claim], sources: list[Source]) -> go.Figure:
    """Citation heat map: which source backs (blue) or disputes (red) which claim; column labels count the citations."""
    cites = Counter(q.source_id for c in claims for q in [*c.evidence, *c.counter])
    cols = [s.id for s in sources if s.id in cites]
    cols.sort(key=lambda sid: -cites[sid])
    z = [[1 if sid in c.citations else -1 if sid in {q.source_id for q in c.counter} else 0 for sid in cols] for c in claims]
    words = {1: "quote", -1: "counter", 0: ""}
    heat = go.Heatmap(z=z, x=[f"{sid} ({cites[sid]})" for sid in cols], zmin=-1, zmax=1, showscale=False, xgap=2, ygap=2,
                      y=[f"C{i + 1}" + (" ✗" if c.status == "not_established" else "") for i, c in enumerate(claims)],
                      text=[[words[v] for v in row] for row in z], texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
                      colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]])
    title = "Citation heat map (blue = supporting quote, red = counter-quote; ✗ = not established)"
    fig = _layout(go.Figure(heat).update_yaxes(autorange="reversed"), title, 180 + 30 * len(claims))
    return fig.update_layout(margin={"l": 70, "r": 10, "t": 50, "b": 60}).update_xaxes(title="source (citations)", side="bottom")


def fig_heatmap(votes: list[Vote]) -> go.Figure:
    models, rows = list(dict.fromkeys(v.model for v in votes)), sorted({v.claim for v in votes})
    grid, score = {(v.claim, v.model): v.verdict for v in votes}, {"supported": 1, "unclear": 0, "contradicted": -1}
    heat = go.Heatmap(z=[[score.get(grid.get((r, mo), "")) for mo in models] for r in rows], zmin=-1, zmax=1, showscale=False,
                      x=[mo.split("/")[-1] for mo in models], y=[f"C{r + 1}" for r in rows], texttemplate="%{text}",
                      text=[[grid.get((r, mo), "") for mo in models] for r in rows], colorscale=[[0, "#dc2626"], [0.5, "#94a3b8"], [1, "#16a34a"]])
    return _layout(go.Figure(heat).update_yaxes(autorange="reversed"), "Consensus heatmap (claim x judge)", 140 + 30 * len(rows))


# ---------------------------------------------------------------- app
@st.cache_data(ttl=3600, show_spinner=False)
def catalogue() -> dict[str, tuple[float, float]]:
    return openrouter_models()


@st.cache_resource(show_spinner="Loading memory...")
def brain() -> memory.Memory:
    return memory.Memory()


def key_input(label: str, env: str, required: bool) -> str:
    if value := os.getenv(env, ""):
        st.sidebar.caption(f"{label}: loaded from .env")
        return value
    value = st.sidebar.text_input(f"{label} ({'required' if required else 'optional'})", type="password", key=f"key_{env}")
    if required and not value:
        st.sidebar.warning(f"Add {env} to .env or enter it here.")
    return value


def sidebar(models: dict[str, tuple[float, float]]) -> dict[str, Any]:
    st.sidebar.header("Settings")
    keys = Keys(key_input("OpenRouter key", "OPENROUTER_API_KEY", True), key_input("Tavily key", "TAVILY_API_KEY", False),
                key_input("Parallel key", "PARALLEL_API_KEY", False))
    default = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    options = [mo for mo in dict.fromkeys([default, *JUDGE_POOL]) if not models or mo in models]  # never offer a model that 404s
    cfg = {"keys": keys, "default_model": st.sidebar.selectbox("Default (extraction) model", options or [default], key="model"),
           "auto_route": st.sidebar.toggle("Topic Router picks models per step", True, key="auto"),
           "auto_approve": st.sidebar.toggle("Auto-approve the research plan", False, key="approve"),
           "judge_count": st.sidebar.slider("Consensus judges", 2, 3, 3, key="judges"),
           "max_sources": st.sidebar.slider("Max sources", 6, 20, 12, key="max_sources"),
           "max_loops": st.sidebar.slider("Supervisor re-search loops", 0, 2, 1, key="loops")}
    backends = [n for n, k in (("LangSmith", "LANGSMITH_API_KEY"), ("Langfuse", "LANGFUSE_PUBLIC_KEY")) if os.getenv(k)]
    if backends:
        os.environ["RF_TRACING"] = str(st.sidebar.toggle(f"Trace runs ({' + '.join(backends)})", False, key="tracing")).lower()
    live = f"{len(models)} OpenRouter models validated live" if models else "OpenRouter catalogue unreachable"
    st.sidebar.caption(f"{live} · memory: {brain().backend}")
    if st.sidebar.button("New session", key="clear"):
        st.session_state.clear()
        st.rerun()
    return cfg


def run(cfg: dict[str, Any], inputs: dict[str, Any] | None, resume: Any = None) -> None:
    thread = st.session_state.setdefault("thread", str(uuid.uuid4()))
    with st.status("Agents working...", expanded=True) as status:
        try:
            res = asyncio.run(research(cfg["keys"], catalogue(), brain(), inputs, thread, resume,
                                       lambda e: status.write(f"**{e.agent}** · {e.action} ({e.ms / 1000:.1f}s) {e.detail}")))
            st.session_state.res = res
            done = f"Done in {res['wall_s']:.1f}s" + (" (from memory)" if res.get("cache_hit") else "")
            label = "Waiting for plan approval" if res["interrupt"] else done
            status.update(label=label, state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Research failed", state="error")
            st.error(redact(f"{type(exc).__name__}: {exc}", list(cfg["keys"])))


def plan_editor(cfg: dict[str, Any], plan: dict[str, Any]) -> None:
    st.info("Review the research plan before any searching or analysis is spent on it.")
    subs = st.text_area("Subquestions (one per line)", "\n".join(plan["subquestions"]), key="plan_subs")
    arxiv = st.text_area("arXiv queries (one per line)", "\n".join(plan["academic_queries"]), key="plan_arxiv")
    premise = st.text_input("Premise to verify", plan.get("premise", ""), key="plan_premise")
    if st.button("Approve plan and research", type="primary", key="plan_go"):
        lines = [s.strip() for s in subs.splitlines() if s.strip()][:3]
        run(cfg, None, {"subquestions": lines, "academic_queries": [a.strip() for a in arxiv.splitlines() if a.strip()][:3], "premise": premise})
        st.rerun()


def show_claim(i: int, c: Claim, votes: list[Vote]) -> None:
    s = c.support
    badge = "⚪" if s is None else "🟢" if s >= 0.67 else "🟡" if s >= 0.34 else "🔴"
    flags = (f" · {s:.0%} of judges support" if s is not None else "") + (" · ⚠ not entailed" if c.entailed is False else "")
    st.markdown(f"{badge} **{i + 1}. {c.text}** `{'Contested' if c.status == 'contested' else 'Established'}`{flags}")
    for q in c.evidence:
        st.caption(f"[{q.source_id}] “{q.quote.strip()}”")
    for q in c.counter:
        st.caption(f"↯ Counter [{q.source_id}] “{q.quote.strip()}”")
    if mine := [v for v in votes if v.claim == i]:
        st.caption(" · ".join(f"{v.model.split('/')[-1]}: {v.verdict}" for v in mine))


def answer_view(cfg: dict[str, Any], res: Result) -> None:
    claims, votes, summary, trust = res.get("claims", []), res.get("votes", []), res.get("summary"), res.get("trust")
    if res.get("cache_hit"):
        st.info("Answered instantly from your second brain (semantic cache). Nothing was searched or paid for.")
        if st.button("Refresh with a new search", key="refresh"):
            run(cfg, {**settings(cfg), "query": res["query"], "force_refresh": True})
            st.rerun()
    if summary and summary.premise_verdict in ("unsupported", "false"):
        st.error(f"**Premise check: {summary.premise_verdict}.** {summary.premise_note}")
    if summary:
        st.success(f"**Bottom line:** {summary.bottom_line}")
        st.write(summary.executive_summary)
    cost = 0.0 if res.get("cache_hit") else sum(ev.run_cost(res.get("usage", []), catalogue()).values())
    answer = ev.answer_claims(claims)
    cols = st.columns(5)
    tiles = [("Quoted findings", len(answer)), ("Sources cited", len({s for c in answer for s in c.citations})),
             ("Runtime", f"{res['wall_s']:.1f}s"), ("Cost", f"${cost:.4f}"), ("Re-search loops", res.get("loops", 0))]
    for col, (label, value) in zip(cols, tiles, strict=True):
        col.metric(label, value)
    if trust:
        st.caption("Trust panel")
        tcols = st.columns(5)
        for col, (k, v) in zip(tcols, trust.model_dump().items(), strict=True):
            col.metric(k.replace("_", " ").title(), f"{v:.0%}")
    for url in res.get("trace_urls", []):
        st.markdown(f"[Open the {'Langfuse' if 'langfuse' in url else 'LangSmith'} trace]({url})")
    if mem := res.get("memory_claims"):
        with st.expander(f"Already in your library: {len(mem)} verified claims related to this question"):
            for c in mem:
                st.markdown(f"- {c.text} _(from “{c.question}”, {c.date}, similarity {c.similarity:.2f})_")
    st.subheader("Findings")
    for i, c in enumerate(claims):
        if c.status != "not_established":
            show_claim(i, c, votes)
    if hyps := res.get("hypotheses"):
        st.subheader("Hypotheses and trends")
        st.info("Speculative: generated from the findings above, not verified by any source.")
        for h in hyps:
            st.markdown(f"- _{h.kind.title()}_: {h.statement}")
    if rejected := [c for c in claims if c.status == "not_established"]:
        st.subheader(f"Not established ({len(rejected)})")
        for c in rejected:
            st.warning(f"{c.text} ({'; '.join(c.issues)})")
    if st.button("Inject fabricated citation", key="fabricated"):
        ranked = res.get("ranked", [])
        real = ranked[0].id if ranked else "S1"
        fakes = [ClaimDraft(text="Quantum computers already train GPT-class models 1000x faster.",
                            evidence=[Quote(source_id="S99", quote="Quantum computers already train GPT models 1000x faster.")]),
                 ClaimDraft(text="This source proves quantum LLMs run in production today.",
                            evidence=[Quote(source_id=real, quote="Quantum LLMs are deployed in production at every major lab today.")])]
        for c in ev.verify(fakes, ranked):
            st.error(f"Caught by the auditor: {c.status.upper()} — {c.text} ({'; '.join(c.issues)})")
    st.download_button("Download report (.md)", res.get("report_md", ""), "research_report.md", key="download")


def detail_tabs(res: Result) -> None:
    claims, ranked, votes, route, prices = res.get("claims", []), res.get("ranked", []), res.get("votes", []), res["route"], catalogue()
    tabs = st.tabs(["Agent Activity", "Evidence", "Dashboard", "Consensus", "Router & cost", "Sources", "Report"])
    tabs[0].plotly_chart(fig_workflow(res.get("edges", []), res), width="stretch", key="fig_workflow")
    with tabs[0]:
        current, (average, runs) = ev.agent_latency(res.get("trace", [])), ev.history_average(brain().history())
        components.html(latency_chart_html(current, average, runs), height=360)
        components.html(accuracy_chart_html(brain().history()), height=360)
        rows, bottleneck = latency_rows(current, average)
        if bottleneck:
            slow = next(r for r in rows if r["agent"] == bottleneck)
            st.caption(f"Bottleneck: **{bottleneck}** averages {slow['run']:.1f}s per call over {slow['calls']} call(s) "
                       f"({slow['total']:.1f}s total). Parallel branches overlap, so check the timeline below for what is on the critical path.")
    tabs[0].plotly_chart(fig_gantt(res), width="stretch", key="fig_gantt")
    tabs[1].plotly_chart(fig_evidence_graph(claims, ranked), width="stretch", key="fig_graph")
    tabs[1].plotly_chart(fig_citations(claims, ranked), width="stretch", key="fig_citations")
    left, right = tabs[2].columns(2)
    left.plotly_chart(fig_source_mix(ranked), width="stretch", key="fig_mix")
    right.plotly_chart(fig_confidence(claims), width="stretch", key="fig_conf")
    left.plotly_chart(fig_funnel(res), width="stretch", key="fig_funnel")
    right.plotly_chart(fig_dates(ranked), width="stretch", key="fig_dates")
    tabs[2].caption(f"Duplicates removed (identity + embedding near-duplicates): {res.get('duplicates_removed', 0)}")
    tabs[3].plotly_chart(fig_heatmap(votes), width="stretch", key="fig_heat")
    tabs[3].dataframe([v.model_dump() | {"claim": f"C{v.claim + 1}"} for v in votes], hide_index=True, key="votes_table")
    judges = ", ".join(f"`{j}`" for j in route.judges)
    tabs[4].markdown(f"**Extraction** `{route.extract_model}` · **Synthesis** `{route.synth_model}` · **Judges** {judges}")
    tabs[4].markdown("\n".join(f"- {n}" for n in route.notes))
    if res.get("profile"):
        tabs[4].json(res["profile"].model_dump())
    cost = ev.run_cost(res.get("usage", []), prices)
    tabs[4].dataframe([{"model": mo, "$/1M in": prices[mo][0] * 1e6, "$/1M out": prices[mo][1] * 1e6, "this run $": round(cost.get(mo, 0), 5)}
                       for mo in dict.fromkeys(u.model for u in res.get("usage", [])) if mo in prices], hide_index=True, key="cost_table")
    tabs[5].dataframe([{"id": s.id, "provider": s.provider, "credibility": ev.credibility(s), "memory": s.from_memory, "title": s.title,
                        "published": s.published, "page": s.page, "url": s.url} for s in ranked], hide_index=True, key="sources_table")
    tabs[5].dataframe([c.model_dump() for c in res.get("tool_calls", [])], hide_index=True, key="calls_table")
    tabs[6].markdown(res.get("report_md", ""))


def library_view() -> None:
    query = st.text_input("Search your verified claims (semantic)", key="library_q")
    claims = brain().similar_claims(memory.embed([query])[0], k=30, floor=0.3) if query else brain().library()
    st.caption(f"{len(claims)} verified claims · backend: {brain().backend}")
    rows = [c.model_dump(exclude={"quotes"}) | {"quote": c.quotes[0] if c.quotes else ""} for c in claims]
    st.dataframe(rows, hide_index=True, key="library_table")


def sessions_view() -> None:
    runs = brain().history()
    if not runs:
        st.info("No research yet. Every finished run is logged here.")
        return
    full = [r for r in runs if not r["cache_hit"]]
    cols = st.columns(4)
    tiles: list[tuple[str, Any]] = [("Runs", len(runs)), ("From memory", len(runs) - len(full)),
             ("Avg time (full runs)", f"{ev.mean([r['seconds'] for r in full]):.0f}s"),
             ("Total spend", f"${sum(r['cost_usd'] for r in runs):.3f}")]
    for col, (label, value) in zip(cols, tiles, strict=True):
        col.metric(label, value)
    st.dataframe([{k: v for k, v in r.items() if k != "latency"} for r in runs], hide_index=True, key="history_table")
    pick = st.selectbox("Open a past report", list(dict.fromkeys(r["question"] for r in runs)), key="history_pick")
    if pick and st.button("Open report", key="history_open"):
        hit = brain().cached_report(memory.embed([pick])[0])
        st.markdown(STATE.validate_json(hit[1]["state"]).get("report_md", "") if hit else "_That report is no longer in memory._")


def history_tab() -> None:
    """One tab, two views of the second brain: what you researched, and what was verified."""
    library = st.toggle("Verified Claims Library", key="history_view", help="Off: Deep Research Sessions · On: Verified Claims Library")
    st.subheader("Verified Claims Library" if library else "Deep Research Sessions")
    if library:
        library_view()
    else:
        sessions_view()


def settings(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k != "keys"}


def main() -> None:
    st.set_page_config(page_title="ResearchForge AI", layout="wide")
    st.title("ResearchForge AI")
    st.caption("Research you can audit, and that remembers: every claim quoted, verified, cross-examined, and kept in your second brain.")
    cfg = sidebar(catalogue())
    research_tab, history = st.tabs(["Research", "History"])
    with research_tab:
        query = st.text_area("Research question (follow-ups reuse this session's evidence)", QUESTION, key="query")
        files = st.file_uploader("Optional: text-based PDFs", type="pdf", accept_multiple_files=True, key="pdfs")
        go_btn, premise_btn = st.columns(2)
        ready = bool(cfg["keys"].openrouter)
        if go_btn.button("Run research", type="primary", disabled=not (query.strip() and ready), key="run"):
            pdfs = []
            for f in files or []:
                try:
                    pdfs += pdf_sources(f.name, f.getvalue())
                except ValueError as exc:
                    st.warning(str(exc))
            run(cfg, {**settings(cfg), "query": query, "pdf_sources": pdfs})
        if premise_btn.button("Demo: false-premise question", disabled=not ready, key="false_premise"):
            run(cfg, {**settings(cfg), "query": FALSE_PREMISE, "auto_approve": True})
        res = st.session_state.get("res")
        if res and res.get("interrupt"):
            plan_editor(cfg, res["interrupt"]["plan"])
        elif res and res.get("route"):
            answer_view(cfg, res)
            detail_tabs(res)
    with history:  # rendered after the research actions, so a run that just finished is already listed
        history_tab()


if __name__ == "__main__":
    main()
