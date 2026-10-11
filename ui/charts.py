"""Figures for the answer tabs: Plotly figures and self-contained Recharts pages. Pure functions of a result, unit-tested."""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from datetime import datetime
from typing import Any

import plotly.graph_objects as go

import evidence as ev
from models import Claim, Source, Vote
from ui.common import Result

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


def ring(names: list[str], r: float) -> dict[str, tuple[float, float]]:
    """Evenly spaced positions on a circle of radius r."""
    step = 2 * math.pi / max(len(names), 1)
    return {n: (r * math.cos(i * step), r * math.sin(i * step)) for i, n in enumerate(names)}


def _verdict_edges(claims: list[Claim], cnames: list[str], pos: dict[str, tuple[float, float]], verdict: str) -> go.Scatter:
    """One line trace with every claim-to-source edge whose majority verdict is `verdict` (None breaks the line)."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    for name, c in zip(cnames, claims, strict=True):
        if (c.majority or "unclear") != verdict:
            continue
        for sid in (s for s in c.citations if s in pos):
            xs += [pos[name][0], pos[sid][0], None]
            ys += [pos[name][1], pos[sid][1], None]
    return go.Scatter(x=xs, y=ys, mode="lines", line={"color": VERDICT_COLOR[verdict], "width": 2}, name=verdict, hoverinfo="skip")


def fig_evidence_graph(claims: list[Claim], sources: list[Source]) -> go.Figure:
    """Claims (inner ring) to sources (outer ring); edge colour = consensus majority verdict."""
    by_id = {s.id: s for s in sources}
    used = list(dict.fromkeys(sid for c in claims for sid in c.citations if sid in by_id))
    cnames = [f"C{i + 1}" for i in range(len(claims))]
    pos = ring(cnames, 0.45) | ring(used, 1.0)
    fig = go.Figure([_verdict_edges(claims, cnames, pos, verdict) for verdict in VERDICT_COLOR])
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


def citation_matrix(claims: list[Claim], sources: list[Source]) -> tuple[list[str], list[list[int]], Counter[str]]:
    """Columns = cited sources, most cited first; cell = 1 supporting quote, -1 counter-quote, 0 none."""
    cites = Counter(q.source_id for c in claims for q in [*c.evidence, *c.counter])
    cols = sorted((s.id for s in sources if s.id in cites), key=lambda sid: -cites[sid])
    rows = []
    for c in claims:
        counter = {q.source_id for q in c.counter}
        rows.append([1 if sid in c.citations else -1 if sid in counter else 0 for sid in cols])
    return cols, rows, cites


def fig_citations(claims: list[Claim], sources: list[Source]) -> go.Figure:
    """Citation heat map: which source backs (blue) or disputes (red) which claim; column labels count the citations."""
    cols, z, cites = citation_matrix(claims, sources)
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
