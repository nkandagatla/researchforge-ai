"""History tab: two views of the second brain, what you researched and what was verified."""

from __future__ import annotations

from typing import Any

import streamlit as st

import evidence as ev
import memory
from agents import STATE
from ui.session import brain


def library_view() -> None:
    """The verified-claims library, filtered by semantic search when a query is typed."""
    query = st.text_input("Search your verified claims (semantic)", key="library_q")
    claims = brain().similar_claims(memory.embed([query])[0], k=30, floor=0.3) if query else brain().library()
    st.caption(f"{len(claims)} verified claims · backend: {brain().backend}")
    rows = [c.model_dump(exclude={"quotes"}) | {"quote": c.quotes[0] if c.quotes else ""} for c in claims]
    st.dataframe(rows, hide_index=True, key="library_table")


def session_tiles(runs: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """Headline numbers over every logged run (time is averaged over full runs only: cache hits take well under a second)."""
    full = [r for r in runs if not r["cache_hit"]]
    return [("Runs", len(runs)), ("From memory", len(runs) - len(full)), ("Avg time (full runs)", f"{ev.mean([r['seconds'] for r in full]):.0f}s"),
            ("Total spend", f"${sum(r['cost_usd'] for r in runs):.3f}")]


def past_report(question: str) -> str:
    """The stored report for a past question, looked up through the semantic cache."""
    hit = brain().cached_report(memory.embed([question])[0])
    return STATE.validate_json(hit[1]["state"]).get("report_md", "") if hit else "_That report is no longer in memory._"


def sessions_view() -> None:
    """Logged research runs: headline tiles, the run table and a picker to reopen a past report."""
    runs = brain().history()
    if not runs:
        st.info("No research yet. Every finished run is logged here.")
        return
    for col, (label, value) in zip(st.columns(4), session_tiles(runs), strict=True):
        col.metric(label, value)
    st.dataframe([{k: v for k, v in r.items() if k != "latency"} for r in runs], hide_index=True, key="history_table")
    pick = st.selectbox("Open a past report", list(dict.fromkeys(r["question"] for r in runs)), key="history_pick")
    if pick and st.button("Open report", key="history_open"):
        st.markdown(past_report(pick))


def history_tab() -> None:
    """The History tab: a toggle between research sessions and the verified-claims library."""
    library = st.toggle("Verified Claims Library", key="history_view", help="Off: Deep Research Sessions · On: Verified Claims Library")
    st.subheader("Verified Claims Library" if library else "Deep Research Sessions")
    if library:
        library_view()
    else:
        sessions_view()
