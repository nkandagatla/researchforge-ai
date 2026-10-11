"""Research tab: the question, the run buttons, plan approval, the live agent story and the finished answer."""

from __future__ import annotations

from typing import Any

import streamlit as st

from tools import pdf_sources, redact
from ui import activity, answer, tabs
from ui.session import busy, catalogue, run, settings

QUESTION = "How is quantum computing going to change the current AI LLM/models and agentic implementations?"
FALSE_PREMISE = "Why have quantum computers already replaced GPUs for training LLMs?"


def show_error(title: str, error: tuple[str, str]) -> None:
    """Friendly message first; the technical detail (type + short, redacted message) stays one click away."""
    st.error(f"{title} Your question and settings are kept, so you can simply run it again.")
    with st.expander("Error details"):
        st.code(f"{error[0]}: {error[1]}", language=None)


def plan_editor(cfg: dict[str, Any], plan: dict[str, Any]) -> None:
    st.info("Review the research plan before any searching or analysis is spent on it.")
    subs = st.text_area("Subquestions (one per line)", "\n".join(plan["subquestions"]), key="plan_subs")
    arxiv = st.text_area("arXiv queries (one per line)", "\n".join(plan["academic_queries"]), key="plan_arxiv")
    premise = st.text_input("Premise to verify", plan.get("premise", ""), key="plan_premise")
    if st.button("Approve plan and research", type="primary", key="plan_go"):
        lines = [s.strip() for s in subs.splitlines() if s.strip()][:3]
        run(cfg, None, {"subquestions": lines or plan["subquestions"][:3],
                        "academic_queries": [a.strip() for a in arxiv.splitlines() if a.strip()][:3], "premise": premise})
        st.rerun()


def uploaded_pdfs(files: list[Any] | None) -> list[Any]:
    pdfs = []
    for f in files or []:
        try:
            pdfs += pdf_sources(f.name, f.getvalue())
        except ValueError as exc:  # scanned or encrypted PDF: say so and research without it
            st.warning(str(exc))
    return pdfs


def ask_form(cfg: dict[str, Any]) -> None:
    query = st.text_area("Research question (follow-ups reuse this session's evidence)", QUESTION, key="query")
    files = st.file_uploader("Optional: text-based PDFs", type="pdf", accept_multiple_files=True, key="pdfs")
    go_btn, premise_btn = st.columns(2)
    ready, running = bool(cfg["keys"].openrouter), busy()
    if go_btn.button("Run research", type="primary", disabled=running or not (query.strip() and ready), key="run"):
        run(cfg, {**settings(cfg), "query": query, "pdf_sources": uploaded_pdfs(files)})
        st.rerun()
    if premise_btn.button("Demo: false-premise question", disabled=running or not ready, key="false_premise"):
        run(cfg, {**settings(cfg), "query": FALSE_PREMISE, "auto_approve": True})
        st.rerun()


def assistant_turn(cfg: dict[str, Any], res: dict[str, Any] | None, error: tuple[str, str] | None) -> bool:
    """Activity log, then the error, the plan to approve or the answer; True when a finished answer was shown."""
    with st.chat_message("assistant"):
        activity.last_status()
        if error:
            show_error("The research run hit a problem and stopped before finishing.", error)
            return False
        if res and res.get("interrupt"):
            plan_editor(cfg, res["interrupt"]["plan"])
            return False
        if res and res.get("route"):
            answer.answer_view(cfg, res)
            return True
    return False


def conversation(cfg: dict[str, Any]) -> None:
    """The asked question, the assistant's turn, then (for a finished answer) demos, execution summary and the detail tabs."""
    res, error = st.session_state.get("res"), st.session_state.get("run_error")
    if question := st.session_state.get("asked") or (res or {}).get("query"):
        answer.user_bubble(question)
    if assistant_turn(cfg, res, error) and res:
        answer.fabricated_demo(res.get("ranked", []))
        st.download_button("Download report (.md)", res.get("report_md", ""), "research_report.md", key="download")
        activity.execution_summary(res, catalogue())
        tabs.detail_tabs(res)


def research_tab(cfg: dict[str, Any]) -> None:
    ask_form(cfg)
    if busy():
        if question := st.session_state.get("asked"):
            answer.user_bubble(question)
        with st.chat_message("assistant"):
            activity.live_status()
    elif st.session_state.get("res") or st.session_state.get("run_error"):
        try:
            conversation(cfg)
        except Exception as exc:  # a rendering bug must never take the whole app down
            show_error("Part of this result could not be displayed.", (type(exc).__name__, redact(str(exc), list(cfg["keys"]))[:500]))
