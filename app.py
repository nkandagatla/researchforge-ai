"""ResearchForge AI v2: research you can audit, and that remembers. Streamlit entry point (the UI lives in ui/)."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

import log
from ui.history import history_tab
from ui.research import research_tab
from ui.session import catalogue, collect_job
from ui.sidebar import sidebar

load_dotenv()
log.configure()  # RF_LOG_LEVEL (default INFO) -> stderr, i.e. the Streamlit server log


def main() -> None:
    """Build the page: sidebar settings, collect any finished background job, then the Research and History tabs."""
    st.set_page_config(page_title="ResearchForge AI", layout="wide")
    st.title("ResearchForge AI")
    st.caption("Research you can audit, and that remembers: every claim quoted, verified, cross-examined, and kept in your second brain.")
    cfg = sidebar(catalogue())
    collect_job()
    research, history = st.tabs(["Research", "History"])
    with research:
        research_tab(cfg)
    with history:  # rendered after the research actions, so a run that just finished is already listed
        history_tab()


if __name__ == "__main__":
    main()
