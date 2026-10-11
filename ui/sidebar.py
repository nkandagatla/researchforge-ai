"""Sidebar: API keys (from .env / Streamlit secrets, or typed in) and the run settings."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from agents import JUDGE_POOL, Keys
from ui.session import brain


def key_input(label: str, env: str, required: bool) -> str:
    """A key from the environment/.env if set (never shown), otherwise a password field."""
    if value := os.getenv(env, ""):
        st.sidebar.caption(f"{label}: loaded from .env")
        return value
    value = st.sidebar.text_input(f"{label} ({'required' if required else 'optional'})", type="password", key=f"key_{env}")
    if required and not value:
        st.sidebar.warning(f"Add {env} to .env or enter it here.")
    return value


def tracing_toggle() -> None:
    """Offer LangSmith/Langfuse tracing only when their keys are configured; sets RF_TRACING for the next run."""
    backends = [n for n, k in (("LangSmith", "LANGSMITH_API_KEY"), ("Langfuse", "LANGFUSE_PUBLIC_KEY")) if os.getenv(k)]
    if backends:
        os.environ["RF_TRACING"] = str(st.sidebar.toggle(f"Trace runs ({' + '.join(backends)})", False, key="tracing")).lower()


def sidebar(models: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Keys and run settings from the sidebar as the cfg dict the Research tab runs with; also the New session button."""
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
    tracing_toggle()
    live = f"{len(models)} OpenRouter models validated live" if models else "OpenRouter catalogue unreachable"
    st.sidebar.caption(f"{live} · memory: {brain().backend}")
    if st.sidebar.button("New session", key="clear"):
        st.session_state.clear()
        st.rerun()
    return cfg
