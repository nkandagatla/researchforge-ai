"""Cached services and the background research job of this browser session (the script thread never blocks)."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

import agents
import jobs
import memory
import tools


@st.cache_data(ttl=3600, show_spinner=False)
def catalogue() -> dict[str, tuple[float, float]]:
    """Live OpenRouter model ids and prices (empty when the catalogue is unreachable)."""
    return tools.openrouter_models()


@st.cache_resource(show_spinner="Loading memory...")
def brain() -> memory.Memory:
    """The process-wide second brain (one Memory per server, cached across reruns and sessions)."""
    return memory.Memory()


def settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run settings without the API keys (keys never enter the graph state or the checkpoint)."""
    return {k: v for k, v in cfg.items() if k != "keys"}


def current_job() -> jobs.Job | None:
    """This browser session's background research job, if it has one."""
    return jobs.get(st.session_state.get("job"))


def busy() -> bool:
    """True while this session's research job is still running (the run buttons are disabled)."""
    job = current_job()
    return bool(job and job.running)


def run(cfg: dict[str, Any], inputs: dict[str, Any] | None, resume: Any = None) -> None:
    """Start one research turn in a background thread; ignored while this session already has one running (double clicks)."""
    if busy():
        return
    thread = st.session_state.setdefault("thread", str(uuid.uuid4()))
    keys, prices, mem, research = cfg["keys"], catalogue(), brain(), agents.research  # resolved here, on the script thread

    def make(emit: jobs.Emit) -> Any:
        return research(keys, prices, mem, inputs, thread, resume, emit)

    st.session_state.job = jobs.start(make, lambda text: tools.redact(text, list(keys))).id
    if inputs and inputs.get("query"):
        st.session_state.asked = inputs["query"]
    for stale in ("run_error", "last_run"):
        st.session_state.pop(stale, None)


def collect_job() -> None:
    """Move a finished background run into this session, exactly once."""
    job_id = st.session_state.get("job")
    job = jobs.get(job_id)
    if job_id and job is None:  # server restarted or the job was pruned
        st.session_state.pop("job", None)
    if job is None or job.running:
        return
    jobs.pop(job.id)
    st.session_state.pop("job", None)
    if job.result is not None:
        res = st.session_state.res = job.result
        done = f"Done in {res['wall_s']:.1f}s" + (" (from memory)" if res.get("cache_hit") else "")
        st.session_state.last_run = ("Waiting for plan approval" if res["interrupt"] else done, "complete", job.events)
    else:
        st.session_state.last_run = ("Research failed", "error", job.events)
        st.session_state.run_error = job.error or ("RuntimeError", "unknown error")
