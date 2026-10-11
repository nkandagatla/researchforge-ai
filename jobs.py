"""Background research jobs.

A research run takes 50-90 s. Running it on the Streamlit script thread blocked that session: clicks made meanwhile were
dropped and long runs could outlive the browser connection. Each run now gets its own worker thread with its own event loop;
the page polls the job (st.fragment(run_every=...)) and collects the result once. The registry is module level, so it
survives Streamlit reruns, and a server-wide slot limit keeps two judges from running two memory-hungry runs at once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("researchforge")
Emit = Callable[[Any], None]  # agent activity: TraceEvents from the graph, plain-text notices from the runner
MakeRun = Callable[[Emit], Coroutine[Any, Any, dict[str, Any]]]
STALE_S = 1800  # finished jobs nobody collected (tab closed) are dropped after this
_SLOTS = threading.BoundedSemaphore(max(1, int(os.getenv("RF_MAX_CONCURRENT_RUNS", "1"))))
_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    events: list[Any] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: tuple[str, str] | None = None  # (exception type, short redacted message)
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.finished is None

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started


def start(make: MakeRun, redact: Callable[[str], str]) -> Job:
    """Run make(emit) in a worker thread; emit(event) streams agent activity to the page."""
    job = Job(uuid.uuid4().hex)

    def target() -> None:
        try:
            if not _SLOTS.acquire(blocking=False):
                job.events.append("Another research run is using this server; yours starts as soon as it finishes...")
                _SLOTS.acquire()
            try:
                job.result = asyncio.run(make(job.events.append))
            finally:
                _SLOTS.release()
        except BaseException as exc:  # the worker must never die silently: the page shows a friendly error + details
            job.error = (type(exc).__name__, redact(str(exc))[:500])
            log.warning("research job %s failed: %s: %s", job.id, *job.error)
        finally:
            job.finished = time.monotonic()

    prune()
    job.thread = threading.Thread(target=target, name=f"research-{job.id[:8]}", daemon=True)
    with _LOCK:
        _JOBS[job.id] = job
    job.thread.start()
    return job


def get(job_id: str | None) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id or "")


def pop(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.pop(job_id, None)


def prune(now: float | None = None) -> None:
    now = now or time.monotonic()
    with _LOCK:
        for jid in [j.id for j in _JOBS.values() if j.finished is not None and now - j.finished > STALE_S]:
            del _JOBS[jid]


def wait_all(timeout: float = 60) -> bool:
    """Tests and scripts: block until every running job has finished."""
    deadline = time.monotonic() + timeout
    with _LOCK:
        threads = [j.thread for j in _JOBS.values() if j.thread]
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    return not any(t.is_alive() for t in threads)
