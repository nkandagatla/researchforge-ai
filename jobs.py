"""Background research jobs.

A research run takes 50-90 s. Running it on the Streamlit script thread blocked that session: clicks made meanwhile were
dropped and long runs could outlive the browser connection. Each run now gets its own worker thread with its own event loop;
the page polls the job (st.fragment(run_every=...)) and collects the result once. The registry is module level, so it
survives Streamlit reruns, and a server-wide slot limit keeps two judges from running two memory-hungry runs at once.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from log import get_logger
from models import AnswerDraft

logger = get_logger(__name__)
Emit = Callable[[Any], None]  # TraceEvents from the graph, plain-text notices from the runner, AnswerDrafts while writing
MakeRun = Callable[[Emit], Coroutine[Any, Any, dict[str, Any]]]
STALE_S = 1800  # finished jobs nobody collected (tab closed) are dropped after this
_SLOTS = threading.BoundedSemaphore(max(1, int(os.getenv("RF_MAX_CONCURRENT_RUNS", "1"))))
_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


@dataclass
class Job:
    """One background research run: streamed events and answer text, then a result or a redacted error."""

    id: str
    events: list[Any] = field(default_factory=list)
    partial: str = ""  # the answer as written so far (replaced whole on each draft: one atomic reference swap per update)
    result: dict[str, Any] | None = None
    error: tuple[str, str] | None = None  # (exception type, short redacted message)
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    thread: threading.Thread | None = None

    def emit(self, item: Any) -> None:
        """Worker-thread side of the stream: an AnswerDraft replaces the partial answer, anything else is an activity step.

        The page thread only reads: list.append and rebinding a str attribute are atomic, so no lock is needed."""
        if isinstance(item, AnswerDraft):
            self.partial = item.text
        else:
            self.events.append(item)

    @property
    def running(self) -> bool:
        """True until the worker thread has finished (successfully or not)."""
        return self.finished is None

    @property
    def elapsed(self) -> float:
        """Seconds since the job was created (frozen once it finishes); includes any wait for a free slot."""
        return (self.finished or time.monotonic()) - self.started


def start(make: MakeRun, redact: Callable[[str], str]) -> Job:
    """Run make(emit) in a worker thread; emit(item) streams agent activity and the answer being written to the page."""
    job = Job(uuid.uuid4().hex)
    logger.info("job %s submitted (%d jobs tracked)", job.id[:8], len(_JOBS))

    def target() -> None:
        try:
            if not _SLOTS.acquire(blocking=False):
                job.events.append("Another research run is using this server; yours starts as soon as it finishes...")
                logger.info("job %s queued: every run slot is busy", job.id[:8])
                _SLOTS.acquire()
                logger.info("job %s got a slot after waiting %.1f s", job.id[:8], job.elapsed)
            try:
                job.result = asyncio.run(make(job.emit))
            finally:
                _SLOTS.release()
        except BaseException as exc:  # the worker must never die silently: the page shows a friendly error + details
            job.error = (type(exc).__name__, redact(str(exc))[:500])
            logger.warning("job %s failed: %s: %s", job.id[:8], *job.error, exc_info=exc)
        finally:
            job.finished = time.monotonic()
            logger.info("job %s %s after %.1f s", job.id[:8], "failed" if job.error else "finished", job.elapsed)

    prune()
    job.thread = threading.Thread(target=target, name=f"research-{job.id[:8]}", daemon=True)
    with _LOCK:
        _JOBS[job.id] = job
    job.thread.start()
    return job


def get(job_id: str | None) -> Job | None:
    """The job with this id, or None (unknown id, already collected, or pruned)."""
    with _LOCK:
        return _JOBS.get(job_id or "")


def pop(job_id: str) -> Job | None:
    """Remove and return a job once its result has been collected."""
    with _LOCK:
        return _JOBS.pop(job_id, None)


def prune(now: float | None = None) -> None:
    """Drop finished jobs nobody collected within STALE_S (e.g. the tab was closed)."""
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
