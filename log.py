"""Logging for ResearchForge: one ``researchforge.*`` logger tree, secrets redacted before any handler sees a record.

Modules call ``get_logger(__name__)``; the entry points (app.py, run_cli.py, evals/run.py) call ``configure()`` once.
The level comes from RF_LOG_LEVEL (default INFO); DEBUG adds per-node timings.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

ROOT = "researchforge"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
SECRET_ENV = ("OPENROUTER_API_KEY", "TAVILY_API_KEY", "PARALLEL_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY",
              "LANGSMITH_API_KEY", "HF_TOKEN", "HF_API_KEY")
_RUN_SECRETS: list[str] = []  # keys pasted into the UI for a run (not in the environment); bounded, newest last
_MAX_RUN_SECRETS = 16


def add_secrets(*values: str) -> None:
    """Also redact these values (e.g. a key typed into the sidebar) from every later log record."""
    for v in values:
        if v and len(v) >= 8 and v not in _RUN_SECRETS:
            _RUN_SECRETS.append(v)
    del _RUN_SECRETS[:-_MAX_RUN_SECRETS]


def secrets() -> list[str]:
    """Every secret value that must never reach a log line: env keys plus keys registered for runs."""
    return [v for v in (os.getenv(k, "") for k in SECRET_ENV) if len(v) >= 8] + _RUN_SECRETS


class RedactingFilter(logging.Filter):
    """Rewrites the message and any traceback with tools.redact, so caplog, stderr and every other handler get clean text."""

    def filter(self, record: logging.LogRecord) -> bool:
        from tools import redact  # lazy: tools itself logs through this module

        known = secrets()
        record.msg, record.args = redact(record.getMessage(), known), None
        if record.exc_info and not record.exc_text:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info), known)
        return True


_FILTER = RedactingFilter()


def get_logger(module: str) -> logging.Logger:
    """The ``researchforge.<module>`` logger, with the redacting filter attached (records are clean at the source)."""
    name = module if module.startswith(ROOT) else f"{ROOT}.{module.rsplit('.', 1)[-1]}"
    logger = logging.getLogger(name)
    if _FILTER not in logger.filters:
        logger.addFilter(_FILTER)
    return logger


def configure(level: str | None = None, stream: TextIO | None = None) -> logging.Logger:
    """Attach one concise stderr handler to the ``researchforge`` logger; calling it again only updates the level."""
    root = logging.getLogger(ROOT)
    name = (level or os.getenv("RF_LOG_LEVEL") or "INFO").upper()
    root.setLevel(getattr(logging, name, logging.INFO))
    if not any(getattr(h, "_rf", False) for h in root.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
        handler.addFilter(_FILTER)  # records from third-party children of this tree are redacted too
        handler._rf = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    return root
