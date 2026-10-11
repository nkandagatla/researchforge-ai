"""Small helpers shared by the UI modules."""

from __future__ import annotations

import re
from typing import Any

Result = dict[str, Any]  # what agents.research() returns
AGENT_ICONS = {
    "Memory": "🧠", "Topic Router": "🔀", "Planner": "🗺️", "Contextual Retriever": "🔎", "Critical Analyst": "🔬",
    "Verifier": "✅", "Supervisor": "🧭", "Consensus Panel": "⚖️", "Insight Generator": "💡", "Trust Judge": "🛡️",
    "Report Builder": "📝", "Reranker (LLM fallback)": "↕️",
}
_MD_SPECIAL = re.compile(r"([\\`*_\[\]$<>#|~])")


def md(text: str) -> str:
    """Escape Markdown so web text (quotes, titles, questions) renders literally: no stray italics, links or LaTeX."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


def icon(agent: str) -> str:
    return AGENT_ICONS.get(agent, "•")
