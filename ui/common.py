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


def link(text: str, url: str) -> str:
    """A Markdown link that survives spaces and parentheses in the URL; plain escaped text when there is no URL."""
    safe = url.strip().replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{md(text)}]({safe})" if safe else md(text)


def icon(agent: str) -> str:
    """The agent's emoji in the activity log ('•' for an unknown agent)."""
    return AGENT_ICONS.get(agent, "•")
