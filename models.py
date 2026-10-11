"""Shared typed contracts for every agent, tool, memory table and view."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from typing_extensions import TypedDict  # pydantic needs this TypedDict on Python < 3.12 (snapshot TypeAdapter)

from pydantic import BaseModel, Field

Provider = Literal["parallel", "tavily", "arxiv", "pdf"]
Status = Literal["established", "contested", "not_established"]
Verdict = Literal["supported", "contradicted", "unclear"]


def accumulate(old: list[Any] | None, new: list[Any] | None) -> list[Any]:
    """Reducer: parallel branches append; a new run passes None to reset (sessions reuse the checkpoint)."""
    return [] if new is None else [*(old or []), *new]


class Source(BaseModel):
    """A retrieved document (web page, arXiv paper, PDF page or remembered chunk); ``id`` is its session-stable citation id."""

    id: str = ""  # stable citation id (S1..Sn) for the whole session
    title: str
    url: str = ""
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    published: str | None = None
    retrieved: str = Field(default_factory=lambda: date.today().isoformat())
    text: str = ""
    provider: Provider
    page: int | None = None
    from_memory: bool = False


class Chunk(BaseModel):
    """A sentence-aligned slice of a source's text, the unit that is embedded, reranked and quoted."""

    source_id: str
    text: str
    page: int | None = None


class Quote(BaseModel):
    """A verbatim quote from one source, cited by its id."""

    source_id: str = Field(description="Source id exactly as shown, e.g. S3")
    quote: str = Field(description="Exact sentence copied verbatim from that source's block")


class ClaimDraft(BaseModel):
    """A finding as an analyst proposed it, before quote verification."""

    text: str = Field(description="One specific, self-contained finding")
    evidence: list[Quote] = Field(description="Supporting quotes, verbatim")
    counter: list[Quote] = Field(default_factory=list, description="Verbatim quotes that dispute or limit the finding")


class Claim(ClaimDraft):
    """A finding after verification: its status, verifier issues, judge support and trust-judge entailment."""

    status: Status = "established"
    issues: list[str] = Field(default_factory=list)
    support: float | None = None  # share of judges voting "supported"
    majority: Verdict | None = None
    entailed: bool | None = None  # trust panel: is the claim entailed by its quotes?

    @property
    def citations(self) -> list[str]:
        """Distinct source ids the claim's supporting quotes cite, in order."""
        return list(dict.fromkeys(q.source_id for q in self.evidence))


class ClaimSet(BaseModel):
    """Quoted findings from one analyst worker, plus the gaps the evidence leaves open."""

    claims: list[ClaimDraft]
    gaps: list[str] = Field(default_factory=list, description="Specific questions the evidence leaves open")


class TopicProfile(BaseModel):
    """How hard and how recent a research question is; drives model routing."""

    domain: str
    complexity: int = Field(ge=1, le=5)
    technical_depth: int = Field(ge=1, le=5)
    needs_recency: bool
    rationale: str


class Route(BaseModel):
    """The models picked for one run: extraction, synthesis and the judge panel, with the reasons."""

    extract_model: str
    synth_model: str
    judges: list[str]
    notes: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """The planner's split of a question into independent sub-questions, arXiv queries and its stated premise."""

    subquestions: list[str] = Field(description="1-3 independent, non-overlapping subquestions")
    academic_queries: list[str] = Field(description="One short arXiv keyword query per subquestion")
    premise: str = Field("", description="Any factual assumption the question takes for granted; empty if none")


class Hypothesis(BaseModel):
    """A speculative, forward-looking statement built on numbered verified findings."""

    statement: str
    kind: Literal["hypothesis", "trend"]
    based_on: list[int] = Field(default_factory=list, description="Indexes of the findings it builds on")


class Insights(BaseModel):
    """Hypotheses and trends proposed from the verified findings."""

    hypotheses: list[Hypothesis]


class JudgeVerdict(BaseModel):
    """One judge's verdict on one numbered claim, from its quotes alone."""

    index: int
    verdict: Verdict
    reason: str = ""


class JudgeReport(BaseModel):
    """One judge model's verdicts on every numbered claim."""

    verdicts: list[JudgeVerdict]


class Vote(JudgeVerdict):
    """A judge verdict tied to the claim it scores and the model that cast it."""

    claim: int
    model: str
    index: int = -1


class Entailment(BaseModel):
    """Whether one numbered claim is entailed by its quotes alone."""

    index: int
    entailed: bool


class TrustJudgement(BaseModel):
    """The trust judge's per-claim entailment and how well the findings answer the question."""

    entailment: list[Entailment] = Field(description="Per claim index: do its quotes entail it?")
    answer_relevance: float = Field(ge=0, le=1, description="How well the findings answer the question")


class Trust(BaseModel):
    """The trust panel's 0-1 scores for a finished run."""

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    citation_accuracy: float = 0.0
    consensus: float = 0.0
    credibility: float = 0.0


class Summary(BaseModel):
    """The written answer: bottom line, executive summary and a verdict on the question's premise."""

    bottom_line: str = Field(description="One plain-language sentence a non-expert understands")
    executive_summary: str = Field(description="3-5 sentences")
    premise_verdict: Literal["none", "supported", "unsupported", "false"] = Field(
        "none", description="Is the question's premise backed by the findings? 'false' if they contradict it"
    )
    premise_note: str = Field("", description="If the premise is unsupported or false, correct it in one sentence")


class ToolCall(BaseModel):
    """One external search or fetch call: provider, tool, outcome, latency and result count (errors are redacted)."""

    provider: str
    tool: str
    ok: bool
    ms: int
    results: int = 0
    error: str = ""


class Snippet(BaseModel):
    """A short preview of one search result, shown in the live agent activity."""

    provider: str
    title: str
    url: str = ""
    text: str = ""  # about 150 characters


class TraceEvent(BaseModel):
    """One step of agent activity shown in the live log; searches also carry the tool, query and result previews."""

    agent: str
    action: str
    started: float = 0.0
    ms: int = 0
    detail: str = ""
    tool: str = ""  # search tool(s) a retriever used, e.g. "Parallel MCP" or "arXiv"
    query: str = ""  # the exact query that tool was sent
    snippets: list[Snippet] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Tokens one agent spent on one LLM call (``cached`` calls cost nothing)."""

    agent: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class StoredClaim(BaseModel):
    """A verified claim as kept in the second brain's library, with its quotes, sources and originating question."""

    text: str
    quotes: list[str]
    sources: list[str]  # "title <url>"
    status: Status
    support: float | None
    question: str
    date: str
    similarity: float = 0.0


class ResearchState(TypedDict, total=False):
    """The LangGraph state of a research session; list fields with reducers accumulate across parallel branches."""

    query: str
    max_sources: int
    default_model: str
    judge_count: int
    auto_route: bool
    auto_approve: bool
    max_loops: int
    force_refresh: bool
    t0: float
    task: str  # payload of one parallel retriever branch
    kind: str  # "web" | "arxiv"
    cache_hit: bool
    memory_claims: list[StoredClaim]
    memory_sources: list[Source]
    profile: TopicProfile | None
    route: Route
    plan: ResearchPlan
    pdf_sources: list[Source]
    ids: dict[str, str]  # canonical source key -> stable id
    ranked: list[Source]
    duplicates_removed: int
    loops: int
    focus: list[str]  # questions the analyst works on this pass (plan, then gaps)
    gaps: list[str]
    claims: list[Claim]
    hypotheses: list[Hypothesis]
    votes: list[Vote]
    entailment: dict[int, bool]
    relevance: float
    trust: Trust
    summary: Summary | None
    report_md: str
    sources: Annotated[list[Source], accumulate]
    drafts: Annotated[list[ClaimDraft], accumulate]
    trace: Annotated[list[TraceEvent], accumulate]
    tool_calls: Annotated[list[ToolCall], accumulate]
    usage: Annotated[list[TokenUsage], accumulate]
    context_chars: Annotated[list[tuple[int, int]], accumulate]
