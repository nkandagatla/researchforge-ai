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
    source_id: str
    text: str
    page: int | None = None


class Quote(BaseModel):
    source_id: str = Field(description="Source id exactly as shown, e.g. S3")
    quote: str = Field(description="Exact sentence copied verbatim from that source's block")


class ClaimDraft(BaseModel):
    text: str = Field(description="One specific, self-contained finding")
    evidence: list[Quote] = Field(description="Supporting quotes, verbatim")
    counter: list[Quote] = Field(default_factory=list, description="Verbatim quotes that dispute or limit the finding")


class Claim(ClaimDraft):
    status: Status = "established"
    issues: list[str] = Field(default_factory=list)
    support: float | None = None  # share of judges voting "supported"
    majority: Verdict | None = None
    entailed: bool | None = None  # trust panel: is the claim entailed by its quotes?

    @property
    def citations(self) -> list[str]:
        return list(dict.fromkeys(q.source_id for q in self.evidence))


class ClaimSet(BaseModel):
    claims: list[ClaimDraft]
    gaps: list[str] = Field(default_factory=list, description="Specific questions the evidence leaves open")


class TopicProfile(BaseModel):
    domain: str
    complexity: int = Field(ge=1, le=5)
    technical_depth: int = Field(ge=1, le=5)
    needs_recency: bool
    rationale: str


class Route(BaseModel):
    extract_model: str
    synth_model: str
    judges: list[str]
    notes: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    subquestions: list[str] = Field(description="1-3 independent, non-overlapping subquestions")
    academic_queries: list[str] = Field(description="One short arXiv keyword query per subquestion")
    premise: str = Field("", description="Any factual assumption the question takes for granted; empty if none")


class Hypothesis(BaseModel):
    statement: str
    kind: Literal["hypothesis", "trend"]
    based_on: list[int] = Field(default_factory=list, description="Indexes of the findings it builds on")


class Insights(BaseModel):
    hypotheses: list[Hypothesis]


class JudgeVerdict(BaseModel):
    index: int
    verdict: Verdict
    reason: str = ""


class JudgeReport(BaseModel):
    verdicts: list[JudgeVerdict]


class Vote(JudgeVerdict):
    claim: int
    model: str
    index: int = -1


class Entailment(BaseModel):
    index: int
    entailed: bool


class TrustJudgement(BaseModel):
    entailment: list[Entailment] = Field(description="Per claim index: do its quotes entail it?")
    answer_relevance: float = Field(ge=0, le=1, description="How well the findings answer the question")


class Trust(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    citation_accuracy: float = 0.0
    consensus: float = 0.0
    credibility: float = 0.0


class Summary(BaseModel):
    bottom_line: str = Field(description="One plain-language sentence a non-expert understands")
    executive_summary: str = Field(description="3-5 sentences")
    premise_verdict: Literal["none", "supported", "unsupported", "false"] = Field(
        "none", description="Is the question's premise backed by the findings? 'false' if they contradict it"
    )
    premise_note: str = Field("", description="If the premise is unsupported or false, correct it in one sentence")


class ToolCall(BaseModel):
    provider: str
    tool: str
    ok: bool
    ms: int
    results: int = 0
    error: str = ""


class TraceEvent(BaseModel):
    agent: str
    action: str
    started: float = 0.0
    ms: int = 0
    detail: str = ""


class TokenUsage(BaseModel):
    agent: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class StoredClaim(BaseModel):
    text: str
    quotes: list[str]
    sources: list[str]  # "title <url>"
    status: Status
    support: float | None
    question: str
    date: str
    similarity: float = 0.0


class ResearchState(TypedDict, total=False):
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
