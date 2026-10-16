from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import BusinessProfile, app as research_graph
from deliverables import Kind, make_deliverable
from ratelimit import enforce_rate_limit
from teardown import MAX_COMPETITORS, run_teardown


api = FastAPI(
    title="AI Marketing Research Agent",
    description="A LangGraph-powered marketing research API.",
    version="1.0.0",
)


api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ai-marketing-research-agent.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    profile: Optional[BusinessProfile] = None   # optional: tailor the research to this business


class ResearchReportResponse(BaseModel):
    question: str
    summary: str
    key_findings: List[str]
    opportunities: List[str]
    recommendations: List[str]
    sources: List[str]


@api.get("/")
def health_check():
    return {
        "status": "online",
        "agent": "AI Marketing Research Agent",
    }


@api.post("/research", response_model=ResearchReportResponse)
def run_research(request: ResearchRequest, http_request: Request):
    # Checked inside the handler (after the body is validated), so invalid
    # requests are rejected without using up anyone's quota.
    enforce_rate_limit(http_request)

    result = research_graph.invoke(
        {
            "question": request.question,
            "profile": request.profile.model_dump() if request.profile else None,
            "search_results": "",
            "report": None,
        }
    )

    report = result["report"]

    return {
        "question": result["question"],
        "summary": report.summary,
        "key_findings": report.key_findings,
        "opportunities": report.opportunities,
        "recommendations": report.recommendations,
        "sources": report.sources,
    }


# ---------------------------------------------------------------- competitor teardown

class TeardownRequest(BaseModel):
    competitors: List[str] = Field(min_length=1, max_length=MAX_COMPETITORS)
    profile: Optional[BusinessProfile] = None
    focus: Optional[str] = Field(default=None, max_length=500)


class CompetitorCardResponse(BaseModel):
    name: str
    website: str
    offer: str
    pricing: str
    target_customer: str
    positioning: str
    channels: str
    sources: List[str]
    note: str


class TeardownResponse(BaseModel):
    summary: str
    competitors: List[CompetitorCardResponse]
    gaps: List[str]
    recommendations: List[str]
    sources: List[str]


@api.post("/teardown", response_model=TeardownResponse)
def run_competitor_teardown(request: TeardownRequest, http_request: Request):
    entries = [c.strip() for c in request.competitors if c and c.strip()]
    if not entries or any(len(c) > 200 for c in entries):
        raise HTTPException(status_code=422, detail="Give 1-3 competitor names or websites (max 200 characters each).")
    enforce_rate_limit(http_request)
    return run_teardown(
        entries,
        profile=request.profile.model_dump() if request.profile else None,
        focus=request.focus,
    )


# ---------------------------------------------------------------- ready-to-use work

class SourceReport(BaseModel):
    """The report the user already has (a research report or a teardown turned into this shape)."""
    question: str = Field(default="", max_length=2000)
    summary: str = Field(default="", max_length=8000)
    key_findings: List[str] = Field(default_factory=list, max_length=40)
    opportunities: List[str] = Field(default_factory=list, max_length=20)
    recommendations: List[str] = Field(default_factory=list, max_length=20)


class DeliverableRequest(BaseModel):
    kind: Kind
    report: SourceReport
    profile: Optional[BusinessProfile] = None


class DeliverableSection(BaseModel):
    heading: str
    items: List[str]


class DeliverableResponse(BaseModel):
    kind: str
    title: str
    sections: List[DeliverableSection]
    text: str


@api.post("/deliverable", response_model=DeliverableResponse)
def create_deliverable(request: DeliverableRequest, http_request: Request):
    report = request.report.model_dump()
    items = report["key_findings"] + report["opportunities"] + report["recommendations"]
    if not report["summary"].strip() and not items:
        raise HTTPException(status_code=422, detail="There is no report content to work from.")
    if any(len(i) > 2000 for i in items):
        raise HTTPException(status_code=422, detail="A report item is too long.")
    enforce_rate_limit(http_request)
    return make_deliverable(
        request.kind,
        report,
        profile=request.profile.model_dump() if request.profile else None,
    )
