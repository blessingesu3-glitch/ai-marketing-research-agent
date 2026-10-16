from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import BusinessProfile, app as research_graph


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
def run_research(request: ResearchRequest):

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
