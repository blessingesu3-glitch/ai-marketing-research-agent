from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import app as research_graph


# ---------------------------------
# API setup
# ---------------------------------

api = FastAPI(
    title="AI Marketing Research Agent",
    description="A LangGraph-powered marketing research API.",
    version="1.0.0",
)


# ---------------------------------
# Allow the frontend to communicate
# with the backend
# ---------------------------------

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------
# Request model
# ---------------------------------

class ResearchRequest(BaseModel):
    question: str


# ---------------------------------
# Response model
# ---------------------------------

class ResearchReportResponse(BaseModel):
    question: str
    summary: str
    key_findings: List[str]
    opportunities: List[str]
    recommendations: List[str]
    sources: List[str]


# ---------------------------------
# Health check
# ---------------------------------

@api.get("/")
def health_check():
    return {
        "status": "online",
        "agent": "AI Marketing Research Agent",
    }


# ---------------------------------
# Research endpoint
# ---------------------------------

@api.post("/research", response_model=ResearchReportResponse)
def run_research(request: ResearchRequest):

    result = research_graph.invoke(
        {
            "question": request.question,
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
