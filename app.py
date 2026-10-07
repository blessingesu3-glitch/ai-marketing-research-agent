import os
import sys
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

load_dotenv()


# ---------------------------------
# 1. Define the report structure
# ---------------------------------

class ResearchReport(BaseModel):
    summary: str = Field(
        description="A concise summary of the most important findings."
    )

    key_findings: List[str] = Field(
        description="The most important evidence-based findings from the research."
    )

    opportunities: List[str] = Field(
        description="Practical marketing or SEO opportunities identified from the research."
    )

    recommendations: List[str] = Field(
        description="Specific actions a marketing professional should take."
    )

    sources: List[str] = Field(
        description="URLs of useful sources that actually appeared in the research results."
    )


# ---------------------------------
# 2. Define the LangGraph state
# ---------------------------------

class ResearchState(TypedDict):
    question: str
    search_results: str
    report: Optional[ResearchReport]


# ---------------------------------
# 3. Set up search and AI model
# ---------------------------------

search = TavilySearch(max_results=5)

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    temperature=0,
)

structured_model = model.with_structured_output(ResearchReport)


# ---------------------------------
# 4. Research node
# ---------------------------------

def research(state: ResearchState):
    question = state["question"]

    print("\n[Research] Searching the web...")

    results = search.invoke({
        "query": question
    })

    print("[Research] Search complete.")

    return {
        "question": question,
        "search_results": str(results),
        "report": None,
    }


# ---------------------------------
# 5. Analysis node
# ---------------------------------

def analyze(state: ResearchState):
    question = state["question"]
    search_results = state["search_results"]

    print("[Analysis] Analyzing the research...")

    prompt = f"""
You are an AI marketing research analyst.

The user wants research on:

{question}

Use the following web search results:

{search_results}

Create a research report based ONLY on the information supported
by these search results.

Rules:

1. Do not invent statistics.
2. Do not invent sources.
3. Do not present assumptions as facts.
4. Keep the findings relevant to the user's question.
5. Make the recommendations practical.
6. Sources must be URLs that actually appear in the search results.
"""

    report = structured_model.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "question": question,
        "search_results": search_results,
        "report": report,
    }


# ---------------------------------
# 6. Build the LangGraph
# ---------------------------------

workflow = StateGraph(ResearchState)

workflow.add_node("research", research)
workflow.add_node("analyze", analyze)

workflow.add_edge(START, "research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", END)

app = workflow.compile()


# ---------------------------------
# 7. Run the agent
# ---------------------------------

def main():
    user_input = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the latest SEO trends for businesses in 2026?"
    )

    result = app.invoke({
        "question": user_input,
        "search_results": "",
        "report": None,
    })

    report = result["report"]

    print("\n" + "=" * 60)
    print("AI MARKETING RESEARCH REPORT")
    print("=" * 60)

    print("\nQUESTION")
    print(result["question"])

    print("\nSUMMARY")
    print(report.summary)

    print("\nKEY FINDINGS")
    for index, finding in enumerate(report.key_findings, start=1):
        print(f"{index}. {finding}")

    print("\nOPPORTUNITIES")
    for index, opportunity in enumerate(report.opportunities, start=1):
        print(f"{index}. {opportunity}")

    print("\nRECOMMENDATIONS")
    for index, recommendation in enumerate(
        report.recommendations,
        start=1,
    ):
        print(f"{index}. {recommendation}")

    print("\nSOURCES")
    for source in report.sources:
        print(f"- {source}")


if __name__ == "__main__":
    main()