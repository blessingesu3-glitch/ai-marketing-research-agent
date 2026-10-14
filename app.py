import os
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

load_dotenv()


# =====================================================================
# 1. Report structure returned to the API / frontend (UNCHANGED SHAPE)
# =====================================================================

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


# =====================================================================
# 2. Planner output: one focused search per part of the question
# =====================================================================

class ResearchTask(BaseModel):
    aspect: str = Field(
        description="Which part of the user's question this search answers, "
        "e.g. 'competitor offerings', 'published pricing', 'customer problems'."
    )
    query: str = Field(
        description="A short, specific web search query (under 15 words). "
        "Include the location and sector when the question names them."
    )


class ResearchPlan(BaseModel):
    target_location: Optional[str] = Field(
        default=None,
        description="The city/region/country the question is about, exactly as "
        "named in the question (e.g. 'Lagos, Nigeria'). Null if none is named.",
    )
    country: Optional[str] = Field(
        default=None,
        description="Lowercase English country name for that location "
        "(e.g. 'nigeria', 'kenya', 'united kingdom'). Null if none is named.",
    )
    tasks: List[ResearchTask] = Field(
        description="Distinct searches, one per part of the question. No duplicates."
    )


# =====================================================================
# 3. Analyst output: every claim carries its type and its source IDs
# =====================================================================

ClaimKind = Literal["evidence", "vendor_claim", "inference"]
ClaimScope = Literal["target_location", "other_or_unspecified"]
SourceType = Literal[
    "company_or_vendor_page",
    "independent_article_or_report",
    "official_or_statistics",
    "directory_listing_or_review",
    "forum_or_social",
    "other",
]


class Claim(BaseModel):
    text: str = Field(description="One specific claim, written as a single sentence.")
    kind: ClaimKind = Field(
        description="'evidence' = a source directly shows this; "
        "'vendor_claim' = a company says this about its own product/results; "
        "'inference' = your reasoning, not directly stated by a source."
    )
    scope: ClaimScope = Field(
        description="'target_location' only if the cited sources are about the "
        "location named in the question; otherwise 'other_or_unspecified'."
    )
    source_ids: List[str] = Field(
        default_factory=list,
        description="IDs like 'S1' of the sources that directly support this claim.",
    )


class Recommendation(BaseModel):
    text: str
    based_on_findings: List[int] = Field(
        default_factory=list,
        description="1-based numbers of the key findings this recommendation relies on.",
    )


class SourceAssessment(BaseModel):
    id: str
    relevant: bool = Field(description="Is this source about the user's question?")
    source_type: SourceType


class AnalysisOutput(BaseModel):
    summary: str = Field(
        description="3-5 sentences. Only state what the findings support."
    )
    findings: List[Claim] = Field(default_factory=list)
    opportunities: List[Claim] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    unknowns: List[str] = Field(
        default_factory=list,
        description="Parts of the question the sources could not answer.",
    )
    source_assessments: List[SourceAssessment] = Field(default_factory=list)


# =====================================================================
# 4. LangGraph state
# =====================================================================

class ResearchState(TypedDict, total=False):
    question: str
    plan: Dict[str, Any]
    sources: List[Dict[str, Any]]
    failed_queries: List[Dict[str, str]]
    search_results: str          # kept for API/CLI compatibility
    report: Optional[ResearchReport]


# =====================================================================
# 5. Configuration (all optional; defaults keep current cost profile)
# =====================================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


MAX_SEARCHES = _env_int("MAX_SEARCHES", 5)
MAX_RESULTS = _env_int("TAVILY_MAX_RESULTS", 5)
SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")          # "advanced" = 2 credits/search
INCLUDE_RAW = os.getenv("TAVILY_INCLUDE_RAW_CONTENT", "false").lower() == "true"
RAW_EXCERPT_CHARS = _env_int("RAW_EXCERPT_CHARS", 1500)
MAX_QUERY_CHARS = 380

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    temperature=0,
)

planner = model.with_structured_output(ResearchPlan)
analyst = model.with_structured_output(AnalysisOutput)

_search_tools: Dict[Optional[str], TavilySearch] = {}


def _get_search_tool(country: Optional[str]) -> TavilySearch:
    if country not in _search_tools:
        kwargs: Dict[str, Any] = {
            "max_results": MAX_RESULTS,
            "search_depth": SEARCH_DEPTH,
            "topic": "general",
        }
        if INCLUDE_RAW:
            kwargs["include_raw_content"] = "markdown"
        if country:
            kwargs["country"] = country
        _search_tools[country] = TavilySearch(**kwargs)
    return _search_tools[country]


def run_search(query: str, country: Optional[str] = None) -> Any:
    """The ONLY place a Tavily call is made. Tests replace this function."""
    return _get_search_tool(country).invoke({"query": query})


# =====================================================================
# 6. Search-result handling (no API calls)
# =====================================================================

def classify_search_result(result: Any) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Return (results, None) on success or (None, reason) on failure.

    langchain-tavily does NOT raise on failure: an API error comes back as
    {'error': ...} and an empty search comes back as a plain string.
    """
    if isinstance(result, str):
        return None, result[:200] or "empty response"
    if not isinstance(result, dict):
        return None, f"unexpected response type: {type(result).__name__}"
    if "error" in result:
        return None, f"search error: {str(result['error'])[:200]}"
    items = result.get("results")
    if not isinstance(items, list) or not items:
        return None, "no results"
    usable = [
        r for r in items
        if isinstance(r, dict) and r.get("url") and (r.get("content") or r.get("raw_content"))
    ]
    if not usable:
        return None, "results had no URL or content"
    return usable, None


_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|mc_)", re.I)


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)])
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower().removeprefix("www."), path, query, ""))


def _clip(text: Optional[str], limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def source_text(source: Dict[str, Any]) -> str:
    return " ".join([source.get("title", ""), source.get("content", ""), source.get("raw_excerpt", "")])


# =====================================================================
# 7. Nodes
# =====================================================================

def plan_research(state: ResearchState):
    question = state["question"]
    print("\n[Plan] Breaking the question into focused searches...")

    prompt = f"""You plan web searches for a marketing research question.

Question:
{question}

Split the question into its distinct parts (for example: market or demand
evidence, customer problems, competitors and what they offer, published prices,
channels, trends). Create ONE short search query per part the question actually
asks about, up to {MAX_SEARCHES} queries. Do not repeat the same search in
different words. Put the location and sector in each query when the question
names them. For prices, search for the providers' own pricing pages.
"""
    try:
        plan = planner.invoke([HumanMessage(content=prompt)])
        tasks, seen = [], set()
        for task in plan.tasks:
            key = re.sub(r"\s+", " ", task.query.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                tasks.append({"aspect": task.aspect, "query": task.query[:MAX_QUERY_CHARS]})
        tasks = tasks[:MAX_SEARCHES]
        plan_dict = {
            "target_location": plan.target_location,
            "country": (plan.country or "").strip().lower() or None,
            "tasks": tasks,
        }
    except Exception as exc:  # planner failure must not stop research
        print(f"[Plan] Planner failed ({type(exc).__name__}); using the question as one search.")
        plan_dict = {"target_location": None, "country": None, "tasks": []}

    if not plan_dict["tasks"]:
        plan_dict["tasks"] = [{"aspect": "general", "query": question[:MAX_QUERY_CHARS]}]

    return {"plan": plan_dict}


def research(state: ResearchState):
    plan = state.get("plan") or {
        "tasks": [{"aspect": "general", "query": state["question"][:MAX_QUERY_CHARS]}],
        "country": None,
    }
    country = plan.get("country")

    sources: List[Dict[str, Any]] = []
    by_url: Dict[str, Dict[str, Any]] = {}
    failed: List[Dict[str, str]] = []

    for task in plan["tasks"]:
        query = task["query"]
        print(f"[Research] Searching: {query}")
        try:
            raw = run_search(query, country)
        except Exception as exc:
            raw = {"error": exc}
        items, reason = classify_search_result(raw)

        # An unsupported country name makes Tavily reject the request.
        # Retry once without the country boost only in that case.
        if items is None and country and isinstance(raw, dict) and "error" in raw:
            try:
                raw = run_search(query, None)
            except Exception as exc:
                raw = {"error": exc}
            items, reason = classify_search_result(raw)

        if items is None:
            failed.append({"aspect": task["aspect"], "query": query, "reason": reason or "unknown"})
            continue

        for item in items:
            key = normalize_url(item["url"])
            if key in by_url:  # duplicate: remember the extra query, keep one record
                if query not in by_url[key]["queries"]:
                    by_url[key]["queries"].append(query)
                continue
            record = {
                "id": f"S{len(sources) + 1}",
                "aspect": task["aspect"],
                "queries": [query],
                "title": item.get("title") or "",
                "url": item["url"],
                "content": _clip(item.get("content"), 1200),
                "raw_excerpt": _clip(item.get("raw_content"), RAW_EXCERPT_CHARS),
                "score": item.get("score"),
            }
            by_url[key] = record
            sources.append(record)

    print(f"[Research] {len(sources)} unique sources, {len(failed)} failed searches.")
    return {
        "sources": sources,
        "failed_queries": failed,
        "search_results": format_sources_for_prompt(sources),
    }


def route_after_research(state: ResearchState) -> str:
    return "analyze" if state.get("sources") else "insufficient_evidence"


def insufficient_evidence(state: ResearchState):
    failed = state.get("failed_queries") or []
    total = len((state.get("plan") or {}).get("tasks") or []) or len(failed)
    print("[Analysis] Skipped: no usable sources.")
    report = ResearchReport(
        summary=(
            f"The research could not be completed: none of the {total} web searches "
            "returned usable sources. No findings are reported, so that nothing "
            "unsupported is presented as fact."
        ),
        key_findings=[
            f"[Not established] Search for '{f['aspect']}' returned nothing usable ({f['reason']})."
            for f in failed
        ] or ["[Not established] No sources were retrieved."],
        opportunities=[],
        recommendations=[
            "Try again in a few minutes. If it keeps failing, check the search "
            "service quota or rephrase the question more specifically."
        ],
        sources=[],
    )
    return {"report": report}


def format_sources_for_prompt(sources: List[Dict[str, Any]]) -> str:
    blocks = []
    for s in sources:
        block = (
            f"[{s['id']}] {s['title']}\n"
            f"URL: {s['url']}\n"
            f"Found by search for: {s['aspect']} ({'; '.join(s['queries'])})\n"
            f"Snippet: {s['content']}"
        )
        if s.get("raw_excerpt"):
            block += f"\nPage excerpt: {s['raw_excerpt']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def analyze(state: ResearchState):
    question = state["question"]
    sources = state["sources"]
    plan = state.get("plan") or {}
    location = plan.get("target_location") or "none named"

    print("[Analysis] Analyzing the research...")

    failed_note = ""
    if state.get("failed_queries"):
        failed_note = "\nThese searches FAILED, so treat their topics as not researched:\n" + "\n".join(
            f"- {f['aspect']}: {f['query']}" for f in state["failed_queries"]
        )

    prompt = f"""You are a marketing research analyst. Answer the question using ONLY
the numbered sources below.

Question:
{question}

Target location: {location}
{failed_note}

Sources:
{format_sources_for_prompt(sources)}

How to answer:
- Each finding is one specific claim with the IDs of the sources that directly state it.
- If a company says something about its own product or results, kind = 'vendor_claim'.
- A competitor offering a service shows supply, not customer demand.
- Only copy numbers and prices that appear in the cited source text.
- If a part of the question is not answered by the sources, add it to 'unknowns'
  (for example: "No published prices were found for <company>").
- Opportunities may be inferences, but must say what they are based on.
- Recommendations must list the finding numbers they rely on.
- Assess every source: is it relevant, and what type is it.
- Fewer, well-supported findings are better than many weak ones. Empty lists are allowed.
"""

    analysis = analyst.invoke([HumanMessage(content=prompt)])
    report = validate_and_render(analysis, sources, question, plan.get("target_location"))
    return {"report": report}


# =====================================================================
# 8. Deterministic validation (no API calls) — enforces the rules in code
# =====================================================================

_STOPWORDS = set("""
a an and are as at be been but by can for from has have how in into is it its
more most not of on or our that the their them there these they this to was
were what when which who will with without your you also than then very such
business businesses company companies service services market marketing
""".split())

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|x|×|k|m|bn|million|billion)?", re.I)
_CURRENCY = re.compile(
    r"(₦|\$|£|€|¥|₹|ngn|usd|gbp|eur|kes|ghs|zar|naira|dollars?|pounds?)\s?\d|\d[\d,.]*\s?(₦|ngn|usd|naira|dollars?|per month|/month|/mo)",
    re.I,
)
_PRICE_WORDS = re.compile(r"\b(pric\w*|cost\w*|fee\w*|rates?|charges?|subscriptions?|plans?)\b", re.I)

_COUNTRY_TLDS = {
    "nigeria": ".ng", "ghana": ".gh", "kenya": ".ke", "south africa": ".za",
    "united kingdom": ".uk", "uk": ".uk", "india": ".in", "canada": ".ca",
    "australia": ".au", "egypt": ".eg", "rwanda": ".rw", "uganda": ".ug",
}


def _numbers(text: str) -> List[str]:
    out = []
    for value, _unit in _NUMBER.findall(text or ""):
        out.append(value.replace(",", "").rstrip("0").rstrip(".") if "." in value else value.replace(",", ""))
    return out


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9'-]{2,}", (text or "").lower()) if w not in _STOPWORDS}


def _location_terms(target_location: Optional[str]) -> List[str]:
    if not target_location:
        return []
    parts = re.split(r"[,/]| and ", target_location.lower())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def mentions_location(source: Dict[str, Any], terms: List[str]) -> bool:
    if not terms:
        return True
    haystack = (source_text(source) + " " + source.get("url", "")).lower()
    host = urlsplit(source.get("url", "")).netloc.lower()
    for term in terms:
        if term in haystack:
            return True
        tld = _COUNTRY_TLDS.get(term)
        if tld and (host.endswith(tld) or tld + "/" in host):
            return True
    return False


def _supports(claim_text: str, source: Dict[str, Any]) -> bool:
    """A cited source must share real vocabulary with the claim (min. 2 content words)."""
    shared = _tokens(claim_text) & _tokens(source_text(source))
    return len(shared) >= 2


def _figures_found(claim_text: str, sources: List[Dict[str, Any]]) -> bool:
    figures = _numbers(claim_text)
    if not figures:
        return True
    pool = " ".join(source_text(s) for s in sources)
    pool_numbers = set(_numbers(pool))
    return all(f in pool_numbers for f in figures)


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def check_claim(
    claim: Claim,
    sources_by_id: Dict[str, Dict[str, Any]],
    assessments: Dict[str, SourceAssessment],
    location_terms: List[str],
) -> Dict[str, Any]:
    """Return a checked claim: {'text','label','sources','status'}.

    status: 'ok' | 'downgraded' | 'rejected_figure'
    """
    notes = []
    cited = []
    for sid in dict.fromkeys(s.strip().upper() for s in claim.source_ids):
        src = sources_by_id.get(sid)
        if src is None:
            notes.append(f"{sid} not retrieved")
            continue
        a = assessments.get(sid)
        if a is not None and not a.relevant:
            notes.append(f"{sid} irrelevant")
            continue
        if not _supports(claim.text, src):
            notes.append(f"{sid} does not mention this")
            continue
        cited.append(src)

    kind = claim.kind
    scope = claim.scope
    status = "ok"

    # Numbers must appear in the cited sources, otherwise the claim is not shown as fact.
    if _numbers(claim.text) and not _figures_found(claim.text, cited):
        return {"text": claim.text, "label": "Not established", "sources": [], "status": "rejected_figure"}

    if not cited:
        if kind != "inference":
            status = "downgraded"
        kind = "inference"
    else:
        types = {
            assessments[s["id"]].source_type for s in cited if s["id"] in assessments
        }
        if kind == "evidence" and types and types <= {"company_or_vendor_page"}:
            kind, status = "vendor_claim", "downgraded"

    if scope == "target_location" and location_terms:
        if not cited or not any(mentions_location(s, location_terms) for s in cited):
            scope, status = "other_or_unspecified", "downgraded"

    if kind == "inference":
        label = "Inference"
    elif kind == "vendor_claim":
        label = "Vendor claim"
    elif scope == "target_location" or not location_terms:
        label = "Evidence"
    else:
        label = "Evidence – not specific to target location"

    return {"text": claim.text, "label": label, "sources": cited, "status": status, "notes": notes}


def _render_claim(checked: Dict[str, Any]) -> str:
    if checked["status"] == "rejected_figure":
        return (
            f"[Not established] {checked['text']} — this figure was not found in "
            "the cited sources, so it is not presented as fact."
        )
    line = f"[{checked['label']}] {checked['text']}"
    if checked["sources"]:
        refs = "; ".join(f"{s['id']} {s['url']}" for s in checked["sources"])
        line += f" (Source: {refs})"
    return line


def _clean_summary(summary: str, sources: List[Dict[str, Any]], question: str) -> str:
    """Drop summary sentences containing figures found in neither the sources nor the question."""
    allowed = set(_numbers(" ".join(source_text(s) for s in sources) + " " + question))
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", (summary or "").strip()):
        if all(n in allowed for n in _numbers(sentence)):
            kept.append(sentence)
    return " ".join(kept).strip()


def validate_and_render(
    analysis: AnalysisOutput,
    sources: List[Dict[str, Any]],
    question: str,
    target_location: Optional[str],
) -> ResearchReport:
    sources_by_id = {s["id"]: s for s in sources}
    assessments = {a.id.strip().upper(): a for a in analysis.source_assessments}
    terms = _location_terms(target_location)

    checked_findings = [check_claim(c, sources_by_id, assessments, terms) for c in analysis.findings]
    checked_opps = [check_claim(c, sources_by_id, assessments, terms) for c in analysis.opportunities]

    key_findings = [_render_claim(c) for c in checked_findings]
    opportunities = [_render_claim(c) for c in checked_opps]

    # Findings that survived as sourced evidence/vendor claims can back recommendations.
    supported = {
        i + 1 for i, c in enumerate(checked_findings)
        if c["sources"] and c["status"] != "rejected_figure"
    }
    recommendations = []
    for rec in analysis.recommendations:
        basis = sorted({n for n in rec.based_on_findings if n in supported})
        if basis:
            recommendations.append(f"{rec.text} (Based on finding{'s' if len(basis) > 1 else ''} {', '.join(map(str, basis))})")
        else:
            recommendations.append(f"[Untested idea] {rec.text} (Not backed by a sourced finding — validate first)")

    unknowns = list(dict.fromkeys(u.strip() for u in analysis.unknowns if u and u.strip()))

    # Honest pricing gap: question asks about prices but no source shows one.
    if _PRICE_WORDS.search(question) and not any(_CURRENCY.search(source_text(s)) for s in sources):
        unknowns.append("No published prices were found in the retrieved sources.")

    key_findings += [f"[Not established] {u}" for u in unknowns]

    cited_ids = []
    for c in checked_findings + checked_opps:
        for s in c["sources"]:
            if s["id"] not in cited_ids:
                cited_ids.append(s["id"])
    cited_urls = [sources_by_id[i]["url"] for i in sorted(cited_ids, key=lambda x: int(x[1:]))]

    summary = _clean_summary(analysis.summary, sources, question)
    n_ok = sum(1 for c in checked_findings if c["sources"] and c["status"] != "rejected_figure")
    summary_note = (
        f" Evidence check: {n_ok} of {len(checked_findings)} findings are backed by cited sources"
        f"{f'; {len(unknowns)} points could not be established' if unknowns else ''}."
    )
    summary = (summary or "The sources did not support a confident summary.") + summary_note

    return ResearchReport(
        summary=summary,
        key_findings=key_findings,
        opportunities=opportunities,
        recommendations=recommendations,
        sources=cited_urls,
    )


# =====================================================================
# 9. Build the LangGraph
# =====================================================================

workflow = StateGraph(ResearchState)

workflow.add_node("plan_research", plan_research)
workflow.add_node("research", research)
workflow.add_node("analyze", analyze)
workflow.add_node("insufficient_evidence", insufficient_evidence)

workflow.add_edge(START, "plan_research")
workflow.add_edge("plan_research", "research")
workflow.add_conditional_edges(
    "research",
    route_after_research,
    {"analyze": "analyze", "insufficient_evidence": "insufficient_evidence"},
)
workflow.add_edge("analyze", END)
workflow.add_edge("insufficient_evidence", END)

app = workflow.compile()


# =====================================================================
# 10. Run the agent from the command line
# =====================================================================

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
    for index, recommendation in enumerate(report.recommendations, start=1):
        print(f"{index}. {recommendation}")

    print("\nSOURCES")
    for source in report.sources:
        print(f"- {source}")


if __name__ == "__main__":
    main()
