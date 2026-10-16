import os
import re
import sys
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

load_dotenv()

# langchain-openai structured output triggers a harmless pydantic serializer warning.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


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
# 1b. Optional business profile (saved in the user's browser, sent per request)
# =====================================================================

class BusinessProfile(BaseModel):
    business_name: Optional[str] = Field(default=None, max_length=120)
    industry: Optional[str] = Field(default=None, max_length=120)
    offering: Optional[str] = Field(default=None, max_length=300, description="What the business sells")
    location: Optional[str] = Field(default=None, max_length=120, description="City, country or markets served")
    target_customer: Optional[str] = Field(default=None, max_length=300)
    goal: Optional[str] = Field(default=None, max_length=300)
    competitors: Optional[str] = Field(default=None, max_length=500, description="Names or websites, comma-separated")
    price_point: Optional[str] = Field(default=None, max_length=120)


_PROFILE_LABELS = [
    ("business_name", "Business"),
    ("industry", "Industry"),
    ("offering", "Sells"),
    ("location", "Location / markets"),
    ("target_customer", "Target customer"),
    ("goal", "Marketing goal"),
    ("competitors", "Known competitors"),
    ("price_point", "Price point"),
]


def clean_profile(profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Keep only filled-in fields. Returns None when nothing useful was given."""
    if not profile:
        return None
    cleaned = {
        key: " ".join(str(profile.get(key) or "").split())
        for key, _label in _PROFILE_LABELS
    }
    cleaned = {k: v for k, v in cleaned.items() if v}
    return cleaned or None


def profile_block(profile: Optional[Dict[str, str]]) -> str:
    if not profile:
        return ""
    lines = [f"- {label}: {profile[key]}" for key, label in _PROFILE_LABELS if profile.get(key)]
    return "\n".join(lines)


def competitor_names(profile: Optional[Dict[str, str]]) -> List[str]:
    """'Acme, https://www.beta.com; Gamma' -> ['Acme', 'beta.com', 'Gamma']"""
    if not profile or not profile.get("competitors"):
        return []
    names = []
    for part in re.split(r"[,;\n]| and ", profile["competitors"]):
        part = part.strip()
        if not part:
            continue
        if "." in part and " " not in part:          # a website: keep the domain
            part = re.sub(r"^https?://", "", part).removeprefix("www.").split("/")[0]
        names.append(part)
    return list(dict.fromkeys(names))[:5]


def country_from_location(location: Optional[str]) -> Optional[str]:
    """Best-effort country for Tavily's country boost, from a free-text location."""
    text = (location or "").lower()
    for country in _COUNTRY_TLDS:
        if len(country) > 2 and re.search(rf"\b{re.escape(country)}\b", text):
            return country
    return None


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
    offering: Optional[str] = Field(
        default=None,
        description="The product or service the user wants to market, in 2-5 words "
        "(e.g. 'WhatsApp chatbot', 'solar panel installation'). Null if unclear.",
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
    text: str = Field(
        description="2-3 sentences: what the source shows (names, offers, prices, numbers "
        "exactly as the source gives them) and why it matters for the user's goal."
    )
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
    text: str = Field(
        description="2-4 sentences: the action, who it is for, and the reasoning from the findings."
    )
    based_on_findings: List[int] = Field(
        default_factory=list,
        description="1-based numbers of the key findings this recommendation relies on.",
    )


class SourceAssessment(BaseModel):
    id: str
    relevant: bool = Field(description="Is this source about the user's question?")
    source_type: SourceType


class AspectCoverage(BaseModel):
    aspect: str = Field(description="One planned research aspect, copied exactly.")
    answering_findings: List[int] = Field(
        default_factory=list,
        description="1-based numbers of the key findings that directly answer this aspect. "
        "Empty if no finding answers it.",
    )


class AnalysisOutput(BaseModel):
    summary: str = Field(
        default="",
        description="Executive summary, 3-5 short paragraphs separated by blank lines "
        "(about 250-450 words). Cover every part of the question, compare the options, "
        "and end with the main recommendation and its reasoning.",
    )
    findings: List[Claim] = Field(default_factory=list)
    opportunities: List[Claim] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    unknowns: List[str] = Field(
        default_factory=list,
        description="Parts of the question the sources could not answer.",
    )
    source_assessments: List[SourceAssessment] = Field(default_factory=list)
    aspect_coverage: List[AspectCoverage] = Field(default_factory=list)
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="Up to 3 short web searches that would fill the most important gaps. "
        "Empty if the sources already answer the question well.",
    )


# =====================================================================
# 4. LangGraph state
# =====================================================================

class ResearchState(TypedDict, total=False):
    question: str
    profile: Optional[Dict[str, str]]     # optional business profile (user-provided context)
    plan: Dict[str, Any]
    sources: List[Dict[str, Any]]
    failed_queries: List[Dict[str, str]]
    search_results: str          # kept for API/CLI compatibility
    report: Optional[ResearchReport]
    pending_tasks: List[Dict[str, str]]   # follow-up searches waiting to run
    round: int                            # 0 = first pass, 1 = after follow-up searches


# =====================================================================
# 5. Configuration (all optional; defaults keep current cost profile)
# =====================================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


MAX_SEARCHES = _env_int("MAX_SEARCHES", 6)
MAX_RESULTS = _env_int("TAVILY_MAX_RESULTS", 8)
SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")          # "advanced" = 2 credits/search
# Read page text, not just the 1-2 sentence snippet. This is what gives the analysis depth.
INCLUDE_RAW = os.getenv("TAVILY_INCLUDE_RAW_CONTENT", "true").lower() == "true"
RAW_EXCERPT_CHARS = _env_int("RAW_EXCERPT_CHARS", 1500)
# The best-scoring sources sent to the analyst (keeps the prompt size predictable).
MAX_SOURCES_FOR_ANALYSIS = _env_int("MAX_SOURCES_FOR_ANALYSIS", 24)
# Extra search rounds for gaps: 1 = up to 3 more searches + 1 more analysis call when needed.
FOLLOW_UP_ROUNDS = _env_int("FOLLOW_UP_ROUNDS", 1)
MAX_FOLLOW_UP_SEARCHES = 3
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
    profile = clean_profile(state.get("profile"))
    print("\n[Plan] Breaking the question into focused searches..."
          + (" (tailored to the business profile)" if profile else ""))

    profile_note = ""
    if profile:
        profile_note = f"""
The user's business (use it to make the searches specific: their industry, what
they sell, their location and their customers; if the question names no location,
use theirs):
{profile_block(profile)}
"""

    prompt = f"""You plan web searches for a marketing research question.

Question:
{question}
{profile_note}
Split the question into the distinct parts it ACTUALLY asks about. Do not add
topics the question does not ask about. A broad question may have several parts
(e.g. market, customer problems, competitors, prices, segments). A narrow question
(e.g. "which keywords should I target") has one part: then search that one topic
from different angles (e.g. keyword lists, search trends, competitor keywords,
customer language). Up to {MAX_SEARCHES} queries. Do not repeat the same search in
different words.

Write each query the way the pages you want are worded, not the way the
question is worded. Examples of the style (adapt to the actual sector/place):
- providers:      "<service> provider <country>" / "<service> company <city>"
- prices:         "<service> pricing <country> per month" / "<service> price list <currency>"
- local market:   "<business type> <city> book appointment" / "<business type> <city> reviews"
- customer needs: "<business type> owners complain <problem>" / "<problem> <business type> <country>"
Avoid abstract phrases like "opportunities for" or "customer segments for".
Every part the question explicitly asks about needs its own query. If it asks
about competitors or providers, include a query that finds the providers'
own websites. If it asks about prices, include a query for pricing pages.
Put the location in each query when the question names one.
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
        location = plan.target_location or (profile or {}).get("location")
        country = (plan.country or "").strip().lower() or country_from_location(location)
        offering = plan.offering or (profile or {}).get("offering")
        tasks = ensure_required_searches(question, tasks, offering, country or location)
        tasks = ensure_named_competitor_search(question, tasks, competitor_names(profile), offering)
        plan_dict = {
            "target_location": location,
            "country": country,
            "offering": offering,
            "tasks": tasks,
        }
    except Exception as exc:  # planner failure must not stop research
        print(f"[Plan] Planner failed ({type(exc).__name__}); using the question as one search.")
        location = (profile or {}).get("location")
        plan_dict = {
            "target_location": location,
            "country": country_from_location(location),
            "offering": (profile or {}).get("offering"),
            "tasks": [],
        }

    if not plan_dict["tasks"]:
        plan_dict["tasks"] = [{"aspect": "general", "query": question[:MAX_QUERY_CHARS]}]

    return {"plan": plan_dict}


# Parts of a question that must get their own search when the question asks for them.
# (question pattern, pattern that shows a query already covers it, aspect, query template)
_REQUIRED_SEARCHES = [
    (re.compile(r"\b(competitor\w*|rivals?|providers?|agencies|alternatives)\b", re.I),
     re.compile(r"\b(competitor\w*|provider\w*|compan\w*|agenc\w*|vendor\w*|alternatives?)\b", re.I),
     "competitors", "{offering} provider {place}"),
    (re.compile(r"\b(pric\w*|cost\w*|fees?|charges?|rates)\b", re.I),
     re.compile(r"\b(pric\w*|cost\w*|fees?|rates?|per month|plans?)\b", re.I),
     "published pricing", "{offering} pricing {place} per month"),
]


def ensure_required_searches(
    question: str, tasks: List[Dict[str, str]], offering: Optional[str], place: Optional[str]
) -> List[Dict[str, str]]:
    """Guarantee a competitor / pricing search when the question asks for one.

    Keeps the total at MAX_SEARCHES by replacing the planner's last searches,
    so this never increases the number of paid Tavily calls.
    """
    offering = (offering or "").strip()
    if not offering:
        return tasks
    place = (place or "").strip()
    tasks = list(tasks)
    added = 0
    for asked, covered, aspect, template in _REQUIRED_SEARCHES:
        if not asked.search(question):
            continue
        if any(covered.search(t["query"]) or covered.search(t["aspect"]) for t in tasks):
            continue
        new = {"aspect": aspect, "query": " ".join(template.format(offering=offering, place=place).split())}
        if len(tasks) >= MAX_SEARCHES:
            replace_at = len(tasks) - 1 - added
            if replace_at < 0:
                continue
            tasks[replace_at] = new
        else:
            tasks.append(new)
        added += 1
    return tasks


_ASKS_ABOUT_COMPETITORS = re.compile(
    r"\b(competitor\w*|competition|rivals?|alternatives?|compare|comparison|vs\.?|versus|positioning)\b", re.I
)


def ensure_named_competitor_search(
    question: str, tasks: List[Dict[str, str]], names: List[str], offering: Optional[str]
) -> List[Dict[str, str]]:
    """If the question is about competitors and the profile names some, search for them by name.

    Replaces the planner's last search when at MAX_SEARCHES, so the cost does not grow.
    """
    if not names or not _ASKS_ABOUT_COMPETITORS.search(question):
        return tasks
    already = " ".join(t["query"].lower() for t in tasks)
    missing = [n for n in names if n.lower() not in already]
    if not missing:
        return tasks
    query = " ".join((" ".join(missing[:3]) + " " + (offering or "") + " pricing services").split())
    new = {"aspect": "named competitors", "query": query[:MAX_QUERY_CHARS]}
    tasks = list(tasks)
    if len(tasks) >= MAX_SEARCHES:
        tasks[-1] = new
    else:
        tasks.append(new)
    return tasks


def research(state: ResearchState):
    plan = state.get("plan") or {
        "tasks": [{"aspect": "general", "query": state["question"][:MAX_QUERY_CHARS]}],
        "country": None,
    }
    country = plan.get("country")

    # Follow-up round: run only the pending searches and ADD to the sources already found.
    follow_up = bool(state.get("pending_tasks"))
    tasks = state.get("pending_tasks") or plan["tasks"]
    if follow_up:
        sources: List[Dict[str, Any]] = [dict(s, queries=list(s["queries"])) for s in state.get("sources") or []]
        failed: List[Dict[str, str]] = list(state.get("failed_queries") or [])
        print(f"[Research] Follow-up: {len(tasks)} more searches for gaps.")
    else:
        sources, failed = [], []
    by_url: Dict[str, Dict[str, Any]] = {normalize_url(s["url"]): s for s in sources}

    for task in tasks:
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
        "pending_tasks": [],
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


def select_sources_for_analysis(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The best-scoring sources, in their original order (keeps the prompt size predictable)."""
    if len(sources) <= MAX_SOURCES_FOR_ANALYSIS:
        return sources
    ranked = sorted(sources, key=lambda s: -(s.get("score") or 0))[:MAX_SOURCES_FOR_ANALYSIS]
    keep = {s["id"] for s in ranked}
    return [s for s in sources if s["id"] in keep]


def follow_up_tasks(
    analysis: "AnalysisOutput", gaps: List[str], state: ResearchState
) -> List[Dict[str, str]]:
    """Up to MAX_FOLLOW_UP_SEARCHES new searches for the gaps (no extra model call)."""
    plan = state.get("plan") or {}
    used = {re.sub(r"\s+", " ", q.lower()).strip()
            for s in state.get("sources") or [] for q in s["queries"]}
    used |= {re.sub(r"\s+", " ", t["query"].lower()).strip() for t in plan.get("tasks", [])}

    candidates = [("follow-up", q) for q in analysis.follow_up_queries]
    offering = (plan.get("offering") or "").strip()
    place = (plan.get("country") or plan.get("target_location") or "").strip()
    candidates += [(g, " ".join(f"{g} {offering} {place}".split())) for g in gaps]

    tasks = []
    for aspect, query in candidates:
        key = re.sub(r"\s+", " ", (query or "").lower()).strip()
        if key and key not in used:
            used.add(key)
            tasks.append({"aspect": aspect, "query": query[:MAX_QUERY_CHARS]})
        if len(tasks) >= MAX_FOLLOW_UP_SEARCHES:
            break
    return tasks


def analyze(state: ResearchState):
    question = state["question"]
    sources = select_sources_for_analysis(state["sources"])
    plan = state.get("plan") or {}
    location = plan.get("target_location") or "none named"
    round_no = state.get("round") or 0
    profile = clean_profile(state.get("profile"))
    profile_note = ""
    if profile:
        profile_note = f"""
About the user's own business (written by the user). Use it to TAILOR the analysis,
comparisons and recommendations to this business. It is NOT evidence: never cite
it as a source, and never present it as a research finding.
{profile_block(profile)}
"""

    print("[Analysis] Analyzing the research..." + (" (after follow-up searches)" if round_no else ""))

    failed_note = ""
    if state.get("failed_queries"):
        failed_note = "\nThese searches FAILED, so treat their topics as not researched:\n" + "\n".join(
            f"- {f['aspect']}: {f['query']}" for f in state["failed_queries"]
        )

    aspect_list = "\n".join(
        f"  - {a}" for a in dict.fromkeys(t["aspect"] for t in plan.get("tasks", []))
    ) or "  - (none)"

    prompt = f"""You are a senior marketing research analyst writing a thorough, specific
report for a client. Use ONLY the numbered sources below.

Question:
{question}

Target location: {location}
{profile_note}{failed_note}

Sources:
{format_sources_for_prompt(sources)}

How to write the report:
- Cover EVERY part of the question in depth. Use every relevant source; when the
  sources allow, aim for 8-15 findings rather than a few general ones.
- Each finding is 2-3 sentences: the specifics (company names, what they offer,
  who they serve, channels, prices and figures exactly as the source gives them)
  and why it matters for the user's goal. Cite the IDs of the sources that state it.
- When several companies or options appear, give each its own finding so they can
  be compared (offer, target customer, channel, published price or "no published price").
- If a company describes its own product or results, kind = 'vendor_claim' and write
  it as "<Company> says ...". A competitor offering a service shows supply, not demand.
- Only use numbers and prices that appear in the cited source text. A price claim
  covers ONE provider and cites only the page that shows that price.
- A company's own website, blog or social post about its product is 'company_or_vendor_page'.
- Opportunities (3-6): each explains which findings point to it.
- Recommendations (3-5): concrete actions with the reasoning, listing the finding
  numbers they rely on. If a business profile is given, every recommendation must
  fit that business (its location, customers, price point and goal), and named
  competitors should be compared directly with it.
- Summary: an executive summary of 3-5 short paragraphs separated by blank lines
  (about 250-450 words). Specific, no filler. Explain what the evidence shows,
  compare the options, say what is uncertain, and end with the main recommendation.
  Attribute company statements ("X says"). Only use numbers that appear in your findings.
- unknowns: the parts of the question the sources could not answer.
- follow_up_queries: up to 3 short searches, worded like the pages you want to find,
  that would fill the most important gaps. Empty if nothing important is missing.
- Assess every source: is it relevant, and what type is it.
- For each planned aspect below, list the finding numbers that directly answer it
  (empty if none):
{aspect_list}
"""

    analysis = analyst.invoke([HumanMessage(content=prompt)])
    report, gaps = build_report(
        analysis,
        sources,
        question,
        plan.get("target_location"),
        planned_aspects=[t["aspect"] for t in plan.get("tasks", [])],
        failed_aspects=[f["aspect"] for f in state.get("failed_queries") or []],
    )

    if round_no < FOLLOW_UP_ROUNDS and (gaps or analysis.follow_up_queries):
        tasks = follow_up_tasks(analysis, gaps, state)
        if tasks:
            return {"pending_tasks": tasks, "round": round_no + 1}

    return {"report": report}


def route_after_analyze(state: ResearchState) -> str:
    return "research" if state.get("pending_tasks") else "done"


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

# Social platforms and forums: useful signals, but anecdotal and often self-promotion.
_SOCIAL_HOSTS = (
    "linkedin.com", "youtube.com", "youtu.be", "facebook.com", "fb.com",
    "instagram.com", "tiktok.com", "x.com", "twitter.com", "threads.net",
    "nairaland.com", "reddit.com", "quora.com", "pinterest.com",
)
# Sites where anyone can upload a document (business plans, essays, slides).
_UPLOAD_HOSTS = (
    "scribd.com", "slideshare.net", "academia.edu", "issuu.com", "studocu.com",
    "coursehero.com", "docplayer.net", "pdfcoffee.com", "dokumen.pub",
    "documents.pub", "yumpu.com", "pdfslide.net",
)
# URL paths that are almost always a company describing its own product.
_VENDOR_PATH = re.compile(
    r"/(services?|pricing|prices|plans|features|products?|solutions|demo|book-a-demo)(/|\.|-|$)", re.I
)
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac", "ltd", "plc"}


def _numbers(text: str) -> List[str]:
    out = []
    for value, _unit in _NUMBER.findall(text or ""):
        out.append(value.replace(",", "").rstrip("0").rstrip(".") if "." in value else value.replace(",", ""))
    return out


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9'-]{2,}", (text or "").lower()) if w not in _STOPWORDS}


def _host(url: str) -> str:
    return urlsplit(url or "").netloc.lower().split(":")[0].removeprefix("www.")


def _host_label(url: str) -> str:
    """'musskart.com' -> 'musskart'; 'shop.brand.com.ng' -> 'brand'."""
    parts = _host(url).split(".")
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL:
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else parts[0]


def is_social(source: Dict[str, Any]) -> bool:
    host = _host(source.get("url", ""))
    return any(host == h or host.endswith("." + h) for h in _SOCIAL_HOSTS)


def is_uploaded(source: Dict[str, Any]) -> bool:
    host = _host(source.get("url", ""))
    return any(host == h or host.endswith("." + h) for h in _UPLOAD_HOSTS)


def is_vendor(source: Dict[str, Any], claim_text: str, assessments: Dict[str, "SourceAssessment"]) -> bool:
    a = assessments.get(source["id"])
    if a is not None and a.source_type == "company_or_vendor_page":
        return True
    if _VENDOR_PATH.search(urlsplit(source.get("url", "")).path):
        return True
    label = _host_label(source.get("url", ""))
    # The claim names the company whose own website is being cited.
    return len(label) >= 4 and label in re.sub(r"[^a-z0-9]", "", claim_text.lower())


def _location_levels(target_location: Optional[str]) -> List[Tuple[str, List[str]]]:
    """'Abuja, Nigeria' -> [('Abuja', ['abuja']), ('Nigeria', ['nigeria'])], most specific first."""
    if not target_location:
        return []
    levels = []
    for part in re.split(r",|/|\band\b", target_location):
        name = part.strip()
        if len(name) >= 2:
            levels.append((name, [name.lower()]))
    return levels


def location_level(source: Dict[str, Any], levels: List[Tuple[str, List[str]]]) -> Optional[int]:
    """Index of the most specific location level the source mentions, or None."""
    haystack = (source_text(source) + " " + source.get("url", "")).lower()
    host = _host(source.get("url", ""))
    for index, (_name, terms) in enumerate(levels):
        for term in terms:
            if term in haystack:
                return index
            tld = _COUNTRY_TLDS.get(term)
            if tld and host.endswith(tld):
                return index
    return None


def matched_levels(source: Dict[str, Any], levels: List[Tuple[str, List[str]]]) -> set:
    """All location levels a source mentions (by text or country web domain)."""
    haystack = (source_text(source) + " " + source.get("url", "")).lower()
    host = _host(source.get("url", ""))
    found = set()
    for index, (_name, terms) in enumerate(levels):
        for term in terms:
            tld = _COUNTRY_TLDS.get(term)
            if term in haystack or (tld and host.endswith(tld)):
                found.add(index)
    return found


def claim_location_level(
    claim_text_value: str, cited: List[Dict[str, Any]], levels: List[Tuple[str, List[str]]]
) -> Optional[int]:
    """How specific the evidence for THIS claim is.

    For a city-level question (e.g. 'Abuja, Nigeria') the claim only counts as
    city evidence when the claim itself is about the city AND a cited source
    mentions it. A country-wide article that merely mentions the city somewhere
    counts as country-wide evidence.
    """
    if not levels or not cited:
        return None
    per_source = [matched_levels(s, levels) for s in cited]
    if not any(per_source):
        return None
    if len(levels) == 1:
        return 0
    claim_lower = claim_text_value.lower()
    claim_is_local = any(term in claim_lower for term in levels[0][1])
    if claim_is_local and any(0 in m for m in per_source):
        return 0
    wider = [lvl for m in per_source for lvl in m if lvl >= 1]
    return min(wider) if wider else 1   # a source naming the city is at least country-level


# Kept for backwards compatibility with earlier callers/tests.
def mentions_location(source: Dict[str, Any], terms: List[str]) -> bool:
    levels = [(t, [t]) for t in terms]
    return not terms or location_level(source, levels) is not None


def _supports(claim_text: str, source: Dict[str, Any]) -> bool:
    """A cited source must share real vocabulary with the claim (min. 2 content words)."""
    shared = _tokens(claim_text) & _tokens(source_text(source))
    return len(shared) >= 2


def _figures_in(figures: List[str], sources: List[Dict[str, Any]]) -> bool:
    pool = set(_numbers(" ".join(source_text(s) for s in sources)))
    return all(f in pool for f in figures)


def _location_suffix(best: Optional[int], levels: List[Tuple[str, List[str]]]) -> str:
    if not levels or best == 0:
        return ""
    if best is None:
        return " – not specific to target location"
    return f" – {levels[best][0]}-wide, not {levels[0][0]}-specific"


def check_claim(
    claim: Claim,
    sources_by_id: Dict[str, Dict[str, Any]],
    assessments: Dict[str, SourceAssessment],
    levels: List[Tuple[str, List[str]]],
) -> Dict[str, Any]:
    """Return a checked claim.

    status:   'ok' | 'downgraded' | 'rejected_figure' | 'rejected_price_mix'
    strength: 'strong' (target-location evidence) | 'medium' (wider-area evidence)
              | 'weak' (vendor, social, other region, or no source)
    """
    cited = []
    for sid in dict.fromkeys(s.strip().upper() for s in claim.source_ids):
        src = sources_by_id.get(sid)
        if src is None:
            continue                                   # not retrieved
        a = assessments.get(sid)
        if a is not None and not a.relevant:
            continue                                   # marked irrelevant
        if not _supports(claim.text, src):
            continue                                   # does not mention the claim
        cited.append(src)

    base = {"text": claim.text, "sources": [], "strength": "weak"}
    figures = _numbers(claim.text)

    # Prices must be attributable to one provider's source.
    if _CURRENCY.search(claim.text) and len(cited) > 1:
        return {**base, "label": "Not established", "status": "rejected_price_mix"}
    # Every figure must appear in the cited source text.
    if figures and not _figures_in(figures, cited):
        return {**base, "label": "Not established", "status": "rejected_figure"}

    kind = claim.kind
    status = "ok"
    if not cited:
        if kind != "inference":
            status = "downgraded"
        kind = "inference"
    elif all(is_uploaded(s) for s in cited):
        if kind != "inference":
            kind, status = "uploaded", "downgraded"
    elif all(is_social(s) or is_uploaded(s) for s in cited):
        if kind != "inference":
            kind, status = "social", "downgraded"
    elif kind == "evidence" and all(is_vendor(s, claim.text, assessments) for s in cited):
        kind, status = "vendor_claim", "downgraded"

    best = claim_location_level(claim.text, cited, levels)
    suffix = _location_suffix(best, levels)

    if kind == "inference":
        label, strength = "Inference", "weak"
    elif kind == "social":
        label, strength = "Social post" + suffix, "weak"
    elif kind == "uploaded":
        label, strength = "Uploaded document" + suffix, "weak"
    elif kind == "vendor_claim":
        label, strength = "Vendor claim" + suffix, "weak"
    else:
        label = "Evidence" + suffix
        if not levels or best == 0:
            strength = "strong"
        elif best is not None:
            strength = "medium"
        else:
            strength = "weak"
        if suffix:
            status = "downgraded" if claim.scope == "target_location" else status

    return {**base, "label": label, "sources": cited, "status": status, "strength": strength}


def _is_sourced(checked: Dict[str, Any]) -> bool:
    return bool(checked["sources"]) and not checked["status"].startswith("rejected")


def claim_text(checked: Dict[str, Any]) -> str:
    """Claim text as shown to users. Prices name the one site they come from."""
    text = checked["text"]
    if checked["sources"] and _CURRENCY.search(text):
        host = _host(checked["sources"][0]["url"])
        if host and host.split(".")[0] not in text.lower():
            text = f"According to {host}: {text}"
    return text


def _render_claim(checked: Dict[str, Any], footnotes: Dict[str, int]) -> str:
    if checked["status"] == "rejected_figure":
        return (
            f"[Not established] {checked['text']} — this figure was not found in "
            "the cited sources, so it is not presented as fact."
        )
    if checked["status"] == "rejected_price_mix":
        return (
            f"[Not established] {checked['text']} — this price combines several sources, "
            "so it cannot be attributed to one provider."
        )
    line = f"[{checked['label']}] {claim_text(checked)}"
    marks = "".join(f"[{footnotes[s['id']]}]" for s in checked["sources"] if s["id"] in footnotes)
    return f"{line} {marks}" if marks else line


def clean_summary(summary: str, verified_text: str, question: str) -> str:
    """Keep the analyst's paragraphs, minus any sentence whose figures are not in a verified finding."""
    allowed = set(_numbers(verified_text + " " + question))
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", (summary or "").strip()):
        kept = [
            s for s in re.split(r"(?<=[.!?])\s+", paragraph.strip())
            if s and all(n in allowed for n in _numbers(s))
        ]
        if kept:
            paragraphs.append(" ".join(kept))
    return "\n\n".join(paragraphs)


def _norm_aspect(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def build_report(
    analysis: AnalysisOutput,
    sources: List[Dict[str, Any]],
    question: str,
    target_location: Optional[str],
    planned_aspects: Optional[List[str]] = None,
    failed_aspects: Optional[List[str]] = None,
) -> Tuple[ResearchReport, List[str]]:
    """Check every claim, then render the report. Also returns the unanswered aspects."""
    sources_by_id = {s["id"]: s for s in sources}
    assessments = {a.id.strip().upper(): a for a in analysis.source_assessments}
    levels = _location_levels(target_location)

    checked_findings = [check_claim(c, sources_by_id, assessments, levels) for c in analysis.findings]
    checked_opps = [check_claim(c, sources_by_id, assessments, levels) for c in analysis.opportunities]

    # Footnote numbers in order of first use; the Sources list follows the same order.
    footnotes: Dict[str, int] = {}
    for c in checked_findings + checked_opps:
        for s in c["sources"]:
            footnotes.setdefault(s["id"], len(footnotes) + 1)

    key_findings = [_render_claim(c, footnotes) for c in checked_findings]
    opportunities = [_render_claim(c, footnotes) for c in checked_opps]

    sourced = {i + 1: c for i, c in enumerate(checked_findings) if _is_sourced(c)}

    # ---- recommendations must rest on sourced findings, and say how strong they are
    recommendations = []
    for rec in analysis.recommendations:
        basis = sorted({n for n in rec.based_on_findings if n in sourced})
        refs = f"finding{'s' if len(basis) > 1 else ''} {', '.join(map(str, basis))}"
        basis_text = " ".join(sourced[n]["text"] for n in basis)
        if not basis:
            recommendations.append(f"[Untested idea] {rec.text} (Not backed by a sourced finding — validate first)")
        elif not _figures_in(_numbers(rec.text), [{"title": "", "content": basis_text}]):
            recommendations.append(
                f"[Untested idea] {rec.text} (Contains figures that are not in the findings it cites — validate first)"
            )
        elif any(sourced[n]["strength"] in ("strong", "medium") for n in basis):
            recommendations.append(f"{rec.text} (Based on {refs})")
        else:
            recommendations.append(
                f"[Weakly supported] {rec.text} (Based on {refs} — vendor, social or non-local evidence only)"
            )

    # ---- unknowns: the analyst's own, plus every planned aspect nothing answered
    unknowns = [u.strip() for u in analysis.unknowns if u and u.strip()]
    gaps: List[str] = []

    failed = {_norm_aspect(a) for a in (failed_aspects or [])}
    coverage: Dict[str, set] = {}
    for item in analysis.aspect_coverage:
        coverage.setdefault(_norm_aspect(item.aspect), set()).update(item.answering_findings)

    small_words = {"in", "of", "for", "the", "and", "a", "an", "to", "on", "with", "by", "at"}
    place_words = {w for _name, terms in levels for term in terms for w in term.split()}

    def key_words(text: str) -> set:
        return {
            w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in text.split() if w not in small_words and w not in place_words
        }

    def coverage_for(key: str) -> Optional[set]:
        """Analyst's answer for this aspect, matching reworded names
        ('competitors' = 'Competitors in Abuja' = 'competitor')."""
        if key in coverage:
            return coverage[key]
        words = key_words(key)
        for other, numbers in coverage.items():
            other_words = key_words(other)
            if words and other_words and (words <= other_words or other_words <= words):
                return numbers
        return None

    for aspect in dict.fromkeys(planned_aspects or []):
        key = _norm_aspect(aspect)
        if not key or key in ("general", "follow up"):
            continue
        listed = coverage_for(key)
        if listed is not None:
            # The analyst said which findings answer it: trust only those that survived checks.
            answered = {n for n in listed if n in sourced}
        else:
            # The analyst skipped it: count sourced findings whose source came from this search.
            answered = {
                n for n, c in sourced.items()
                if any(_norm_aspect(s.get("aspect", "")) == key for s in c["sources"])
            }
        if answered:
            continue
        gaps.append(aspect)
        aspect_words = set(key.split())
        if any(aspect_words <= set(_norm_aspect(u).split()) for u in unknowns):
            continue  # the analyst already reported this gap
        if key in failed:
            unknowns.append(f"Not researched: the search for '{aspect}' failed.")
        else:
            unknowns.append(f"No usable evidence was found for: {aspect}.")

    if _PRICE_WORDS.search(question) and not any(_CURRENCY.search(source_text(s)) for s in sources):
        unknowns.append("No published prices were found in the retrieved sources.")

    unknowns = list(dict.fromkeys(unknowns))
    key_findings += [f"[Not established] {u}" for u in unknowns]

    # ---- sources: numbered in footnote order
    by_number = sorted(footnotes.items(), key=lambda item: item[1])
    cited_urls = [sources_by_id[sid]["url"] for sid, _n in by_number]

    # ---- summary: the analyst's own (fuller) summary, with unverified figures removed
    verified_text = " ".join(claim_text(c) for c in sourced.values())
    summary = clean_summary(analysis.summary, verified_text, question)
    if not summary:
        summary = "The sources did not support a confident summary."
    if unknowns:
        shown = "; ".join(u.rstrip(".") for u in unknowns[:4])
        more = f" (and {len(unknowns) - 4} more)" if len(unknowns) > 4 else ""
        summary += f"\n\nNot established from the available sources: {shown}{more}."

    report = ResearchReport(
        summary=summary,
        key_findings=key_findings,
        opportunities=opportunities,
        recommendations=recommendations,
        sources=cited_urls,
    )
    return report, gaps


def validate_and_render(
    analysis: AnalysisOutput,
    sources: List[Dict[str, Any]],
    question: str,
    target_location: Optional[str],
    planned_aspects: Optional[List[str]] = None,
    failed_aspects: Optional[List[str]] = None,
) -> ResearchReport:
    """Backwards-compatible wrapper: the checked report only."""
    report, _gaps = build_report(
        analysis, sources, question, target_location, planned_aspects, failed_aspects
    )
    return report


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
# After analysis: one optional follow-up round of searches for gaps, then analyse again.
workflow.add_conditional_edges(
    "analyze",
    route_after_analyze,
    {"research": "research", "done": END},
)
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
