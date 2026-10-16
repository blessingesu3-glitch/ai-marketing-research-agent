"""Competitor teardown: read up to 3 competitors' own websites and compare them side by side.

Steps (each search is one Tavily call; one model call at the end):
  1. find_website   - use a pasted website as-is, or look a plain name up (1 search)
  2. read_site      - search inside that website for offer / pricing pages (1 search)
  3. analyse        - one structured model call over all pages read
  4. check_cards    - code checks: prices must appear on that competitor's own pages,
                      sources must belong to that competitor, profile is never a source
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import app as core

MAX_COMPETITORS = 3
PAGES_PER_SITE = 4
PAGE_TEXT_CHARS = 2000
NOT_FOUND = "Not found on their website"

# Results that are never a company's own website.
_NOT_OWN_SITE = (
    "wikipedia.org", "crunchbase.com", "glassdoor.com", "indeed.com", "yelp.com",
    "trustpilot.com", "g2.com", "capterra.com", "bloomberg.com", "zoominfo.com",
    "nairaland.com", "jiji.ng", "vconnect.com", "businesslist.com.ng", "finelib.com",
)


# ---------------------------------------------------------------- model output

class CompetitorCard(BaseModel):
    name: str
    offer: str = Field(description="What they sell: products, services, packages. 1-3 sentences.")
    pricing: str = Field(
        description="Published prices exactly as on their pages (plan names + amounts), "
        "or 'Not found on their website'."
    )
    target_customer: str = Field(description="Who they say they serve, or 'Not found on their website'.")
    positioning: str = Field(description="Their main promise or message, in 1-2 sentences.")
    channels: str = Field(
        description="How customers reach them: WhatsApp, phone, booking form, social pages, "
        "branches - only those shown on their pages."
    )
    source_ids: List[str] = Field(default_factory=list, description="IDs like 'C1-P2' of pages used.")


class TeardownAnalysis(BaseModel):
    summary: str = Field(
        default="",
        description="2-4 short paragraphs separated by blank lines: how the competitors "
        "differ, who looks strongest on what, and the clearest openings.",
    )
    competitors: List[CompetitorCard] = Field(default_factory=list)
    gaps: List[str] = Field(
        default_factory=list,
        description="3-6 openings: things customers would value that these competitors "
        "do not offer or do not make clear. If a business profile is given, say how "
        "THAT business can use each opening.",
    )
    recommendations: List[str] = Field(
        default_factory=list, description="3-5 concrete actions, each with its reasoning."
    )


teardown_analyst = core.model.with_structured_output(TeardownAnalysis)


# ---------------------------------------------------------------- helpers

def looks_like_website(text: str) -> bool:
    text = text.strip().lower()
    return bool(re.match(r"^(https?://)?([a-z0-9-]+\.)+[a-z]{2,}(/\S*)?$", text))


def site_domain(url_or_domain: str) -> str:
    text = url_or_domain.strip()
    if not re.match(r"^https?://", text, re.I):
        text = "https://" + text
    return core._host(text)


def _name_tokens(name: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) >= 3]


def _is_own_site_candidate(url: str) -> bool:
    host = core._host(url)
    if core.is_social({"url": url}) or core.is_uploaded({"url": url}):
        return False
    return not any(host == h or host.endswith("." + h) for h in _NOT_OWN_SITE)


def run_site_search(query: str, domain: str) -> Any:
    """Search only inside one website. The only Tavily call made by the teardown
    besides name lookups. Tests replace this function."""
    return core._get_search_tool(None).invoke({"query": query, "include_domains": [domain]})


# ---------------------------------------------------------------- steps

def find_website(entry: str, profile: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Return {'name', 'domain' or None, 'note'}."""
    entry = " ".join(entry.split())
    if looks_like_website(entry):
        domain = site_domain(entry)
        return {"name": core._host_label("https://" + domain).capitalize(), "domain": domain, "note": ""}

    place = (profile or {}).get("location", "")
    country = core.country_from_location(place)
    query = " ".join(f"{entry} official website {place}".split())
    try:
        raw = core.run_search(query, country)
    except Exception as exc:
        raw = {"error": exc}
    items, reason = core.classify_search_result(raw)
    if not items:
        return {"name": entry, "domain": None, "note": f"Website not found ({reason})."}

    candidates = [i["url"] for i in items if _is_own_site_candidate(i["url"])]
    tokens = _name_tokens(entry)
    for url in candidates:                      # prefer a domain that contains the name
        label = re.sub(r"[^a-z0-9]", "", core._host_label(url))
        if any(t in label for t in tokens):
            return {"name": entry, "domain": core._host(url), "note": ""}
    if candidates:
        return {"name": entry, "domain": core._host(candidates[0]),
                "note": "Website matched by search, not by name - please check it is the right company."}
    return {"name": entry, "domain": None, "note": "Only directory or social pages found, no own website."}


def read_site(competitor: Dict[str, Any], index: int, offering: Optional[str]) -> List[Dict[str, Any]]:
    """Pages from the competitor's own website, as numbered sources 'C<index>-P<n>'."""
    if not competitor["domain"]:
        return []
    query = " ".join(f"{competitor['name']} {offering or ''} pricing plans packages services contact".split())
    try:
        raw = run_site_search(query, competitor["domain"])
    except Exception as exc:
        raw = {"error": exc}
    items, reason = core.classify_search_result(raw)
    if not items:
        competitor["note"] = (competitor.get("note") or "") + f" Website could not be read ({reason})."
        return []

    pages, seen = [], set()
    for item in items:
        if core._host(item["url"]) != competitor["domain"] and not core._host(item["url"]).endswith(
            "." + competitor["domain"]
        ):
            continue                              # only the competitor's own pages
        key = core.normalize_url(item["url"])
        if key in seen:
            continue
        seen.add(key)
        pages.append({
            "id": f"C{index}-P{len(pages) + 1}",
            "url": item["url"],
            "title": item.get("title") or "",
            "content": core._clip(item.get("content"), 600),
            "raw_excerpt": core._clip(item.get("raw_content"), PAGE_TEXT_CHARS),
        })
        if len(pages) >= PAGES_PER_SITE:
            break
    if not pages:
        competitor["note"] = (competitor.get("note") or "") + " No pages from their own website were found."
    return pages


def _pages_block(competitors: List[Dict[str, Any]]) -> str:
    blocks = []
    for c in competitors:
        header = f"=== Competitor: {c['name']} (website: {c['domain'] or 'not found'})"
        if not c["pages"]:
            blocks.append(header + "\n(no pages could be read)")
            continue
        pages = "\n\n".join(
            f"[{p['id']}] {p['title']}\nURL: {p['url']}\n{p['raw_excerpt'] or p['content']}" for p in c["pages"]
        )
        blocks.append(header + "\n" + pages)
    return "\n\n".join(blocks)


def check_cards(analysis: TeardownAnalysis, competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One checked card per requested competitor, in the order the user gave them."""
    cards_by_name = {c.name.strip().lower(): c for c in analysis.competitors}
    checked = []
    for comp in competitors:
        card = cards_by_name.get(comp["name"].strip().lower())
        if card is None:                                   # match loosely by name tokens
            tokens = _name_tokens(comp["name"])
            card = next((c for c in analysis.competitors
                         if any(t in c.name.lower() for t in tokens)), None)
        own_pages = {p["id"]: p for p in comp["pages"]}

        if card is None or not own_pages:
            checked.append({
                "name": comp["name"], "website": comp["domain"] or "",
                "offer": NOT_FOUND, "pricing": NOT_FOUND, "target_customer": NOT_FOUND,
                "positioning": NOT_FOUND, "channels": NOT_FOUND, "sources": [],
                "note": (comp.get("note") or "").strip() or "Nothing could be read from their website.",
            })
            continue

        used = [own_pages[i.strip().upper()] for i in card.source_ids if i.strip().upper() in own_pages]

        pricing = card.pricing.strip() or NOT_FOUND
        note = (comp.get("note") or "").strip()
        figures = core._numbers(pricing)
        if figures and not core._figures_in(figures, comp["pages"]):
            pricing = NOT_FOUND                            # price not on their own pages
            note = (note + " A price was suggested but it does not appear on their pages, so it was removed.").strip()

        checked.append({
            "name": comp["name"], "website": comp["domain"] or "",
            "offer": card.offer.strip() or NOT_FOUND,
            "pricing": pricing,
            "target_customer": card.target_customer.strip() or NOT_FOUND,
            "positioning": card.positioning.strip() or NOT_FOUND,
            "channels": card.channels.strip() or NOT_FOUND,
            "sources": [p["url"] for p in (used or comp["pages"])],
            "note": note,
        })
    return checked


def run_teardown(
    entries: List[str], profile: Optional[Dict[str, Any]] = None, focus: Optional[str] = None
) -> Dict[str, Any]:
    profile = core.clean_profile(profile)
    entries = [e for e in dict.fromkeys(" ".join(e.split()) for e in entries) if e][:MAX_COMPETITORS]
    offering = (profile or {}).get("offering")

    print(f"\n[Teardown] {len(entries)} competitors" + (" (tailored to the business profile)" if profile else ""))
    competitors = []
    for index, entry in enumerate(entries, start=1):
        comp = find_website(entry, profile)
        comp["pages"] = read_site(comp, index, offering)
        print(f"[Teardown] {comp['name']}: {comp['domain'] or 'no website'} - {len(comp['pages'])} pages read")
        competitors.append(comp)

    if not any(c["pages"] for c in competitors):
        print("[Teardown] Skipped analysis: no pages could be read.")
        cards = check_cards(TeardownAnalysis(), competitors)
        return {
            "summary": "None of the competitors' websites could be read, so no comparison was made. "
                       "Check the names or paste their website addresses and try again.",
            "competitors": cards, "gaps": [], "recommendations": [], "sources": [],
        }

    profile_note = ""
    if profile:
        profile_note = f"""
The user's own business (written by the user; use it to find openings for THIS
business; it is NOT evidence and must never be described as a competitor):
{core.profile_block(profile)}
"""
    focus_note = f"\nThe user especially wants to know: {focus.strip()}\n" if focus and focus.strip() else ""

    prompt = f"""You are a senior marketing strategist doing a competitor teardown.
Use ONLY the pages below, which come from each competitor's own website.
{profile_note}{focus_note}
{_pages_block(competitors)}

How to write it:
- Return one card per competitor, using the competitor names exactly as given above.
- Offer, target customer, positioning and channels: only what their pages show.
- Pricing: copy plan names and amounts exactly as written on their pages. If no price
  is shown, write "{NOT_FOUND}". Never estimate a price.
- source_ids: the page IDs (like C1-P2) each card is based on.
- Summary: 2-4 short paragraphs separated by blank lines comparing them directly.
- Gaps: 3-6 openings a newcomer could use{', and how the user business can use each' if profile else ''}.
- Recommendations: 3-5 concrete actions, each with its reasoning.
- A competitor whose pages could not be read gets a card saying "{NOT_FOUND}".
"""
    analysis = teardown_analyst.invoke([HumanMessage(content=prompt)])
    cards = check_cards(analysis, competitors)

    # Summary sentences may only use figures that appear on the competitors' pages.
    all_pages = [p for c in competitors for p in c["pages"]]
    summary = core.clean_summary(analysis.summary, " ".join(core.source_text(p) for p in all_pages), "")
    sources = list(dict.fromkeys(url for card in cards for url in card["sources"]))

    # Gaps and recommendations may only use figures from the pages or the user's own profile.
    allowed = set(core._numbers(" ".join(core.source_text(p) for p in all_pages) + " " + core.profile_block(profile)))

    def keep(items: List[str]) -> List[str]:
        return [i.strip() for i in items if i.strip() and all(n in allowed for n in core._numbers(i))]

    return {
        "summary": summary or "The competitors' pages did not support a confident summary.",
        "competitors": cards,
        "gaps": keep(analysis.gaps),
        "recommendations": keep(analysis.recommendations),
        "sources": sources,
    }
