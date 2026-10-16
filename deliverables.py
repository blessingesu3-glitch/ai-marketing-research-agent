"""Turn a finished report into ready-to-use work: campaign brief, ad copy, content calendar.

One model call per deliverable and NO new searches: it only uses the report the
user already has (plus their business profile, if the report was tailored).

Code check afterwards: any sentence containing a figure (price, %, count) that
is not in the report or profile is removed, so ads and briefs cannot invent
statistics or prices.
"""
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import app as core

Kind = Literal["campaign_brief", "ad_copy", "content_calendar"]

TITLES = {
    "campaign_brief": "Campaign brief",
    "ad_copy": "Ad copy",
    "content_calendar": "4-week content calendar",
}


# ---------------------------------------------------------------- model outputs

class CampaignBrief(BaseModel):
    objective: str = Field(description="One measurable-sounding objective, without inventing numbers.")
    audience: str
    key_message: str = Field(description="The single main message, one or two sentences.")
    proof_points: List[str] = Field(description="3-5 supporting points taken from the report's findings.")
    offer: str = Field(description="What is offered and why it beats the alternatives found in the report.")
    channels: List[str] = Field(description="3-5 channels, each with one line on why it fits.")
    success_measures: List[str] = Field(description="3-5 things to track (no invented targets).")
    timeline: List[str] = Field(description="3-5 phases, e.g. 'Weeks 1-2: ...'.")
    open_questions: List[str] = Field(description="What the report could not establish and should be checked first.")


class AdVariant(BaseModel):
    channel: str = Field(description="e.g. Instagram/Facebook feed, Google Search, WhatsApp broadcast, LinkedIn")
    angle: str = Field(description="The idea behind this ad, e.g. 'convenience', 'price clarity'")
    headline: str
    body: str
    call_to_action: str
    based_on: str = Field(description="Which finding or opening from the report this ad uses.")


class AdCopy(BaseModel):
    variants: List[AdVariant] = Field(description="6 variants: 2 angles x 3 channels that fit the report.")


class CalendarPost(BaseModel):
    day: str = Field(description="e.g. Monday")
    channel: str
    format: str = Field(description="e.g. Reel, carousel, story, post, blog, WhatsApp status")
    idea: str
    hook: str = Field(description="The first line or caption opener.")


class CalendarWeek(BaseModel):
    week: int
    theme: str
    posts: List[CalendarPost] = Field(description="3 posts for the week.")


class ContentCalendar(BaseModel):
    weeks: List[CalendarWeek] = Field(description="Exactly 4 weeks.")


SCHEMAS = {"campaign_brief": CampaignBrief, "ad_copy": AdCopy, "content_calendar": ContentCalendar}
writers = {kind: core.model.with_structured_output(schema) for kind, schema in SCHEMAS.items()}

INSTRUCTIONS = {
    "campaign_brief": "Write a one-page campaign brief a marketer could hand to a designer or agency today.",
    "ad_copy": "Write 6 ad variants: 2 different angles for each of 3 channels that suit the audience in the report. "
               "Keep headlines short. Write the way the target customer talks.",
    "content_calendar": "Write a 4-week social media content calendar with 3 posts per week, "
                        "building from awareness (week 1) to a clear offer (week 4).",
}


# ---------------------------------------------------------------- helpers

def report_text(report: Dict[str, Any]) -> str:
    parts = [f"Question: {report.get('question', '')}", f"Summary:\n{report.get('summary', '')}"]
    for key, label in (("key_findings", "Findings"), ("opportunities", "Opportunities"),
                       ("recommendations", "Recommendations")):
        items = report.get(key) or []
        if items:
            parts.append(label + ":\n" + "\n".join(f"- {i}" for i in items))
    return "\n\n".join(parts)


_RATIO = re.compile(r"\b(\d+)\s*(?:out of|in every|in|of every)\s*(\d+)\b", re.I)
_MONEY_BEFORE = re.compile(r"(₦|\$|£|€|¥|₹|ngn|usd|naira)\s?$", re.I)


def figures_ok(sentence: str, allowed: set) -> bool:
    """Every figure must be in the report/profile. Exception: plain small counts
    (1-12, e.g. 'Weeks 1-2', '3 posts') - but never %, multipliers, money or
    ratios like '9 out of 10' / '1 in 5', which read as statistics."""
    for ratio in _RATIO.finditer(sentence):
        if not {ratio.group(1), ratio.group(2)} <= allowed:
            return False
    for match in core._NUMBER.finditer(sentence):
        value, unit = match.group(1), (match.group(2) or "")
        end = match.end()
        if unit.lower() in ("x", "k", "m") and end < len(sentence) and sentence[end].isalpha():
            unit = ""                                   # '2 minutes', '3 kids': a word, not a unit
        normal = core._numbers(match.group(0))[0] if core._numbers(match.group(0)) else value
        if normal in allowed:
            continue
        is_money = bool(_MONEY_BEFORE.search(sentence[: match.start()]))
        if not unit and not is_money and value.isdigit() and int(value) <= 12:
            continue
        return False
    return True


def _clean(text: str, allowed: set) -> str:
    """Drop any sentence containing a figure that is not in the report/profile."""
    kept = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip())
            if s and figures_ok(s, allowed)]
    return " ".join(kept).strip()


def _clean_list(items: List[str], allowed: set) -> List[str]:
    return [c for c in (_clean(i, allowed) for i in items) if c]


def to_sections(kind: str, out: BaseModel, allowed: set) -> List[Dict[str, Any]]:
    """Turn the model output into simple titled sections for the website, checking figures."""
    c = lambda t: _clean(t, allowed)          # noqa: E731
    if kind == "campaign_brief":
        b: CampaignBrief = out                 # type: ignore[assignment]
        sections = [
            ("Objective", [c(b.objective)]), ("Audience", [c(b.audience)]),
            ("Key message", [c(b.key_message)]), ("Proof points", _clean_list(b.proof_points, allowed)),
            ("Offer", [c(b.offer)]), ("Channels", _clean_list(b.channels, allowed)),
            ("What to measure", _clean_list(b.success_measures, allowed)),
            ("Timeline", _clean_list(b.timeline, allowed)),
            ("Check before launch", _clean_list(b.open_questions, allowed)),
        ]
    elif kind == "ad_copy":
        a: AdCopy = out                        # type: ignore[assignment]
        sections = []
        for i, v in enumerate(a.variants, start=1):
            lines = [f"Headline: {c(v.headline)}" if c(v.headline) else "",
                     c(v.body), f"Call to action: {c(v.call_to_action)}" if c(v.call_to_action) else "",
                     f"Based on: {c(v.based_on)}" if c(v.based_on) else ""]
            sections.append((f"Ad {i} · {v.channel} · {v.angle}", [l for l in lines if l]))
    else:
        cal: ContentCalendar = out             # type: ignore[assignment]
        sections = []
        for w in cal.weeks[:4]:
            posts = [f"{p.day} · {p.channel} · {p.format}: {c(p.idea)}" + (f" — Hook: \"{c(p.hook)}\"" if c(p.hook) else "")
                     for p in w.posts if c(p.idea)]
            sections.append((f"Week {w.week}: {c(w.theme) or 'Theme'}", posts))
    return [{"heading": h, "items": [i for i in items if i]} for h, items in sections if any(items)]


def plain_text(title: str, sections: List[Dict[str, Any]]) -> str:
    out = [title, ""]
    for s in sections:
        out.append(s["heading"])
        out += [f"- {i}" for i in s["items"]]
        out.append("")
    return "\n".join(out).strip()


def make_deliverable(kind: str, report: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = core.clean_profile(profile)
    source = report_text(report)
    profile_note = ""
    if profile:
        profile_note = f"\nThe user's business (write everything for THIS business):\n{core.profile_block(profile)}\n"

    prompt = f"""You are a senior marketing strategist. {INSTRUCTIONS[kind]}

Use ONLY the research report below{' and the business details' if profile else ''}. Rules:
- Every claim must come from the report. Do not invent statistics, prices, results,
  customer numbers or testimonials. If a number is not in the report, do not use one.
- Items marked [Vendor claim] are a company's own marketing, not facts: never repeat
  them as proven results.
- Items marked [Not established] are unknowns: do not treat them as facts; a brief
  may list them as things to check.
- Be specific to the audience, place and offer in the report. No generic filler.
{profile_note}
RESEARCH REPORT
{source}
"""
    out = writers[kind].invoke([HumanMessage(content=prompt)])
    allowed = set(core._numbers(source + " " + core.profile_block(profile)))
    title = TITLES[kind]
    sections = to_sections(kind, out, allowed)
    return {"kind": kind, "title": title, "sections": sections, "text": plain_text(title, sections)}
