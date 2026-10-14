"""Tests for the depth upgrade: more reading, fuller summary, footnotes, follow-up searches.

SYNTHETIC data only. No real API calls (network blocked in conftest).
"""
from test_pipeline import A, C, SA, make_plan

SEGMENTS_ONLY = "Which customer segments should I target with WhatsApp chatbots in Nigeria?"
QUESTION = "Who are the WhatsApp chatbot providers in Nigeria, what do they charge, and which customer segments should I target?"

SYNTHETIC = {
    "q-providers": [
        {"title": "ChatFlowX (synthetic vendor)", "url": "https://chatflowx.example/features",
         "content": "ChatFlowX offers WhatsApp chatbots and booking reminders for Nigerian businesses.", "score": 0.9},
        {"title": "BotPalX pricing (synthetic vendor)", "url": "https://botpalx.example/pricing",
         "content": "BotPalX WhatsApp chatbot plans start at ₦25,000 per month for Nigerian businesses.", "score": 0.8},
    ],
    "q-pricing": [
        {"title": "Chatbot pricing guide Nigeria (synthetic)", "url": "https://guide.example/chatbot-pricing-nigeria",
         "content": "WhatsApp chatbot plans in Nigeria range widely between providers.", "score": 0.5},
    ],
    "segments salon owners WhatsApp chatbot nigeria": [
        {"title": "Salon owners survey Nigeria (synthetic)", "url": "https://survey.example/salons",
         "content": "In a survey, Nigerian salon owners said WhatsApp is their main booking channel.", "score": 0.7},
    ],
    "WhatsApp chatbot customer segments nigeria": [
        {"title": "Segments study (synthetic)", "url": "https://study.example/segments",
         "content": "Clinics and salons in Nigeria were the most active WhatsApp chatbot buyers in the study.", "score": 0.6},
    ],
}


class SequenceModel:
    """Fake structured model returning a different output on each call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return out


def setup(app, monkeypatch, analyses, tasks, offering="WhatsApp chatbot"):
    plan = app.ResearchPlan(
        target_location="Nigeria", country="nigeria", offering=offering,
        tasks=[app.ResearchTask(aspect=a, query=q) for a, q in tasks],
    )
    searched = []

    def fake_search(query, country=None):
        searched.append(query)
        return {"query": query, "results": SYNTHETIC.get(query, [])} if query in SYNTHETIC else {"results": []}

    analyst = SequenceModel(analyses)
    monkeypatch.setattr(app, "planner", SequenceModel([plan]))
    monkeypatch.setattr(app, "analyst", analyst)
    monkeypatch.setattr(app, "run_search", fake_search)
    return analyst, searched


# ---------------------------------------------------------------- defaults

def test_default_settings_read_page_text_and_more_results(app_module):
    app = app_module
    assert app.INCLUDE_RAW is True              # page text, not just 1-2 sentence snippets
    assert app.MAX_RESULTS == 8 and app.MAX_SEARCHES == 6
    assert app.MAX_SOURCES_FOR_ANALYSIS == 24


def test_search_tool_requests_page_text(app_module):
    app = app_module
    app._search_tools.clear()
    tool = app._get_search_tool("nigeria")
    assert tool.include_raw_content == "markdown" and tool.max_results == 8


def test_only_best_sources_sent_to_analyst_in_original_order(app_module, monkeypatch):
    app = app_module
    monkeypatch.setattr(app, "MAX_SOURCES_FOR_ANALYSIS", 2)
    sources = [{"id": "S1", "score": 0.1}, {"id": "S2", "score": 0.9}, {"id": "S3", "score": 0.5}]
    assert [s["id"] for s in app.select_sources_for_analysis(sources)] == ["S2", "S3"]


# ---------------------------------------------------------------- prompt asks for depth

def test_analyst_prompt_asks_for_depth_and_comparison(app_module, monkeypatch):
    app = app_module
    analyst, _ = setup(app, monkeypatch, [A(app)], [("competitors", "q-providers")])
    app.app.invoke({"question": QUESTION, "search_results": "", "report": None})
    prompt = analyst.prompts[0]
    assert "Cover EVERY part of the question in depth" in prompt
    assert "aim for 8-15 findings" in prompt
    assert "executive summary of 3-5 short paragraphs" in prompt
    assert "Fewer, well-supported findings" not in prompt


def test_planner_prompt_does_not_invent_topics(app_module, monkeypatch):
    app = app_module
    _, _ = setup(app, monkeypatch, [A(app)], [("keywords", "q-providers")])
    planner = app.planner
    app.app.invoke({"question": "Which keywords should an automation company target?",
                    "search_results": "", "report": None})
    assert "Do not add\ntopics the question does not ask about" in planner.prompts[0]


# ---------------------------------------------------------------- footnotes

def test_findings_use_footnote_numbers_and_sources_follow_that_order(app_module, monkeypatch):
    app = app_module
    analysis = A(app,
        summary="Two providers were found.\n\nOne publishes a price.",
        findings=[
            C(app, "BotPalX says its WhatsApp chatbot plans start at ₦25,000 per month.",
              kind="vendor_claim", scope="other_or_unspecified", ids=["S2"]),
            C(app, "ChatFlowX says it offers WhatsApp chatbots and booking reminders.",
              kind="vendor_claim", scope="other_or_unspecified", ids=["S1"]),
            C(app, "BotPalX and ChatFlowX both sell WhatsApp chatbots to Nigerian businesses.",
              kind="vendor_claim", scope="other_or_unspecified", ids=["S1", "S2"]),
        ],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page"),
                            SA(app, "S2", source_type="company_or_vendor_page")],
    )
    setup(app, monkeypatch, [analysis], [("competitors", "q-providers")])
    r = app.app.invoke({"question": QUESTION, "search_results": "", "report": None})["report"]
    assert r.key_findings[0].endswith("₦25,000 per month. [1]")
    assert r.key_findings[1].endswith("booking reminders. [2]")
    assert r.key_findings[2].endswith("[2][1]")
    assert r.sources == ["https://botpalx.example/pricing", "https://chatflowx.example/features"]
    assert not any("https://" in f for f in r.key_findings)


# ---------------------------------------------------------------- summary

def test_summary_keeps_multiple_paragraphs(app_module, monkeypatch):
    app = app_module
    analysis = A(app,
        summary="First paragraph about providers.\n\nSecond paragraph about pricing.\n\nThird: recommendation.",
        findings=[C(app, "ChatFlowX says it offers WhatsApp chatbots and booking reminders.",
                    kind="vendor_claim", ids=["S1"])],
        aspect_coverage=[app.AspectCoverage(aspect="competitors", answering_findings=[1])],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    setup(app, monkeypatch, [analysis], [("competitors", "q-providers")])
    r = app.app.invoke({"question": "Who are the WhatsApp chatbot providers in Nigeria?",
                        "search_results": "", "report": None})["report"]
    assert r.summary.split("\n\n") == [
        "First paragraph about providers.", "Second paragraph about pricing.", "Third: recommendation."]


# ---------------------------------------------------------------- follow-up searches

def test_gaps_trigger_one_follow_up_round_then_final_report(app_module, monkeypatch):
    app = app_module
    monkeypatch.setattr(app, "FOLLOW_UP_ROUNDS", 1)
    first = A(app,
        summary="First pass.",
        findings=[C(app, "ChatFlowX says it offers WhatsApp chatbots and booking reminders.",
                    kind="vendor_claim", ids=["S1"])],
        aspect_coverage=[app.AspectCoverage(aspect="competitors", answering_findings=[1]),
                         app.AspectCoverage(aspect="customer segments", answering_findings=[])],
        follow_up_queries=["segments salon owners WhatsApp chatbot nigeria"],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    second = A(app,
        summary="After follow-up, salon owners use WhatsApp for bookings.",
        findings=[
            C(app, "ChatFlowX says it offers WhatsApp chatbots and booking reminders.", kind="vendor_claim", ids=["S1"]),
            C(app, "Nigerian salon owners said WhatsApp is their main booking channel.",
              scope="target_location", ids=["S4"]),
        ],
        aspect_coverage=[app.AspectCoverage(aspect="competitors", answering_findings=[1]),
                         app.AspectCoverage(aspect="customer segments", answering_findings=[2])],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page"), SA(app, "S4")],
    )
    analyst, searched = setup(app, monkeypatch, [first, second],
                              [("competitors", "q-providers"), ("customer segments", "q-pricing")])
    result = app.app.invoke({"question": QUESTION, "search_results": "", "report": None})

    assert analyst.calls == 2                                     # exactly one extra analysis
    assert searched[:2] == ["q-providers", "q-pricing"]
    follow_ups = searched[2:]
    assert 1 <= len(follow_ups) <= app.MAX_FOLLOW_UP_SEARCHES      # at most 3 extra searches
    assert follow_ups[0] == "segments salon owners WhatsApp chatbot nigeria"
    assert [s["id"] for s in result["sources"]][:4] == ["S1", "S2", "S3", "S4"]   # IDs continue
    assert "S4" in analyst.prompts[1] and "S4" not in analyst.prompts[0]
    r = result["report"]
    assert r.summary.startswith("After follow-up")
    assert r.key_findings[1].startswith("[Evidence] Nigerian salon owners")
    assert not any("customer segments" in f for f in r.key_findings)


def test_template_follow_up_when_analyst_gives_no_queries_and_no_repeats(app_module, monkeypatch):
    app = app_module
    monkeypatch.setattr(app, "FOLLOW_UP_ROUNDS", 1)
    gap_only = A(app, summary="x.",
                 aspect_coverage=[app.AspectCoverage(aspect="customer segments", answering_findings=[])],
                 follow_up_queries=["q-providers"])           # already searched -> must not repeat
    analyst, searched = setup(app, monkeypatch, [gap_only, gap_only],
                              [("customer segments", "q-providers")])
    app.app.invoke({"question": SEGMENTS_ONLY, "search_results": "", "report": None})
    assert searched == ["q-providers", "customer segments WhatsApp chatbot nigeria"]
    assert analyst.calls == 2


def test_no_follow_up_when_nothing_missing(app_module, monkeypatch):
    app = app_module
    monkeypatch.setattr(app, "FOLLOW_UP_ROUNDS", 1)
    complete = A(app, summary="Complete.",
                 findings=[C(app, "ChatFlowX says it offers WhatsApp chatbots and booking reminders.",
                             kind="vendor_claim", ids=["S1"])],
                 aspect_coverage=[app.AspectCoverage(aspect="competitors", answering_findings=[1])],
                 source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")])
    analyst, searched = setup(app, monkeypatch, [complete], [("competitors", "q-providers")])
    r = app.app.invoke({"question": "Who are the WhatsApp chatbot providers in Nigeria?",
                        "search_results": "", "report": None})["report"]
    assert analyst.calls == 1 and searched == ["q-providers"]
    assert r.summary == "Complete."


def test_follow_up_can_be_switched_off(app_module, monkeypatch):
    app = app_module
    monkeypatch.setattr(app, "FOLLOW_UP_ROUNDS", 0)
    gap = A(app, summary="x.", follow_up_queries=["segments salon owners WhatsApp chatbot nigeria"],
            aspect_coverage=[app.AspectCoverage(aspect="customer segments", answering_findings=[])])
    analyst, searched = setup(app, monkeypatch, [gap], [("customer segments", "q-providers")])
    app.app.invoke({"question": SEGMENTS_ONLY, "search_results": "", "report": None})
    assert analyst.calls == 1 and searched == ["q-providers"]
