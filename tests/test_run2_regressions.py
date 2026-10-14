"""Regression tests for problems seen in the second real run (14 Oct 2026).

SYNTHETIC data only (fictional companies on .example domains, except the
platform domains whose handling is being tested). No real API calls.
"""
from test_pipeline import A, C, SA, make_plan, run_pipeline

QUESTION = (
    "I want to market AI chatbot and WhatsApp automation services to beauty businesses "
    "in Abuja, Nigeria. Who are the competitors and what is their pricing?"
)

SYNTHETIC = {
    "q-vendor-price": [
        {"title": "MusskartX AI chatbot & WhatsApp automation Nigeria (synthetic vendor)",
         "url": "https://musskartx.example/ai-chatbot-whatsapp-automation-nigeria.html",
         "content": "AI chatbot setup in Nigeria typically costs ₦300,000 to ₦1,500,000, with monthly running costs of ₦30,000 to ₦120,000."},
    ],
    "q-scribd": [
        {"title": "Beauty Touch Biz Plan (synthetic upload)",
         "url": "https://www.scribd.com/document/000000/Beauty-Touch-Biz-Plan",
         "content": "Beauty businesses in Abuja face long wait times and uncertain service outcomes when booking appointments."},
    ],
    "q-market": [
        {"title": "Nigeria beauty market report (synthetic)",
         "url": "https://market-reports.example/nigeria-beauty",
         "content": "The Nigeria beauty and personal care market is expected to grow 9.1% a year from 2026 to 2034."},
        {"title": "Abuja salon survey (synthetic)",
         "url": "https://abuja-salons.example/survey",
         "content": "Abuja salon owners said missed WhatsApp enquiries after closing hours were their main booking problem."},
    ],
}


def responses():
    return {q: {"query": q, "results": r} for q, r in SYNTHETIC.items()}


def run(app, monkeypatch, fake_model, analysis, tasks):
    plan = make_plan(app, tasks, location="Abuja, Nigeria", country="nigeria")
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, plan, responses(), analysis, question=QUESTION)
    return result


# 1. "The setup cost for an AI chatbot in Nigeria typically ranges ₦300,000–₦1,500,000"
#    came from one vendor's page but read like a market-wide fact.
def test_price_from_one_site_names_that_site(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        findings=[C(app, "AI chatbot setup in Nigeria typically costs ₦300,000 to ₦1,500,000.", ids=["S1"])],
        source_assessments=[SA(app, "S1")],          # model wrongly calls it independent
    )
    result = run(app, monkeypatch, fake_model, analysis, [("published pricing", "q-vendor-price")])
    finding = result["report"].key_findings[0]
    assert "According to musskartx.example: AI chatbot setup in Nigeria typically costs ₦300,000" in finding


# 2. Summary said "In Abuja, beauty businesses face challenges..." from a vendor/upload claim.
def test_summary_does_not_restate_weak_claims_as_fact(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        summary="In Abuja, beauty businesses face challenges with booking systems.",   # ignored now
        findings=[C(app, "Beauty businesses in Abuja face long wait times when booking appointments.", ids=["S1"])],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    result = run(app, monkeypatch, fake_model, analysis, [("customer problems", "q-scribd")])
    summary = result["report"].summary
    assert "In Abuja, beauty businesses face challenges" not in summary
    assert summary.startswith("The sources found no independent evidence for this question in Abuja.")
    assert "1 further finding rests only on vendor websites, social posts, uploaded documents" in summary


# 3. A business plan uploaded to Scribd was treated as a competitor / evidence.
def test_uploaded_documents_are_labelled_as_such(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        findings=[C(app, "Beauty businesses in Abuja face long wait times when booking appointments.", ids=["S1"])],
        source_assessments=[SA(app, "S1")],
    )
    result = run(app, monkeypatch, fake_model, analysis, [("customer problems", "q-scribd")])
    assert result["report"].key_findings[0].startswith("[Uploaded document] Beauty businesses in Abuja")


# Summary shows strong and country-wide evidence separately, in plain words.
def test_summary_separates_local_and_country_wide_evidence(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        findings=[
            C(app, "Abuja salon owners said missed WhatsApp enquiries after closing hours were their main booking problem.", ids=["S2"]),
            C(app, "The Nigeria beauty and personal care market is expected to grow 9.1% a year.", ids=["S1"]),
        ],
        source_assessments=[SA(app, "S1"), SA(app, "S2")],
    )
    result = run(app, monkeypatch, fake_model, analysis, [("market", "q-market")])
    summary = result["report"].summary
    assert summary.startswith("Independent evidence about Abuja: Abuja salon owners said missed WhatsApp")
    assert "Evidence about Nigeria as a whole (not specific to Abuja): The Nigeria beauty" in summary
    assert "Evidence check: 2 of 2 findings have a cited source; 2 of these" in summary


# 4. The planner skipped the competitor part of the question.
def test_missing_competitor_search_is_added_without_extra_calls(app_module):
    app = app_module
    tasks = [
        {"aspect": "market", "query": "beauty businesses Abuja market analysis"},
        {"aspect": "problems", "query": "beauty salons Abuja booking issues"},
        {"aspect": "industry", "query": "AI chatbot services beauty industry Nigeria"},
        {"aspect": "pricing", "query": "WhatsApp automation pricing Nigeria beauty businesses"},
        {"aspect": "engagement", "query": "beauty business customer engagement solutions Abuja"},
    ]
    out = app.ensure_required_searches(QUESTION, tasks, "WhatsApp chatbot", "nigeria")
    assert len(out) == len(tasks) == app.MAX_SEARCHES          # no extra Tavily calls
    assert out[-1] == {"aspect": "competitors", "query": "WhatsApp chatbot provider nigeria"}
    assert out[3]["aspect"] == "pricing"                        # pricing already covered, kept


def test_missing_competitor_and_pricing_both_added(app_module):
    app = app_module
    tasks = [{"aspect": "market", "query": "beauty salons Abuja"}]
    out = app.ensure_required_searches(QUESTION, tasks, "WhatsApp chatbot", "nigeria")
    assert [t["aspect"] for t in out] == ["market", "competitors", "published pricing"]
    assert out[2]["query"] == "WhatsApp chatbot pricing nigeria per month"


def test_no_search_added_when_not_asked_or_offering_unknown(app_module):
    app = app_module
    tasks = [{"aspect": "market", "query": "beauty salons Abuja"}]
    assert app.ensure_required_searches("What are beauty trends in Abuja?", tasks, "WhatsApp chatbot", "nigeria") == tasks
    assert app.ensure_required_searches(QUESTION, tasks, None, "nigeria") == tasks


def test_planner_output_goes_through_required_search_check(app_module, monkeypatch, fake_model):
    app = app_module
    plan = app.ResearchPlan(
        target_location="Abuja, Nigeria", country="nigeria", offering="WhatsApp chatbot",
        tasks=[app.ResearchTask(aspect="market", query="q-market")],
    )
    searched = []
    monkeypatch.setattr(app, "planner", fake_model(output=plan))
    monkeypatch.setattr(app, "analyst", fake_model(output=A(app)))
    monkeypatch.setattr(app, "run_search", lambda q, c=None: searched.append(q) or responses()["q-market"])
    app.app.invoke({"question": QUESTION, "search_results": "", "report": None})
    assert searched == ["q-market", "WhatsApp chatbot provider nigeria", "WhatsApp chatbot pricing nigeria per month"]
