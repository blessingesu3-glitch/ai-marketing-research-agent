"""Regression tests for problems seen in the first real run (14 Oct 2026).

Each test recreates one observed failure with SYNTHETIC data (fictional
companies on .example domains). No real API calls; see conftest.py.
"""
from test_pipeline import A, C, SA, make_plan, run_pipeline

QUESTION = (
    "I want to market AI chatbot and WhatsApp automation services to beauty "
    "businesses in Abuja, Nigeria. Which customer segments are most promising, "
    "who are the competitors, and what is their pricing?"
)

SYNTHETIC = {  # all invented for tests
    "q-social": [
        {"title": "BookGlowX on LinkedIn (synthetic)",
         "url": "https://www.linkedin.com/posts/bookglowx_abuja-beautybusiness-salonowner-activity-1",
         "content": "Most beauty salons in Abuja still rely on walk-ins and phone calls for bookings and enquiries."},
        {"title": "Beauty industry video (synthetic)",
         "url": "https://www.youtube.com/watch?v=synthetic123",
         "content": "The beauty and wellness industry in Nigeria still relies on manual appointment records and booking books."},
    ],
    "q-pricing": [
        {"title": "AI chatbot pricing guide Nigeria (synthetic blog)",
         "url": "https://pricing-guide.example/ai-chatbot-pricing-nigeria",
         "content": "Basic AI chatbot plans in Nigeria start from ₦10,000 to ₦30,000 monthly; enterprise setups can reach ₦1,500,000."},
        {"title": "MusskartX WhatsApp automation (synthetic vendor)",
         "url": "https://musskartx.example/ai-chatbot-whatsapp-automation-nigeria.html",
         "content": "MusskartX builds AI chatbots and WhatsApp automation for Nigerian businesses. Plans from ₦30,000 monthly."},
    ],
    "q-local": [
        {"title": "Abuja salon association survey (synthetic)",
         "url": "https://abuja-salons.example/survey",
         "content": "A survey of Abuja salon owners found missed WhatsApp enquiries after closing hours were the main booking complaint."},
    ],
}

PLAN_TASKS = [("social signals", "q-social"), ("competitor pricing", "q-pricing"),
              ("customer segments", "q-local")]


def responses():
    return {q: {"query": q, "results": r} for q, r in SYNTHETIC.items()}


def run(app, monkeypatch, fake_model, analysis, tasks=PLAN_TASKS, resp=None):
    plan = make_plan(app, tasks, location="Abuja, Nigeria", country="nigeria")
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, plan, resp or responses(), analysis, question=QUESTION)
    return result, {s["url"]: s["id"] for s in result["sources"]}


# 1. Summary repeated "₦10,000 to ₦1,500,000" although no verified finding contained it.
def test_summary_cannot_repeat_figures_missing_from_verified_findings(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        summary="Competitors charge from ₦10,000 to ₦1,500,000. Salons in Abuja miss WhatsApp enquiries after hours.",
        findings=[C(app, "Abuja salon owners report missed WhatsApp enquiries after closing hours.", ids=["S5"])],
        source_assessments=[SA(app, "S5")],
    )
    result, ids = run(app, monkeypatch, fake_model, analysis)
    assert ids["https://abuja-salons.example/survey"] == "S5"
    summary = result["report"].summary
    assert "1,500,000" not in summary and "10,000" not in summary
    assert summary.startswith("Salons in Abuja miss WhatsApp enquiries after hours.")


# 2 + 3. A company's LinkedIn post and a YouTube description were labelled "Evidence".
def test_linkedin_and_youtube_are_social_posts_not_evidence(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app, summary="x.",
        findings=[
            C(app, "Beauty salons in Abuja rely on walk-ins and phone calls for bookings.", ids=["S1"]),
            C(app, "Beauty businesses in Nigeria rely on manual appointment records.", ids=["S2"]),
        ],
        source_assessments=[SA(app, "S1"), SA(app, "S2")],   # model wrongly calls them independent
    )
    result, _ = run(app, monkeypatch, fake_model, analysis)
    f = result["report"].key_findings
    assert f[0].startswith("[Social post] Beauty salons in Abuja")
    assert f[1].startswith("[Social post – Nigeria-wide, not Abuja-specific]")
    assert not any(line.startswith("[Evidence") for line in f)


# 4. "ICS and Musskart ... ₦10,000 to ₦30,000" mixed a blog and a vendor page.
def test_price_claim_combining_two_sources_is_not_established(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app, summary="x.",
        findings=[C(app, "Competitors like ICSX and MusskartX charge ₦10,000 to ₦30,000 monthly for AI chatbot plans.",
                    ids=["S3", "S4"])],
        source_assessments=[SA(app, "S3"), SA(app, "S4", source_type="company_or_vendor_page")],
    )
    result, _ = run(app, monkeypatch, fake_model, analysis)
    finding = result["report"].key_findings[0]
    assert finding.startswith("[Not established] Competitors like ICSX and MusskartX charge")
    assert "cannot be attributed to one provider" in finding
    assert result["report"].sources == []


def test_single_source_price_claim_is_kept_and_attributed(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app, summary="x.",
        findings=[C(app, "MusskartX WhatsApp automation plans start from ₦30,000 monthly.", ids=["S4"])],
        source_assessments=[SA(app, "S4", source_type="independent_article_or_report")],  # model wrong
    )
    result, _ = run(app, monkeypatch, fake_model, analysis)
    finding = result["report"].key_findings[0]
    # detected as the company's own page because the claim names the cited domain
    assert finding.startswith("[Vendor claim – Nigeria-wide, not Abuja-specific] MusskartX")
    assert finding.endswith("[1]")
    assert result["report"].sources == ["https://musskartx.example/ai-chatbot-whatsapp-automation-nigeria.html"]


# 5. Recommendations rested only on vendor/social findings, or added a price.
def test_recommendations_reflect_strength_of_their_evidence(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app, summary="x.",
        findings=[
            C(app, "Beauty salons in Abuja rely on walk-ins and phone calls for bookings.", ids=["S1"]),          # social
            C(app, "MusskartX builds WhatsApp automation for Nigerian businesses.", ids=["S4"]),                 # vendor
            C(app, "Abuja salon owners report missed WhatsApp enquiries after closing hours.", ids=["S5"]),     # local
        ],
        recommendations=[
            app.Recommendation(text="Target salons first.", based_on_findings=[1, 2]),
            app.Recommendation(text="Offer after-hours WhatsApp auto-replies.", based_on_findings=[3]),
            app.Recommendation(text="Price the starter package at ₦20,000 per month.", based_on_findings=[3]),
        ],
        source_assessments=[SA(app, "S1"), SA(app, "S4", source_type="company_or_vendor_page"), SA(app, "S5")],
    )
    result, _ = run(app, monkeypatch, fake_model, analysis)
    recs = result["report"].recommendations
    assert recs[0].startswith("[Weakly supported] Target salons first.")
    assert recs[1] == "Offer after-hours WhatsApp auto-replies. (Based on finding 3)"
    assert recs[2].startswith("[Untested idea] Price the starter package at ₦20,000")


# 6. "Most promising customer segments" was asked, searched, and silently dropped.
def test_unanswered_planned_aspects_become_not_established(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app, summary="x.",
        findings=[
            C(app, "Abuja salon owners report missed WhatsApp enquiries after closing hours.", ids=["S5"]),
            C(app, "Salons are the most promising segment with 70% adoption.", ids=["S5"]),  # figure rejected
        ],
        aspect_coverage=[
            app.AspectCoverage(aspect="customer segments", answering_findings=[2]),   # rejected finding
            app.AspectCoverage(aspect="social signals", answering_findings=[1]),
        ],
        source_assessments=[SA(app, "S5")],
    )
    result, _ = run(app, monkeypatch, fake_model, analysis)
    f = result["report"].key_findings
    assert "[Not established] No usable evidence was found for: customer segments." in f
    assert "[Not established] No usable evidence was found for: competitor pricing." in f
    assert not any("for: social signals" in line for line in f)


def test_failed_aspect_is_reported_as_not_researched(app_module, monkeypatch, fake_model):
    app = app_module
    resp = responses()
    resp["q-pricing"] = {"error": ValueError("Error 432: usage limit exceeded")}
    analysis = A(app, summary="x.",
        findings=[C(app, "Abuja salon owners report missed WhatsApp enquiries after closing hours.", ids=["S3"])],
        aspect_coverage=[app.AspectCoverage(aspect="customer segments", answering_findings=[1])],
        source_assessments=[SA(app, "S3")],
    )
    result, _ = run(app, monkeypatch, fake_model, analysis, resp=resp)
    assert "[Not established] Not researched: the search for 'competitor pricing' failed." in result["report"].key_findings


# Location levels: Nigeria-wide evidence for an Abuja question is labelled as such.
def test_country_level_evidence_is_labelled_for_city_question(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app, summary="x.",
        findings=[C(app, "Basic AI chatbot plans in Nigeria start from ₦10,000 monthly according to a pricing guide.",
                    ids=["S3"])],
        source_assessments=[SA(app, "S3")],
    )
    result, _ = run(app, monkeypatch, fake_model, analysis)
    assert result["report"].key_findings[0].startswith("[Evidence – Nigeria-wide, not Abuja-specific]")


# Planner prompt asks for page-style queries, not question-style phrases.
def test_planner_prompt_asks_for_page_style_queries(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app, summary="x.")
    plan = make_plan(app, [("competitor pricing", "q-pricing")], location="Abuja, Nigeria")
    planner = fake_model(output=plan)
    monkeypatch.setattr(app, "planner", planner)
    monkeypatch.setattr(app, "analyst", fake_model(output=analysis))
    monkeypatch.setattr(app, "run_search", lambda q, c=None: responses()[q])
    app.app.invoke({"question": QUESTION, "search_results": "", "report": None})
    assert "Write each query the way the pages you want are worded" in planner.last_prompt
    assert "pricing <country> per month" in planner.last_prompt
