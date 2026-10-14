"""Regression tests for problems seen in the third real run (14 Oct 2026).

SYNTHETIC data only (fictional companies on .example domains). No real API calls.
"""
from test_pipeline import A, C, SA, make_plan, run_pipeline

QUESTION = (
    "I want to market AI chatbot and WhatsApp automation services to beauty businesses "
    "in Abuja, Nigeria. Which customer segments are most promising, who are the "
    "competitors and what is their pricing?"
)

SYNTHETIC = {
    "q-industry": [
        {"title": "Nigeria beauty industry blog (synthetic)",
         "url": "https://industry-blog.example/nigeria-beauty",
         "content": "Across Nigeria, from Lagos to Abuja and Kano, local alternatives often outsell imports "
                    "because they fit household budgets, showing strong price sensitivity."},
    ],
    "q-providers": [
        {"title": "SiluxX Chat Abuja (synthetic vendor)",
         "url": "https://siluxx.example/locations/nigeria/abuja",
         "content": "SiluxX Chat provides AI chatbot, live chat and WhatsApp automation for businesses in Abuja."},
        {"title": "DMlyX WhatsApp automation (synthetic vendor)",
         "url": "https://dmlyx.example/ng/whatsapp-automation",
         "content": "DMlyX WhatsApp automation plans start from ₦29,000 a month for Nigerian businesses."},
    ],
}

SYNTHETIC["q-segments"] = [
    {"title": "Generic small business tips (synthetic, off-topic)",
     "url": "https://tips.example/small-business",
     "content": "Small businesses should keep clear records and review cash flow monthly."},
]

TASKS = [("market", "q-industry"), ("competitors", "q-providers"), ("customer segments", "q-segments")]


def responses():
    return {q: {"query": q, "results": r} for q, r in SYNTHETIC.items()}


def run(app, monkeypatch, fake_model, analysis):
    plan = make_plan(app, TASKS, location="Abuja, Nigeria", country="nigeria")
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, plan, responses(), analysis, question=QUESTION)
    return result["report"]


# Bug 1: a Nigeria-wide article that merely mentions Abuja was shown as Abuja evidence,
#        and led the summary as "Independent evidence about Abuja".
def test_country_article_mentioning_city_is_not_city_evidence(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        findings=[C(app, "Local alternatives often outsell imports because they fit household budgets.", ids=["S1"])],
        source_assessments=[SA(app, "S1")],
    )
    report = run(app, monkeypatch, fake_model, analysis)
    assert report.key_findings[0].startswith("[Evidence – Nigeria-wide, not Abuja-specific]")


def test_claim_about_the_city_with_city_source_is_still_city_evidence(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        findings=[C(app, "In Abuja, local alternatives often outsell imports because they fit household budgets.",
                    ids=["S1"])],
        source_assessments=[SA(app, "S1")],
    )
    report = run(app, monkeypatch, fake_model, analysis)
    assert report.key_findings[0].startswith("[Evidence] In Abuja")


# Bug 2: five competitors were listed, yet the report also said
#        "No usable evidence was found for: competitors".
def test_competitors_found_are_not_reported_as_missing(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        findings=[
            C(app, "SiluxX Chat provides AI chatbot and WhatsApp automation for businesses in Abuja.",
              kind="vendor_claim", ids=["S2"]),
            C(app, "DMlyX WhatsApp automation plans start from ₦29,000 a month.", kind="vendor_claim", ids=["S3"]),
        ],
        aspect_coverage=[],                       # analyst skipped the coverage list
        source_assessments=[SA(app, "S2", source_type="company_or_vendor_page"),
                            SA(app, "S3", source_type="company_or_vendor_page")],
    )
    report = run(app, monkeypatch, fake_model, analysis)
    assert not any("for: competitors" in f for f in report.key_findings)
    assert "[Not established] No usable evidence was found for: customer segments." in report.key_findings


def test_reworded_aspect_names_still_match(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        findings=[C(app, "SiluxX Chat provides AI chatbot and WhatsApp automation for businesses in Abuja.",
                    kind="vendor_claim", ids=["S2"])],
        aspect_coverage=[app.AspectCoverage(aspect="Competitors in Abuja", answering_findings=[1])],
        source_assessments=[SA(app, "S2", source_type="company_or_vendor_page")],
    )
    report = run(app, monkeypatch, fake_model, analysis)
    assert not any("for: competitors" in f for f in report.key_findings)


def test_gap_already_reported_by_analyst_is_not_repeated(app_module, monkeypatch, fake_model):
    app = app_module
    analysis = A(app,
        findings=[C(app, "SiluxX Chat provides AI chatbot and WhatsApp automation for businesses in Abuja.",
                    kind="vendor_claim", ids=["S2"])],
        unknowns=["No specific customer segments were identified from the sources."],
        source_assessments=[SA(app, "S2", source_type="company_or_vendor_page")],
    )
    report = run(app, monkeypatch, fake_model, analysis)
    segment_lines = [f for f in report.key_findings if "customer segments" in f]
    assert segment_lines == ["[Not established] No specific customer segments were identified from the sources."]


def test_explicit_analyst_answer_is_still_checked(app_module, monkeypatch, fake_model):
    """If the analyst names a finding that failed the checks, the gap is still reported."""
    app = app_module
    analysis = A(app,
        findings=[C(app, "Salons are the most promising segment with 70% adoption.", ids=["S1"])],  # figure rejected
        aspect_coverage=[app.AspectCoverage(aspect="customer segments", answering_findings=[1])],
        source_assessments=[SA(app, "S1")],
    )
    report = run(app, monkeypatch, fake_model, analysis)
    assert "[Not established] No usable evidence was found for: customer segments." in report.key_findings


def test_reworded_aspect_name_is_matched_even_when_source_came_from_another_search(
        app_module, monkeypatch, fake_model):
    """Only the analyst's (reworded) coverage list can link this finding to 'competitors'."""
    app = app_module
    analysis = A(app,
        findings=[C(app, "SiluxX Chat provides AI chatbot and WhatsApp automation for businesses in Abuja.",
                    kind="vendor_claim", ids=["S1"])],
        aspect_coverage=[app.AspectCoverage(aspect="Competitors in Abuja", answering_findings=[1])],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    plan = make_plan(app, [("market", "q-providers"), ("competitors", "q-segments")],
                     location="Abuja, Nigeria", country="nigeria")
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, plan, responses(), analysis, question=QUESTION)
    assert not any("for: competitors" in f for f in result["report"].key_findings)
