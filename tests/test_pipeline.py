"""Offline regression tests for the research pipeline.

All search results and model outputs are SYNTHETIC (see fixtures/). No test
makes a real OpenAI or Tavily call: `app.run_search`, `app.planner` and
`app.analyst` are replaced with fakes, and the network is blocked in conftest.
"""
import pytest

QUESTION = (
    "Which customer segments should I target with WhatsApp booking automation "
    "in Lagos, Nigeria? Who are the competitors and what is their pricing?"
)


# ---------------------------------------------------------------- helpers

def make_plan(app, tasks, location="Lagos, Nigeria", country="nigeria"):
    return app.ResearchPlan(
        target_location=location,
        country=country,
        tasks=[app.ResearchTask(aspect=a, query=q) for a, q in tasks],
    )


def run_pipeline(app, monkeypatch, fake_model, plan, responses, analysis, question=QUESTION):
    """Run the real LangGraph graph with fake search + fake models."""
    searches = []

    def fake_search(query, country=None):
        searches.append((query, country))
        value = responses[query]
        if isinstance(value, Exception):
            raise value
        return value

    planner = fake_model(output=plan)
    analyst = fake_model(output=analysis)
    monkeypatch.setattr(app, "run_search", fake_search)
    monkeypatch.setattr(app, "planner", planner)
    monkeypatch.setattr(app, "analyst", analyst)

    result = app.app.invoke({"question": question, "search_results": "", "report": None})
    return result, analyst, searches


def tasks_from(fixture, *keys):
    return [(k, fixture[k]["query"]) for k in keys]


def responses_from(fixture, *keys):
    return {fixture[k]["query"]: {"query": fixture[k]["query"], "results": fixture[k]["results"]} for k in keys}


def A(app, **kw):
    return app.AnalysisOutput(**kw)


def C(app, text, kind="evidence", scope="target_location", ids=()):
    return app.Claim(text=text, kind=kind, scope=scope, source_ids=list(ids))


def SA(app, sid, relevant=True, source_type="independent_article_or_report"):
    return app.SourceAssessment(id=sid, relevant=relevant, source_type=source_type)


def source_id_for(result, url_part):
    return next(s["id"] for s in result["sources"] if url_part in s["url"])


# ------------------------------------------------- 1. relevant local evidence

def test_relevant_local_source_supports_finding_with_inline_citation(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("local_bookings",)
    analysis = A(app,
        summary="Lagos salon owners in the survey mostly take bookings on WhatsApp.",
        findings=[C(app, "28 of 40 surveyed Lagos salon owners take most bookings through WhatsApp messages.", ids=["S1"])],
        source_assessments=[SA(app, "S1")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    finding = result["report"].key_findings[0]
    assert finding.startswith("[Evidence] 28 of 40 surveyed Lagos salon owners")
    assert finding.endswith("[1]")                     # footnote number, not a web address
    assert result["report"].sources == ["https://lagos-salon-survey.example/report?utm_source=newsletter"]


# ------------------------------------------------- 2. irrelevant source

def test_irrelevant_source_cannot_support_a_claim(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("local_bookings",)
    analysis = A(app,
        summary="x.",
        findings=[
            # cites the car-dealership page, marked irrelevant by the analyst
            C(app, "Salon owners rotate vehicle inventory every quarter.", ids=["S2"]),
        ],
        source_assessments=[SA(app, "S1"), SA(app, "S2", relevant=False, source_type="other")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    finding = result["report"].key_findings[0]
    assert finding.startswith("[Inference]")
    assert "cars.example" not in finding
    assert result["report"].sources == []


# ------------------------------------------------- 3. international != local

def test_international_evidence_is_not_presented_as_local(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("international",)
    analysis = A(app,
        summary="Chatbots reduce missed enquiries.",
        findings=[C(app, "Chatbots reduced missed booking enquiries by 35 percent for salons.", ids=["S1"])],
        source_assessments=[SA(app, "S1")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    finding = result["report"].key_findings[0]
    assert finding.startswith("[Evidence – not specific to target location]")
    assert finding.endswith("[1]")
    assert result["report"].sources == ["https://us-smb-study.example/chatbots"]


# ------------------------------------------------- 4. competitor offer != demand

def test_competitor_offering_is_not_treated_as_customer_demand(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("competitors",)
    analysis = A(app,
        summary="x.",
        findings=[C(app, "Nigerian businesses want WhatsApp chatbots with automated booking reminders.", ids=["S1"])],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    assert result["report"].key_findings[0].startswith("[Vendor claim – Nigeria-wide, not Lagos-specific]")


# ------------------------------------------------- 5. missing pricing

def test_missing_pricing_is_reported_honestly(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("competitors",)
    analysis = A(app,
        summary="ChatFlowX offers WhatsApp chatbots.",
        findings=[
            C(app, "ChatFlowX offers WhatsApp chatbots and automated booking reminders.", ids=["S1"]),
            C(app, "ChatFlowX charges ₦50,000 per month for WhatsApp chatbots.", ids=["S1"]),
        ],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    findings = result["report"].key_findings
    assert any(f.startswith("[Not established] ChatFlowX charges ₦50,000") for f in findings)
    assert "[Not established] No published prices were found in the retrieved sources." in findings


def test_published_price_is_kept_and_no_false_pricing_gap(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("with_price",)
    analysis = A(app,
        summary="BotPal publishes a starter price.",
        findings=[C(app, "BotPal's Starter plan costs ₦25,000 per month for one business line.",
                    kind="vendor_claim", scope="other_or_unspecified", ids=["S1"])],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    findings = result["report"].key_findings
    assert findings[0].startswith("[Vendor claim – Nigeria-wide, not Lagos-specific] BotPal's Starter plan costs ₦25,000")
    assert not any("No published prices" in f for f in findings)


# ------------------------------------------------- 6. unsupported statistics

def test_unsupported_statistic_is_not_presented_as_fact(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("competitors",)
    analysis = A(app,
        summary="Chatbots cut service workload by 60%. ChatFlowX offers WhatsApp chatbots.",
        findings=[
            C(app, "WhatsApp chatbots reduce service workload by 60% for ChatFlowX customers.", ids=["S1"]),
            C(app, "ChatFlowX customers capture 3x more leads with WhatsApp chatbots.", ids=["S1"]),
        ],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    report = result["report"]
    assert report.key_findings[0].startswith("[Not established] WhatsApp chatbots reduce service workload by 60%")
    # figure exists in the source, but only as the vendor's own claim
    assert report.key_findings[1].startswith("[Vendor claim – Nigeria-wide, not Lagos-specific] ChatFlowX customers capture 3x more leads")
    assert "60%" not in report.summary


# ------------------------------------------------- 7. missing / duplicate sources

def test_duplicate_urls_are_merged_and_unretrieved_ids_are_dropped(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("competitors",)
    analysis = A(app,
        summary="x.",
        findings=[C(app, "ChatFlowX offers automated booking reminders for Nigerian businesses.",
                    kind="vendor_claim", scope="other_or_unspecified", ids=["S1", "S9"])],
        source_assessments=[SA(app, "S1", source_type="company_or_vendor_page")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    assert len(result["sources"]) == 1  # www. + trailing slash duplicate merged
    finding = result["report"].key_findings[0]
    assert "S9" not in finding and finding.endswith("[1]")
    assert result["report"].sources == ["https://chatflowx.example/features"]


# ------------------------------------------------- 8. failed / empty searches

@pytest.mark.parametrize("bad_response", [
    {"error": ValueError("Error 432: usage limit exceeded")},       # langchain-tavily API error shape
    "No search results found for 'x'. Suggestions: ...",              # langchain-tavily empty-result shape
    {"query": "x", "results": []},
    {"query": "x", "results": [{"title": "no url"}]},
    TimeoutError("network down"),
])
def test_failed_or_empty_search_never_produces_a_report(app_module, monkeypatch, fake_model, bad_response):
    app = app_module
    plan = make_plan(app, [("competitors", "q1"), ("pricing", "q2")])
    result, analyst, searches = run_pipeline(
        app, monkeypatch, fake_model, plan, {"q1": bad_response, "q2": bad_response},
        analysis=A(app, summary="SHOULD NEVER BE USED"),
    )
    report = result["report"]
    assert analyst.calls == 0                       # no paid analysis call
    assert report.summary.startswith("The research could not be completed")
    assert report.sources == [] and report.opportunities == []
    assert all(f.startswith("[Not established]") for f in report.key_findings)


def test_partial_failure_is_disclosed_to_the_analyst(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    plan = make_plan(app, [("local_bookings", synthetic["local_bookings"]["query"]), ("pricing", "q-fail")])
    responses = responses_from(synthetic, "local_bookings")
    responses["q-fail"] = {"error": ValueError("boom")}
    result, analyst, _ = run_pipeline(app, monkeypatch, fake_model, plan, responses,
                                      A(app, summary="x.", source_assessments=[SA(app, "S1")]))
    assert analyst.calls == 1
    assert "These searches FAILED" in analyst.last_prompt and "q-fail" in analyst.last_prompt
    assert result["failed_queries"][0]["aspect"] == "pricing"


# ------------------------------------------------- 9. citation does not support claim

def test_citation_that_does_not_mention_the_claim_is_removed(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("local_bookings",)
    analysis = A(app,
        summary="x.",
        findings=[C(app, "Restaurants in Lagos prefer email newsletters for promotions.", ids=["S1"])],
        source_assessments=[SA(app, "S1")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    finding = result["report"].key_findings[0]
    assert finding.startswith("[Inference]") and "Source:" not in finding


# ------------------------------------------------- 10. categories kept separate

def test_report_separates_evidence_inference_recommendations_and_unknowns(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    keys = ("local_bookings", "competitors")
    analysis = A(app,
        summary="Survey shows WhatsApp is the main booking channel.",
        findings=[
            C(app, "28 of 40 surveyed Lagos salon owners take most bookings through WhatsApp messages.", ids=["S1"]),
            C(app, "ChatFlowX offers automated booking reminders for Nigerian businesses.",
              kind="vendor_claim", scope="other_or_unspecified", ids=["S3"]),
        ],
        opportunities=[C(app, "Salons losing appointments from missed messages may value automated replies.",
                         kind="inference", scope="target_location", ids=["S1"])],
        recommendations=[
            app.Recommendation(text="Pilot automated WhatsApp replies with five salons.", based_on_findings=[1]),
            app.Recommendation(text="Run Instagram ads.", based_on_findings=[]),
            app.Recommendation(text="Bundle CRM.", based_on_findings=[7]),
        ],
        unknowns=["Competitor prices in Lagos."],
        source_assessments=[SA(app, "S1"), SA(app, "S2", relevant=False, source_type="other"),
                            SA(app, "S3", source_type="company_or_vendor_page")],
    )
    result, _, _ = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks_from(synthetic, *keys)),
                                responses_from(synthetic, *keys), analysis)
    r = result["report"]
    assert r.key_findings[0].startswith("[Evidence]")
    assert r.key_findings[1].startswith("[Vendor claim – Nigeria-wide, not Lagos-specific]")
    assert "[Not established] Competitor prices in Lagos." in r.key_findings
    assert r.opportunities[0].startswith("[Inference]") and r.opportunities[0].endswith("[1]")
    assert r.recommendations[0].endswith("(Based on finding 1)")
    assert r.recommendations[1].startswith("[Untested idea]")
    assert r.recommendations[2].startswith("[Untested idea]")   # finding 7 does not exist
    assert r.summary.startswith("Survey shows WhatsApp is the main booking channel.")
    assert "Not established from the available sources: Competitor prices in Lagos" in r.summary
    assert r.sources == [
        "https://lagos-salon-survey.example/report?utm_source=newsletter",
        "https://chatflowx.example/features",
    ]


# ------------------------------------------------- planner + search plumbing

def test_planner_failure_falls_back_to_one_search(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    searches = []
    monkeypatch.setattr(app, "planner", fake_model(error=RuntimeError("planner down")))
    monkeypatch.setattr(app, "analyst", fake_model(output=A(app, summary="x.")))
    monkeypatch.setattr(app, "run_search", lambda q, c=None: searches.append(q) or
                        {"results": synthetic["local_bookings"]["results"]})
    app.app.invoke({"question": "short question", "search_results": "", "report": None})
    assert searches == ["short question"]


def test_planner_duplicates_removed_and_capped(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    tasks = [("a", "Same Query"), ("b", "same   query")] + [(f"t{i}", f"q{i}") for i in range(10)]
    responses = {q: {"results": synthetic["local_bookings"]["results"]} for _, q in tasks}
    _, _, searches = run_pipeline(app, monkeypatch, fake_model, make_plan(app, tasks), responses,
                                  A(app, summary="x."))
    queries = [q for q, _ in searches]
    assert len(queries) == app.MAX_SEARCHES
    assert sum(1 for q in queries if q.lower().split() == ["same", "query"]) == 1


def test_bad_country_retries_once_without_country(app_module, monkeypatch, fake_model, synthetic):
    app = app_module
    calls = []

    def fake_search(query, country=None):
        calls.append(country)
        if country:
            return {"error": ValueError("Error 400: invalid country")}
        return {"results": synthetic["local_bookings"]["results"]}

    monkeypatch.setattr(app, "planner", fake_model(output=make_plan(app, [("x", "q")], country="atlantis")))
    monkeypatch.setattr(app, "analyst", fake_model(output=A(app, summary="x.")))
    monkeypatch.setattr(app, "run_search", fake_search)
    result = app.app.invoke({"question": QUESTION, "search_results": "", "report": None})
    assert calls == ["atlantis", None]
    assert len(result["sources"]) == 2


def test_classify_search_result_shapes(app_module):
    app = app_module
    ok, err = app.classify_search_result({"results": [{"url": "https://a.example", "content": "c"}]})
    assert ok and err is None
    for bad in [{"error": "x"}, "No search results found", {"results": []}, None, 42]:
        items, reason = app.classify_search_result(bad)
        assert items is None and reason


# ------------------------------------------------- API contract unchanged

def test_api_response_shape_is_unchanged(app_module, monkeypatch, fake_model, synthetic):
    from fastapi.testclient import TestClient
    import api

    app = app_module
    keys = ("local_bookings",)
    monkeypatch.setattr(app, "planner", fake_model(output=make_plan(app, tasks_from(synthetic, *keys))))
    monkeypatch.setattr(app, "analyst", fake_model(output=A(app, summary="ok.", source_assessments=[SA(app, "S1")])))
    monkeypatch.setattr(app, "run_search", lambda q, c=None: responses_from(synthetic, *keys)[q])

    response = TestClient(api.api).post("/research", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"question", "summary", "key_findings", "opportunities", "recommendations", "sources"}
    assert all(isinstance(body[k], list) for k in ("key_findings", "opportunities", "recommendations", "sources"))
