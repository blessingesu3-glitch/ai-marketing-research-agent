"""Tests for the optional business profile. SYNTHETIC data only; no real API calls."""
import pytest
from fastapi.testclient import TestClient

from test_pipeline import A, C, SA

PROFILE = {   # synthetic business, invented for tests
    "business_name": "BrightSmile Dental (synthetic)",
    "industry": "Dental clinic",
    "offering": "teeth whitening",
    "location": "Lekki, Lagos, Nigeria",
    "target_customer": "young professionals",
    "goal": "more bookings",
    "competitors": "AcmeDentX, https://www.betadentx.example/about",
    "price_point": "mid-range",
}

RESULTS = {"results": [{"title": "Teeth whitening Lagos (synthetic)", "url": "https://clinic.example/whitening",
                        "content": "Teeth whitening clinics in Lagos, Nigeria offer in-office whitening."}]}


class Recorder:
    def __init__(self, output):
        self.output, self.prompts = output, []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return self.output


def run(app, monkeypatch, question, profile, plan=None, analysis=None):
    plan = plan or app.ResearchPlan(tasks=[app.ResearchTask(aspect="market", query="q-market")])
    planner, analyst = Recorder(plan), Recorder(analysis or A(app, summary="x."))
    searched = []
    monkeypatch.setattr(app, "planner", planner)
    monkeypatch.setattr(app, "analyst", analyst)
    monkeypatch.setattr(app, "run_search", lambda q, c=None: searched.append((q, c)) or RESULTS)
    result = app.app.invoke({"question": question, "profile": profile, "search_results": "", "report": None})
    return result, planner, analyst, searched


def test_no_profile_behaves_exactly_as_before(app_module, monkeypatch):
    app = app_module
    result, planner, analyst, searched = run(app, monkeypatch, "Which keywords should I target?", None)
    assert "user's business" not in planner.prompts[0]
    assert "About the user's own business" not in analyst.prompts[0]
    assert result["plan"]["target_location"] is None and searched == [("q-market", None)]


def test_empty_profile_is_ignored(app_module, monkeypatch):
    app = app_module
    _, planner, analyst, _ = run(app, monkeypatch, "Which keywords should I target?",
                                 {k: "  " for k in PROFILE})
    assert "user's business" not in planner.prompts[0]
    assert "About the user's own business" not in analyst.prompts[0]


def test_profile_reaches_planner_and_analyst(app_module, monkeypatch):
    app = app_module
    _, planner, analyst, _ = run(app, monkeypatch, "Which keywords should I target?", PROFILE)
    for prompt in (planner.prompts[0], analyst.prompts[0]):
        assert "BrightSmile Dental (synthetic)" in prompt and "young professionals" in prompt
    assert "It is NOT evidence" in analyst.prompts[0]           # profile is context, not a source
    assert "every recommendation must\n  fit that business" in analyst.prompts[0]


def test_profile_location_and_offering_used_when_question_has_none(app_module, monkeypatch):
    app = app_module
    result, _, _, searched = run(app, monkeypatch, "What are my customers' main complaints?", PROFILE)
    plan = result["plan"]
    assert plan["target_location"] == "Lekki, Lagos, Nigeria"
    assert plan["country"] == "nigeria" and plan["offering"] == "teeth whitening"
    assert searched[0] == ("q-market", "nigeria")              # Tavily country boost applied


def test_question_location_beats_profile_location(app_module, monkeypatch):
    app = app_module
    plan = app.ResearchPlan(target_location="Accra, Ghana", country="ghana",
                            tasks=[app.ResearchTask(aspect="market", query="q-market")])
    result, _, _, _ = run(app, monkeypatch, "Should I expand to Accra, Ghana?", PROFILE, plan=plan)
    assert result["plan"]["target_location"] == "Accra, Ghana" and result["plan"]["country"] == "ghana"


def test_named_competitors_get_their_own_search(app_module, monkeypatch):
    app = app_module
    result, _, _, searched = run(app, monkeypatch, "How do I compare with my competitors?", PROFILE)
    queries = [q for q, _ in searched]
    assert "AcmeDentX betadentx.example teeth whitening pricing services" in queries
    assert any(t["aspect"] == "named competitors" for t in result["plan"]["tasks"])


def test_no_competitor_search_when_question_is_not_about_competitors(app_module, monkeypatch):
    app = app_module
    _, _, _, searched = run(app, monkeypatch, "What are my customers' main complaints?", PROFILE)
    assert not any("AcmeDentX" in q for q, _ in searched)


def test_competitor_search_stays_within_search_limit(app_module, monkeypatch):
    app = app_module
    tasks = [app.ResearchTask(aspect=f"a{i}", query=f"q{i}") for i in range(app.MAX_SEARCHES)]
    _, _, _, searched = run(app, monkeypatch, "How do I compare with my competitors?", PROFILE,
                            plan=app.ResearchPlan(tasks=tasks))
    assert len(searched) == app.MAX_SEARCHES
    assert searched[-1][0].startswith("AcmeDentX")


def test_profile_is_never_listed_as_a_source(app_module, monkeypatch):
    app = app_module
    analysis = A(app, summary="Whitening is offered across Lagos.",
                 findings=[C(app, "Teeth whitening clinics in Lagos offer in-office whitening.", ids=["S1"])],
                 source_assessments=[SA(app, "S1")])
    result, _, _, _ = run(app, monkeypatch, "Who offers teeth whitening in Lagos?", PROFILE, analysis=analysis)
    assert result["report"].sources == ["https://clinic.example/whitening"]


def test_competitor_names_parsing(app_module):
    app = app_module
    assert app.competitor_names({"competitors": "Acme; https://www.beta.com/x\nGamma and Delta"}) == \
        ["Acme", "beta.com", "Gamma", "Delta"]
    assert app.competitor_names(None) == []


# ---------------------------------------------------------------- API

@pytest.fixture
def client(app_module, monkeypatch):
    import api
    app = app_module
    monkeypatch.setattr(app, "planner", Recorder(app.ResearchPlan(tasks=[app.ResearchTask(aspect="m", query="q")])))
    monkeypatch.setattr(app, "analyst", Recorder(A(app, summary="ok.")))
    monkeypatch.setattr(app, "run_search", lambda q, c=None: RESULTS)
    return TestClient(api.api), app


def test_api_accepts_requests_with_and_without_profile(client):
    http, app = client
    assert http.post("/research", json={"question": "Which keywords should I target?"}).status_code == 200
    response = http.post("/research", json={"question": "Which keywords should I target?", "profile": PROFILE})
    assert response.status_code == 200
    assert set(response.json()) == {"question", "summary", "key_findings", "opportunities", "recommendations", "sources"}
    assert "BrightSmile" in app.analyst.prompts[-1]


def test_api_rejects_oversized_input_before_any_paid_call(client):
    http, app = client
    calls_before = len(app.analyst.prompts)
    assert http.post("/research", json={"question": "x" * 2001}).status_code == 422
    assert http.post("/research", json={"question": "hi"}).status_code == 422
    too_long = dict(PROFILE, goal="g" * 301)
    assert http.post("/research", json={"question": "Which keywords?", "profile": too_long}).status_code == 422
    assert len(app.analyst.prompts) == calls_before          # nothing reached the model
