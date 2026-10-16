"""Competitor teardown tests. SYNTHETIC companies and pages only; no real API calls."""
import pytest
from fastapi.testclient import TestClient

PAGES = {  # pages on each competitor's own (fictional) website
    "acmedentx.example": [
        {"title": "AcmeDentX pricing", "url": "https://acmedentx.example/pricing",
         "content": "Teeth whitening from ₦45,000. Braces consultation free.",
         "raw_content": "AcmeDentX offers teeth whitening from ₦45,000 and braces. Book on WhatsApp."},
        {"title": "AcmeDentX home", "url": "https://www.acmedentx.example/",
         "content": "Family dental clinic in Lekki for busy professionals.", "raw_content": None},
        {"title": "Unrelated blog", "url": "https://other-site.example/post",       # not their own site
         "content": "Dental tips.", "raw_content": None},
    ],
    "smilecox.example": [
        {"title": "SmileCoX services", "url": "https://smilecox.example/services",
         "content": "Cosmetic dentistry and whitening. Prices on request.", "raw_content": None},
    ],
}

NAME_LOOKUP = {  # results of '<name> official website ...' searches
    "SmileCoX": [
        {"title": "SmileCoX on Instagram", "url": "https://instagram.com/smilecox", "content": "SmileCoX dental"},
        {"title": "SmileCoX - Wikipedia", "url": "https://en.wikipedia.org/wiki/SmileCoX", "content": "SmileCoX dental"},
        {"title": "SmileCoX Dental", "url": "https://smilecox.example/", "content": "SmileCoX dental clinic"},
    ],
    "GhostDent": [
        {"title": "GhostDent on Facebook", "url": "https://facebook.com/ghostdent", "content": "GhostDent"},
    ],
}


class Recorder:
    def __init__(self, output=None):
        self.output, self.prompts = output, []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return self.output


@pytest.fixture
def td(app_module, monkeypatch):
    import teardown
    calls = {"search": [], "site": []}

    def fake_search(query, country=None):
        calls["search"].append((query, country))
        name = next((n for n in NAME_LOOKUP if query.startswith(n)), None)
        return {"results": NAME_LOOKUP[name]} if name else "No search results found"

    def fake_site_search(query, domain):
        calls["site"].append((query, domain))
        return {"results": PAGES[domain]} if domain in PAGES else {"results": []}

    monkeypatch.setattr(app_module, "run_search", fake_search)
    monkeypatch.setattr(teardown, "run_site_search", fake_site_search)
    return teardown, calls


def card(td_mod, name, **kw):
    base = dict(name=name, offer="o", pricing="Not found on their website", target_customer="t",
                positioning="p", channels="c", source_ids=[])
    base.update(kw)
    return td_mod.CompetitorCard(**base)


# ---------------------------------------------------------------- finding websites

def test_website_entries_are_used_directly_without_a_lookup(td):
    teardown, calls = td
    comp = teardown.find_website("https://www.AcmeDentX.example/about", None)
    assert comp["domain"] == "acmedentx.example" and calls["search"] == []


def test_name_lookup_skips_social_and_directory_sites(td):
    teardown, calls = td
    comp = teardown.find_website("SmileCoX", {"location": "Lekki, Lagos, Nigeria"})
    assert comp["domain"] == "smilecox.example"
    assert calls["search"] == [("SmileCoX official website Lekki, Lagos, Nigeria", "nigeria")]


def test_name_with_only_social_pages_is_reported_not_guessed(td):
    teardown, _ = td
    comp = teardown.find_website("GhostDent", None)
    assert comp["domain"] is None and "no own website" in comp["note"].lower()


# ---------------------------------------------------------------- reading sites

def test_only_the_competitors_own_pages_are_read(td):
    teardown, calls = td
    comp = {"name": "AcmeDentX", "domain": "acmedentx.example", "note": ""}
    pages = teardown.read_site(comp, 1, "teeth whitening")
    assert [p["url"] for p in pages] == ["https://acmedentx.example/pricing", "https://www.acmedentx.example/"]
    assert [p["id"] for p in pages] == ["C1-P1", "C1-P2"]
    assert calls["site"][0][1] == "acmedentx.example"          # search restricted to their domain


# ---------------------------------------------------------------- full teardown

def test_full_teardown_compares_competitors_and_checks_prices(td, monkeypatch):
    teardown, calls = td
    analysis = teardown.TeardownAnalysis(
        summary="AcmeDentX publishes whitening from ₦45,000.\n\nSmileCoX hides prices and charges ₦99,000.",
        competitors=[
            card(teardown, "AcmeDentX", pricing="Teeth whitening from ₦45,000", source_ids=["C1-P1"]),
            card(teardown, "SmileCoX", pricing="Whitening ₦80,000", source_ids=["C2-P1", "C1-P1"]),
        ],
        gaps=["Nobody offers evening appointments.", "Undercut at ₦30,000."],
        recommendations=["Publish clear prices online."],
    )
    analyst = Recorder(analysis)
    monkeypatch.setattr(teardown, "teardown_analyst", analyst)
    result = teardown.run_teardown(["acmedentx.example", "SmileCoX", "GhostDent"],
                                   profile={"business_name": "BrightSmile (synthetic)", "location": "Lagos, Nigeria"})

    acme, smile, ghost = result["competitors"]
    assert acme["pricing"] == "Teeth whitening from ₦45,000"                  # on their own page: kept
    assert smile["pricing"] == teardown.NOT_FOUND                             # ₦80,000 not on their pages
    assert "removed" in smile["note"]
    assert smile["sources"] == ["https://smilecox.example/services"]          # C1-P1 is not theirs
    assert ghost["offer"] == teardown.NOT_FOUND and ghost["website"] == ""
    assert result["summary"] == "AcmeDentX publishes whitening from ₦45,000."  # unverified ₦99,000 sentence dropped
    assert result["gaps"] == ["Nobody offers evening appointments."]          # ₦30,000 not on any page
    assert "BrightSmile (synthetic)" in analyst.prompts[0] and "NOT evidence" in analyst.prompts[0]
    assert len(analyst.prompts) == 1                                          # one model call
    assert len(calls["site"]) == 2                                            # GhostDent: no site to read


def test_no_model_call_when_no_website_could_be_read(td, monkeypatch):
    teardown, _ = td
    analyst = Recorder(teardown.TeardownAnalysis())
    monkeypatch.setattr(teardown, "teardown_analyst", analyst)
    result = teardown.run_teardown(["GhostDent", "unknown-site.example"])
    assert analyst.prompts == []
    assert result["summary"].startswith("None of the competitors' websites could be read")
    assert all(c["pricing"] == teardown.NOT_FOUND for c in result["competitors"])


def test_at_most_three_competitors_and_duplicates_removed(td, monkeypatch):
    teardown, calls = td
    monkeypatch.setattr(teardown, "teardown_analyst", Recorder(teardown.TeardownAnalysis()))
    result = teardown.run_teardown(["acmedentx.example", "acmedentx.example", "a.example", "b.example", "c.example"])
    assert [c["website"] for c in result["competitors"]] == ["acmedentx.example", "a.example", "b.example"]


def test_card_matched_by_partial_name(td, monkeypatch):
    teardown, _ = td
    analysis = teardown.TeardownAnalysis(competitors=[card(teardown, "SmileCoX Dental Clinic", offer="Cosmetic dentistry",
                                                           source_ids=["C1-P1"])])
    monkeypatch.setattr(teardown, "teardown_analyst", Recorder(analysis))
    result = teardown.run_teardown(["SmileCoX"])
    assert result["competitors"][0]["offer"] == "Cosmetic dentistry"


# ---------------------------------------------------------------- API

def test_teardown_api(td, monkeypatch):
    import api
    teardown, _ = td
    monkeypatch.setattr(teardown, "teardown_analyst", Recorder(teardown.TeardownAnalysis(
        summary="ok.", competitors=[card(teardown, "AcmeDentX", source_ids=["C1-P1"])])))
    http = TestClient(api.api)
    ok = http.post("/teardown", json={"competitors": ["acmedentx.example"], "focus": "pricing"})
    assert ok.status_code == 200
    body = ok.json()
    assert set(body) == {"summary", "competitors", "gaps", "recommendations", "sources"}
    assert set(body["competitors"][0]) == {"name", "website", "offer", "pricing", "target_customer",
                                           "positioning", "channels", "sources", "note"}
    assert http.post("/teardown", json={"competitors": []}).status_code == 422
    assert http.post("/teardown", json={"competitors": ["a", "b", "c", "d"]}).status_code == 422
    assert http.post("/teardown", json={"competitors": ["x" * 201]}).status_code == 422
    assert http.post("/teardown", json={"competitors": ["   "]}).status_code == 422
