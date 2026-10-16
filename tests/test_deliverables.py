"""Tests for campaign brief / ad copy / content calendar. SYNTHETIC report; no real API calls."""
import pytest
from fastapi.testclient import TestClient

REPORT = {  # synthetic report, as the website would send it back
    "question": "How should BrightSmile market teeth whitening in Lekki?",
    "summary": "Clinics in Lekki advertise whitening from ₦45,000.\n\nFew publish evening hours.",
    "key_findings": [
        "[Vendor claim] AcmeDentX says its whitening lasts 2 years. [1]",
        "[Evidence] Lekki clinics advertise whitening from ₦45,000. [2]",
        "[Not established] No source showed how many people search for whitening in Lekki.",
    ],
    "opportunities": ["[Inference] Evening appointments are rarely advertised. [2]"],
    "recommendations": ["Advertise evening slots to busy professionals. (Based on finding 2)"],
}


class Recorder:
    def __init__(self, output):
        self.output, self.prompts = output, []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return self.output


@pytest.fixture
def dl(app_module):
    import deliverables
    return deliverables


def patch_writer(monkeypatch, dl, kind, output):
    writer = Recorder(output)
    monkeypatch.setitem(dl.writers, kind, writer)
    return writer


def test_campaign_brief_keeps_report_figures_and_drops_invented_ones(dl, monkeypatch):
    brief = dl.CampaignBrief(
        objective="Fill evening whitening slots. Grow bookings by 300%.",
        audience="Busy professionals in Lekki.",
        key_message="Whitening after work, from ₦45,000.",
        proof_points=["Lekki clinics advertise whitening from ₦45,000.", "Results last 2 years for 98% of clients."],
        offer="Evening whitening. Launch price ₦30,000.",
        channels=["Instagram: where Lekki professionals browse after work."],
        success_measures=["Evening bookings per week."],
        timeline=["Weeks 1-2: teaser posts.", "Weeks 3-4: offer push."],
        open_questions=["How many people search for whitening in Lekki?"],
    )
    writer = patch_writer(monkeypatch, dl, "campaign_brief", brief)
    out = dl.make_deliverable("campaign_brief", REPORT, profile={"business_name": "BrightSmile (synthetic)"})
    sections = {s["heading"]: s["items"] for s in out["sections"]}

    assert sections["Objective"] == ["Fill evening whitening slots."]              # 300% removed
    assert sections["Key message"] == ["Whitening after work, from ₦45,000."]      # report figure kept
    assert sections["Proof points"] == ["Lekki clinics advertise whitening from ₦45,000."]
    assert sections["Offer"] == ["Evening whitening."]                            # invented ₦30,000 removed
    assert sections["Timeline"] == ["Weeks 1-2: teaser posts.", "Weeks 3-4: offer push."]   # small counts OK
    assert out["title"] == "Campaign brief" and out["text"].startswith("Campaign brief\n\nObjective")
    prompt = writer.prompts[0]
    assert "BrightSmile (synthetic)" in prompt and "[Vendor claim] AcmeDentX" in prompt
    assert "never repeat\n  them as proven results" in prompt


def test_ad_copy_drops_invented_stats(dl, monkeypatch):
    ads = dl.AdCopy(variants=[
        dl.AdVariant(channel="Instagram", angle="convenience", headline="Whiter smile after work",
                     body="Book an evening slot in Lekki. 9 out of 10 clients love it. Book in 2 minutes.",
                     call_to_action="Book on WhatsApp", based_on="Evening appointments are rarely advertised."),
        dl.AdVariant(channel="Google Search", angle="price", headline="Whitening from ₦45,000",
                     body="Clear prices, no surprises.", call_to_action="See prices", based_on="Finding 2"),
    ])
    patch_writer(monkeypatch, dl, "ad_copy", ads)
    out = dl.make_deliverable("ad_copy", REPORT)
    first, second = out["sections"]
    assert first["heading"] == "Ad 1 · Instagram · convenience"
    assert first["items"][1] == "Book an evening slot in Lekki. Book in 2 minutes."   # '9 out of 10' removed
    assert second["items"][0] == "Headline: Whitening from ₦45,000"


def test_content_calendar_has_four_weeks(dl, monkeypatch):
    week = lambda n: dl.CalendarWeek(week=n, theme=f"Theme {n}", posts=[           # noqa: E731
        dl.CalendarPost(day="Monday", channel="Instagram", format="Reel", idea="Evening whitening behind the scenes.",
                        hook="Ever whitened your teeth after work?")])
    cal = dl.ContentCalendar(weeks=[week(n) for n in range(1, 6)])                 # model gave 5 weeks
    patch_writer(monkeypatch, dl, "content_calendar", cal)
    out = dl.make_deliverable("content_calendar", REPORT)
    assert [s["heading"] for s in out["sections"]] == ["Week 1: Theme 1", "Week 2: Theme 2", "Week 3: Theme 3", "Week 4: Theme 4"]
    assert out["sections"][0]["items"][0].startswith("Monday · Instagram · Reel: Evening whitening")


def test_figure_rules(dl):
    allowed = {"45000"}
    assert dl.figures_ok("Weeks 1-2: teasers.", allowed)
    assert dl.figures_ok("Book in 2 minutes.", allowed)
    assert dl.figures_ok("From ₦45,000.", allowed)
    for invented in ("Get 3x more bookings.", "Save 40% today.", "From ₦30,000.", "Join 500 clients.", "₦12 off.",
                     "9 out of 10 clients love it.", "1 in 5 people prefer evenings."):
        assert not dl.figures_ok(invented, allowed), invented


# ---------------------------------------------------------------- API

def test_deliverable_api(dl, monkeypatch):
    import api
    patch_writer(monkeypatch, dl, "campaign_brief", dl.CampaignBrief(
        objective="Fill evening slots.", audience="Professionals.", key_message="After-work whitening.",
        proof_points=[], offer="Evening whitening.", channels=[], success_measures=[], timeline=[], open_questions=[]))
    http = TestClient(api.api)
    ok = http.post("/deliverable", json={"kind": "campaign_brief", "report": REPORT})
    assert ok.status_code == 200
    assert set(ok.json()) == {"kind", "title", "sections", "text"}
    assert http.post("/deliverable", json={"kind": "poem", "report": REPORT}).status_code == 422
    assert http.post("/deliverable", json={"kind": "ad_copy", "report": {"summary": " "}}).status_code == 422
    assert http.post("/deliverable", json={"kind": "ad_copy",
                                           "report": dict(REPORT, key_findings=["x"] * 41)}).status_code == 422


def test_deliverable_counts_toward_the_hourly_limit(dl, monkeypatch):
    import api
    import ratelimit
    monkeypatch.setattr(ratelimit, "limiter", ratelimit.RateLimiter(per_ip_per_hour=1, daily_total=0))
    patch_writer(monkeypatch, dl, "ad_copy", dl.AdCopy(variants=[]))
    http = TestClient(api.api)
    headers = {"X-Forwarded-For": "203.0.113.7"}
    assert http.post("/deliverable", json={"kind": "ad_copy", "report": REPORT}, headers=headers).status_code == 200
    assert http.post("/deliverable", json={"kind": "ad_copy", "report": REPORT}, headers=headers).status_code == 429
