"""Credit protection tests. No real API calls: the research and teardown work is faked."""
import pytest
from fastapi.testclient import TestClient


class FakeClock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def setup(app_module, monkeypatch):
    import api
    import ratelimit
    import teardown

    clock = FakeClock()
    limiter = ratelimit.RateLimiter(per_ip_per_hour=3, daily_total=5, clock=clock)
    monkeypatch.setattr(ratelimit, "limiter", limiter)

    runs = {"research": 0, "teardown": 0}

    class FakeGraph:
        def invoke(self, state):
            runs["research"] += 1
            return {"question": state["question"], "report": app_module.ResearchReport(
                summary="ok", key_findings=[], opportunities=[], recommendations=[], sources=[])}

    def fake_teardown(entries, profile=None, focus=None):
        runs["teardown"] += 1
        return {"summary": "ok", "competitors": [], "gaps": [], "recommendations": [], "sources": []}

    monkeypatch.setattr(api, "research_graph", FakeGraph())
    monkeypatch.setattr(api, "run_teardown", fake_teardown)
    return TestClient(api.api), clock, runs, limiter


def research(http, ip="203.0.113.7"):
    return http.post("/research", json={"question": "Which keywords should I target?"},
                     headers={"X-Forwarded-For": ip})


def test_visitor_limit_blocks_before_any_paid_work(setup):
    http, _, runs, _ = setup
    assert [research(http).status_code for _ in range(3)] == [200, 200, 200]
    blocked = research(http)
    assert blocked.status_code == 429
    assert runs["research"] == 3                                   # 4th run never started
    assert "limit of 3 reports per hour" in blocked.json()["detail"]
    assert 0 < int(blocked.headers["Retry-After"]) <= 3600


def test_limit_shared_between_research_and_teardown(setup):
    http, _, runs, _ = setup
    research(http)
    research(http)
    ok = http.post("/teardown", json={"competitors": ["acme.example"]}, headers={"X-Forwarded-For": "203.0.113.7"})
    assert ok.status_code == 200
    blocked = http.post("/teardown", json={"competitors": ["acme.example"]}, headers={"X-Forwarded-For": "203.0.113.7"})
    assert blocked.status_code == 429 and runs["teardown"] == 1


def test_other_visitors_are_not_affected(setup):
    http, _, _, _ = setup
    for _ in range(3):
        research(http, "203.0.113.7")
    assert research(http, "203.0.113.7").status_code == 429
    assert research(http, "198.51.100.9").status_code == 200


def test_limit_frees_up_after_an_hour(setup):
    http, clock, _, _ = setup
    for _ in range(3):
        research(http)
    clock.now += 3599
    assert research(http).status_code == 429
    clock.now += 2
    assert research(http).status_code == 200


def test_daily_limit_covers_all_visitors_together(setup):
    http, clock, runs, _ = setup
    for i in range(5):                                             # 5 different visitors
        assert research(http, f"198.51.100.{i}").status_code == 200
    blocked = research(http, "192.0.2.50")                         # brand-new visitor
    assert blocked.status_code == 429 and "daily limit" in blocked.json()["detail"]
    assert runs["research"] == 5
    clock.now += 24 * 3600
    assert research(http, "192.0.2.50").status_code == 200


def test_invalid_requests_do_not_use_up_the_quota(setup):
    http, _, runs, _ = setup
    for _ in range(10):
        assert http.post("/research", json={"question": "hi"}, headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 422
        assert http.post("/teardown", json={"competitors": []}, headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 422
        assert http.post("/teardown", json={"competitors": ["  "]}, headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 422
    assert research(http).status_code == 200 and runs["research"] == 1


def test_faked_forwarded_header_cannot_dodge_the_limit(setup):
    """A visitor can type their own X-Forwarded-For; the proxy appends the real IP last."""
    http, _, _, _ = setup
    for i in range(3):
        assert research(http, f"10.0.0.{i}, 203.0.113.7").status_code == 200
    assert research(http, "10.9.9.9, 203.0.113.7").status_code == 429


def test_health_check_is_never_limited(setup):
    http, _, _, _ = setup
    for _ in range(3):
        research(http)
    assert all(http.get("/").status_code == 200 for _ in range(20))


def test_limits_can_be_switched_off(app_module):
    import ratelimit
    off = ratelimit.RateLimiter(per_ip_per_hour=0, daily_total=0, clock=FakeClock())
    assert all(off.check_and_record("1.2.3.4")[0] for _ in range(1000))


def test_wait_message_is_readable(app_module):
    import ratelimit
    assert ratelimit._human(30) == "1 minute"
    assert ratelimit._human(600) == "10 minutes"
    assert ratelimit._human(5000) == "2 hours"


def test_simultaneous_requests_cannot_slip_past_the_limit(app_module):
    """FastAPI runs requests on several threads at once; the count must stay exact."""
    import threading
    import ratelimit
    limiter = ratelimit.RateLimiter(per_ip_per_hour=10, daily_total=0, clock=FakeClock())
    results, start = [], threading.Barrier(50)

    def hit():
        start.wait()
        results.append(limiter.check_and_record("203.0.113.7")[0])

    threads = [threading.Thread(target=hit) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 10
