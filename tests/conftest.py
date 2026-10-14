"""Offline test setup.

- Dummy API keys are set BEFORE `app` is imported, so real keys from .env are
  never used (python-dotenv does not override variables that already exist).
- All network access is blocked, so any accidental real API call fails the test
  instead of spending OpenAI or Tavily credits.
"""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

os.environ["OPENAI_API_KEY"] = "sk-test-dummy-not-real"
os.environ["TAVILY_API_KEY"] = "tvly-test-dummy-not-real"
os.environ["MODEL_NAME"] = "gpt-4o-mini"
# Follow-up searches are switched on per test (see test_depth.py) so call counts stay predictable.
os.environ["FOLLOW_UP_ROUNDS"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkBlocked("Network access is disabled in offline tests.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture
def synthetic():
    data = json.loads((FIXTURES / "synthetic_search_results.json").read_text(encoding="utf-8"))
    assert data["_synthetic"] is True
    return data


@pytest.fixture
def app_module():
    import app  # noqa: WPS433 (import after env is prepared)
    return app


class FakeStructuredModel:
    """Stands in for model.with_structured_output(...). Counts calls."""

    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = 0
        self.last_prompt = None

    def invoke(self, messages):
        self.calls += 1
        self.last_prompt = messages[0].content
        if self.error:
            raise self.error
        return self.output


@pytest.fixture
def fake_model():
    return FakeStructuredModel
