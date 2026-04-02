import datetime as dt

import pytest
import requests

from InsuLearner.nightscout.nightscout_api import NightscoutAPI
from InsuLearner.tidepool.tidepool_api import TidepoolAPI

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, url="https://example.test/api"):
        self.status_code = status_code
        self._payload = [] if payload is None else payload
        self.url = url
        self.text = "fake"

    def json(self):
        return self._payload


def test_nightscout_entries_uses_full_day_range_and_token(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "params": dict(params),
                "timeout": timeout,
            }
        )
        return _FakeResponse(status_code=200, payload=[])

    monkeypatch.setattr("requests.get", fake_get)

    api = NightscoutAPI(base_url="https://ns.example", token="abc123")
    start_date = dt.datetime(2026, 2, 1, 12, 34, 56)
    end_date = dt.datetime(2026, 2, 3, 7, 8, 9)
    normalized_start, normalized_end = api._normalize_day_range(start_date, end_date)

    api.get_entries(start_date, end_date)

    assert calls[0]["url"].endswith("/api/v1/entries.json")
    assert calls[0]["params"]["find[date][$gte]"] == api._date_to_nightscout_millis(normalized_start)
    assert calls[0]["params"]["find[date][$lte]"] == api._date_to_nightscout_millis(normalized_end)
    assert calls[0]["params"]["token"] == "abc123"


def test_nightscout_entries_falls_back_to_client_side_filtering(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append({"url": url, "params": dict(params)})
        if "find[date][$gte]" in params:
            return _FakeResponse(status_code=200, payload=[], url=url)

        payload = [
            {"_id": "in-range", "date": 1775161822455, "sgv": 207},
            {"_id": "out-of-range", "date": 1774800000000, "sgv": 150},
        ]
        return _FakeResponse(status_code=200, payload=payload, url=url)

    monkeypatch.setattr("requests.get", fake_get)

    api = NightscoutAPI(base_url="https://ns.example", token="abc123")
    result = api.get_entries(dt.datetime(2026, 4, 2), dt.datetime(2026, 4, 2))

    assert len(calls) == 2
    assert "find[date][$gte]" in calls[0]["params"]
    assert calls[1]["params"] == {"count": 10000, "token": "abc123"}
    assert [row["_id"] for row in result] == ["in-range"]


def test_nightscout_auth_fallback_hash_on_401(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(status_code=401, payload={"error": "unauthorized"}, url=url)
        return _FakeResponse(status_code=200, payload=[], url=url)

    monkeypatch.setattr("requests.get", fake_get)

    api = NightscoutAPI(base_url="https://ns.example", api_secret="secret123")
    api.get_treatments(dt.datetime(2026, 2, 1), dt.datetime(2026, 2, 1))

    assert calls["n"] == 2


def test_nightscout_raises_http_error_with_context(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeResponse(status_code=500, payload={"error": "boom"}, url=url)

    monkeypatch.setattr("requests.get", fake_get)

    api = NightscoutAPI(base_url="https://ns.example")
    with pytest.raises(requests.HTTPError) as exc:
        api.get_entries(dt.datetime(2026, 2, 1), dt.datetime(2026, 2, 1))

    msg = str(exc.value)
    assert "HTTP 500" in msg
    assert "/api/v1/entries.json" in msg


def test_tidepool_date_filter_string_spans_full_day():
    api = TidepoolAPI(username="u", password="p")
    start, end = api.get_date_filter_string(dt.datetime(2026, 2, 1, 15, 45), dt.datetime(2026, 2, 4, 1, 2))
    assert start == "2026-02-01T00:00:00.000Z"
    assert end == "2026-02-04T23:59:59.999Z"


def test_tidepool_login_guard_enforced():
    api = TidepoolAPI(username="u", password="p")
    with pytest.raises(Exception, match="Not logged in"):
        api.get_user_event_data(dt.datetime(2026, 2, 1), dt.datetime(2026, 2, 2))


def test_tidepool_pending_invitations_returns_empty_on_http_error(monkeypatch):
    class _PendingInvitesResponse:
        def raise_for_status(self):
            raise requests.HTTPError("boom")

        def json(self):
            return [{"id": 1}]

    def fake_get(url, headers=None):
        return _PendingInvitesResponse()

    monkeypatch.setattr("requests.get", fake_get)

    api = TidepoolAPI(username="u", password="p")
    api._login_user_id = "abc"
    api._login_headers = {"x-tidepool-session-token": "t"}

    assert api.get_pending_observer_invitations() == []
