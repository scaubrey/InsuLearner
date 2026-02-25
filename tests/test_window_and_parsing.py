import datetime as dt

import pytest

from InsuLearner.tidepool.tidepool_user_model import TidepoolUser
from InsuLearner.util import parse_tidepool_api_date_str

pytestmark = pytest.mark.unit


def test_parse_tidepool_api_date_str_handles_offset_timestamp():
    parsed = parse_tidepool_api_date_str("2026-02-01T10:30:00-05:00")
    assert isinstance(parsed, dt.datetime)
    assert parsed.tzinfo is None


def test_window_stats_boundary_two_days():
    # Descending time order to match parser expectations.
    events = [
        {"type": "cbg", "id": "c4", "uploadId": "u", "deviceId": "d", "time": "2026-02-02T20:00:00Z", "value": 130, "units": "mg/dL"},
        {"type": "cbg", "id": "c3", "uploadId": "u", "deviceId": "d", "time": "2026-02-02T08:00:00Z", "value": 120, "units": "mg/dL"},
        {"type": "food", "id": "f2", "uploadId": "u", "deviceId": "d", "time": "2026-02-02T12:00:00Z", "nutrition": {"carbohydrate": {"net": 20, "units": "g"}}},
        {"type": "bolus", "subType": "normal", "id": "b2", "uploadId": "u", "deviceId": "d", "time": "2026-02-02T12:00:00Z", "normal": 2.0},
        {"type": "basal", "id": "ba2", "uploadId": "u", "deviceId": "d", "time": "2026-02-02T00:00:00Z", "rate": 1.0, "duration": 3600000, "deliveryType": "scheduled"},
        {"type": "cbg", "id": "c2", "uploadId": "u", "deviceId": "d", "time": "2026-02-01T20:00:00Z", "value": 115, "units": "mg/dL"},
        {"type": "cbg", "id": "c1", "uploadId": "u", "deviceId": "d", "time": "2026-02-01T08:00:00Z", "value": 110, "units": "mg/dL"},
        {"type": "food", "id": "f1", "uploadId": "u", "deviceId": "d", "time": "2026-02-01T12:00:00Z", "nutrition": {"carbohydrate": {"net": 10, "units": "g"}}},
        {"type": "bolus", "subType": "normal", "id": "b1", "uploadId": "u", "deviceId": "d", "time": "2026-02-01T12:00:00Z", "normal": 1.0},
        {"type": "basal", "id": "ba1", "uploadId": "u", "deviceId": "d", "time": "2026-02-01T00:00:00Z", "rate": 1.0, "duration": 3600000, "deliveryType": "scheduled"},
    ]

    user = TidepoolUser(device_data_json=events, notes_json=None)
    start = dt.datetime(2026, 2, 1, 0, 0, 0)
    end = dt.datetime(2026, 2, 3, 0, 0, 0)

    windows = user.compute_window_stats(
        start_date=start,
        end_date=end,
        use_circadian=False,
        window_size_hours=24,
        hop_size_hours=24,
        plot_hop_raw=False,
    )

    assert len(windows) == 2
    assert windows[0]["total_carbs"] == 10
    assert windows[1]["total_carbs"] == 20
    assert windows[0]["total_bolus"] == 1.0
    assert windows[1]["total_bolus"] == 2.0
