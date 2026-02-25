import datetime as dt
import os

import pytest

from InsuLearner.insulearner import analyze_settings_lr
from InsuLearner.nightscout.nightscout_api import NightscoutAPI
from InsuLearner.nightscout.nightscout_user_model import NightscoutUser
from InsuLearner.tidepool.tidepool_api import TidepoolAPI
from InsuLearner.tidepool.tidepool_user_model import TidepoolUser
from InsuLearner.util import get_logger

EPSILON = 1e-6
logger = get_logger(__name__)
pytestmark = pytest.mark.integration


def _assert_close(name, lhs, rhs, rel_tol=0.0, abs_tol=1e-6):
    lhs = float(lhs)
    rhs = float(rhs)
    delta = abs(lhs - rhs)
    threshold = max(abs_tol, rel_tol * max(abs(lhs), abs(rhs), 1.0))
    assert delta <= threshold, f"{name} mismatch: lhs={lhs} rhs={rhs} delta={delta} tol={threshold}"


def _parity_debug_summary(tp_user, ns_user, tp_insulin, ns_insulin, tp_carbs, ns_carbs):
    return (
        "\nParity debug:"
        f"\n  Tidepool timeline sizes: basal={len(tp_user.basal_timeline)} bolus={len(tp_user.bolus_timeline)} food={len(tp_user.food_timeline)} cgm={len(tp_user.cgm_timeline)}"
        f"\n  Nightscout timeline sizes: basal={len(ns_user.basal_timeline)} bolus={len(ns_user.bolus_timeline)} food={len(ns_user.food_timeline)} cgm={len(ns_user.cgm_timeline)}"
        f"\n  Tidepool insulin totals: total={tp_insulin['total_insulin']:.4f} bolus={tp_insulin['total_bolus']:.4f} basal={tp_insulin['total_basal']:.4f}"
        f"\n  Nightscout insulin totals: total={ns_insulin['total_insulin']:.4f} bolus={ns_insulin['total_bolus']:.4f} basal={ns_insulin['total_basal']:.4f}"
        f"\n  Tidepool carbs total: {tp_carbs['total_carbs']:.4f}"
        f"\n  Nightscout carbs total: {ns_carbs['total_carbs']:.4f}"
    )


def _window_debug_summary(tp_windows, ns_windows, limit=10):
    lines = ["\nWindow debug (TP vs NS):"]
    for idx, (tp, ns) in enumerate(zip(tp_windows[:limit], ns_windows[:limit])):
        lines.append(
            "  "
            f"{idx}: {tp['start_date'].strftime('%Y-%m-%d')} "
            f"TP(ins={tp['total_insulin']:.2f}, basal={tp['total_basal']:.2f}, carbs={tp['total_carbs']:.2f}, cgm={tp['cgm_mean']:.1f}) "
            f"NS(ins={ns['total_insulin']:.2f}, basal={ns['total_basal']:.2f}, carbs={ns['total_carbs']:.2f}, cgm={ns['cgm_mean']:.1f})"
        )
    if len(tp_windows) != len(ns_windows):
        lines.append(f"  length mismatch: TP={len(tp_windows)} NS={len(ns_windows)}")
    return "\n".join(lines)


def _build_equivalent_fixture_data():
    start_date = dt.datetime(2024, 1, 1, 0, 0, 0)
    end_date = dt.datetime(2024, 1, 3, 0, 0, 0)

    # 2 windows:
    # day1 totals: carbs=40, bolus=4, basal=1
    # day2 totals: carbs=30, bolus=3, basal=1
    tidepool_events = [
        {
            "type": "cbg",
            "id": "tp-cgm-4",
            "uploadId": "u1",
            "deviceId": "dexcom",
            "time": "2024-01-02T13:00:00Z",
            "value": 120,
            "units": "mg/dL",
        },
        {
            "type": "cbg",
            "id": "tp-cgm-3",
            "uploadId": "u1",
            "deviceId": "dexcom",
            "time": "2024-01-02T01:00:00Z",
            "value": 105,
            "units": "mg/dL",
        },
        {
            "type": "bolus",
            "subType": "normal",
            "id": "tp-bolus-2",
            "uploadId": "u1",
            "deviceId": "pump",
            "time": "2024-01-02T12:00:00Z",
            "normal": 3.0,
        },
        {
            "type": "food",
            "id": "tp-food-2",
            "uploadId": "u1",
            "deviceId": "loop",
            "time": "2024-01-02T12:00:00Z",
            "nutrition": {"carbohydrate": {"net": 30, "units": "g"}},
        },
        {
            "type": "basal",
            "id": "tp-basal-2",
            "uploadId": "u1",
            "deviceId": "pump",
            "time": "2024-01-02T03:00:00Z",
            "rate": 1.0,
            "duration": 3600000,
            "deliveryType": "scheduled",
        },
        {
            "type": "cbg",
            "id": "tp-cgm-2",
            "uploadId": "u1",
            "deviceId": "dexcom",
            "time": "2024-01-01T13:00:00Z",
            "value": 115,
            "units": "mg/dL",
        },
        {
            "type": "cbg",
            "id": "tp-cgm-1",
            "uploadId": "u1",
            "deviceId": "dexcom",
            "time": "2024-01-01T01:00:00Z",
            "value": 110,
            "units": "mg/dL",
        },
        {
            "type": "bolus",
            "subType": "normal",
            "id": "tp-bolus-1",
            "uploadId": "u1",
            "deviceId": "pump",
            "time": "2024-01-01T12:00:00Z",
            "normal": 4.0,
        },
        {
            "type": "food",
            "id": "tp-food-1",
            "uploadId": "u1",
            "deviceId": "loop",
            "time": "2024-01-01T12:00:00Z",
            "nutrition": {"carbohydrate": {"net": 40, "units": "g"}},
        },
        {
            "type": "basal",
            "id": "tp-basal-1",
            "uploadId": "u1",
            "deviceId": "pump",
            "time": "2024-01-01T03:00:00Z",
            "rate": 1.0,
            "duration": 3600000,
            "deliveryType": "scheduled",
        },
    ]

    nightscout_entries = [
        {"_id": "ns-cgm-4", "dateString": "2024-01-02T13:00:00Z", "sgv": 120, "units": "mg/dL", "device": "dexcom"},
        {"_id": "ns-cgm-3", "dateString": "2024-01-02T01:00:00Z", "sgv": 105, "units": "mg/dL", "device": "dexcom"},
        {"_id": "ns-cgm-2", "dateString": "2024-01-01T13:00:00Z", "sgv": 115, "units": "mg/dL", "device": "dexcom"},
        {"_id": "ns-cgm-1", "dateString": "2024-01-01T01:00:00Z", "sgv": 110, "units": "mg/dL", "device": "dexcom"},
    ]

    nightscout_treatments = [
        {"_id": "ns-meal-2", "created_at": "2024-01-02T12:00:00Z", "eventType": "Meal Bolus", "carbs": 30, "insulin": 3.0, "enteredBy": "loop"},
        {"_id": "ns-basal-2", "created_at": "2024-01-02T03:00:00Z", "eventType": "Temp Basal", "absolute": 1.0, "duration": 60, "enteredBy": "loop"},
        {"_id": "ns-meal-1", "created_at": "2024-01-01T12:00:00Z", "eventType": "Meal Bolus", "carbs": 40, "insulin": 4.0, "enteredBy": "loop"},
        {"_id": "ns-basal-1", "created_at": "2024-01-01T03:00:00Z", "eventType": "Temp Basal", "absolute": 1.0, "duration": 60, "enteredBy": "loop"},
    ]

    return start_date, end_date, tidepool_events, nightscout_entries, nightscout_treatments


def test_tidepool_nightscout_fixture_parity():
    start_date, end_date, tidepool_events, nightscout_entries, nightscout_treatments = _build_equivalent_fixture_data()

    tidepool_user = TidepoolUser(device_data_json=tidepool_events, notes_json=None)
    nightscout_user = NightscoutUser(entries_json=nightscout_entries, treatments_json=nightscout_treatments)

    tp_insulin = tidepool_user.get_insulin_stats(start_date, end_date)
    ns_insulin = nightscout_user.get_insulin_stats(start_date, end_date)
    _assert_close("total insulin", tp_insulin["total_insulin"], ns_insulin["total_insulin"], abs_tol=EPSILON)
    _assert_close("total bolus", tp_insulin["total_bolus"], ns_insulin["total_bolus"], abs_tol=EPSILON)
    _assert_close("total basal", tp_insulin["total_basal"], ns_insulin["total_basal"], abs_tol=EPSILON)

    tp_carbs = tidepool_user.get_carb_stats(start_date, end_date)
    ns_carbs = nightscout_user.get_carb_stats(start_date, end_date)
    _assert_close("total carbs", tp_carbs["total_carbs"], ns_carbs["total_carbs"], abs_tol=EPSILON)

    tp_windows = tidepool_user.compute_window_stats(
        start_date=start_date,
        end_date=end_date,
        use_circadian=False,
        window_size_hours=24,
        hop_size_hours=24,
        plot_hop_raw=False,
    )
    ns_windows = nightscout_user.compute_window_stats(
        start_date=start_date,
        end_date=end_date,
        use_circadian=False,
        window_size_hours=24,
        hop_size_hours=24,
        plot_hop_raw=False,
    )

    assert len(tp_windows) == len(ns_windows) == 2
    for idx, (tp_row, ns_row) in enumerate(zip(tp_windows, ns_windows)):
        _assert_close(f"window {idx} total_insulin", tp_row["total_insulin"], ns_row["total_insulin"], abs_tol=EPSILON)
        _assert_close(f"window {idx} total_carbs", tp_row["total_carbs"], ns_row["total_carbs"], abs_tol=EPSILON)
        _assert_close(f"window {idx} cgm_mean", tp_row["cgm_mean"], ns_row["cgm_mean"], abs_tol=EPSILON)

    K = 10.0
    tp_cir, tp_basal, tp_isf, _ = analyze_settings_lr(
        tidepool_user,
        data_start_date=start_date,
        data_end_date=end_date,
        K=K,
        do_plots=False,
        use_circadian_hour_estimate=False,
        agg_period_window_size_hours=24,
        agg_period_hop_size_hours=24,
        weight_scheme=None,
    )
    ns_cir, ns_basal, ns_isf, _ = analyze_settings_lr(
        nightscout_user,
        data_start_date=start_date,
        data_end_date=end_date,
        K=K,
        do_plots=False,
        use_circadian_hour_estimate=False,
        agg_period_window_size_hours=24,
        agg_period_hop_size_hours=24,
        weight_scheme=None,
    )

    _assert_close("CIR estimate", tp_cir, ns_cir, abs_tol=EPSILON)
    _assert_close("basal estimate", tp_basal, ns_basal, abs_tol=EPSILON)
    _assert_close("ISF estimate", tp_isf, ns_isf, abs_tol=EPSILON)

    # For this fixture: y = 0.1*x + 1 over 24h windows.
    _assert_close("fixture CIR expected", tp_cir, 10.0, abs_tol=1e-4)
    _assert_close("fixture basal period expected", tp_basal, 1.0, abs_tol=1e-4)
    _assert_close("fixture ISF expected", tp_isf, 100.0, abs_tol=1e-4)


def _parse_env_date(name, default_days_ago):
    value = os.getenv(name)
    if value:
        return dt.datetime.strptime(value, "%Y-%m-%d")
    d = (dt.datetime.now() - dt.timedelta(days=default_days_ago)).date()
    return dt.datetime(d.year, d.month, d.day)


def _have_live_compare_env():
    return bool(
        os.getenv("INSULEARNER_TP_USERNAME")
        and os.getenv("INSULEARNER_TP_PASSWORD")
        and os.getenv("INSULEARNER_NS_URL")
    )


@pytest.mark.skipif(
    not _have_live_compare_env(),
    reason="Set INSULEARNER_TP_USERNAME, INSULEARNER_TP_PASSWORD, and INSULEARNER_NS_URL to run live parity test.",
)
@pytest.mark.live
def test_tidepool_nightscout_live_parity():
    default_num_days = int(os.getenv("INSULEARNER_COMPARE_NUM_DAYS", "7"))
    end_date = _parse_env_date("INSULEARNER_COMPARE_END_DATE", default_days_ago=1)
    start_date = _parse_env_date("INSULEARNER_COMPARE_START_DATE", default_days_ago=default_num_days + 1)

    rel_tol = float(os.getenv("INSULEARNER_COMPARE_REL_TOL", "0.10"))
    abs_tol = float(os.getenv("INSULEARNER_COMPARE_ABS_TOL", "3.0"))

    tp_user = TidepoolUser()
    tp_api = TidepoolAPI(
        username=os.environ["INSULEARNER_TP_USERNAME"],
        password=os.environ["INSULEARNER_TP_PASSWORD"],
    )
    tp_user.load_from_api(
        tp_api_obj=tp_api,
        start_date=start_date,
        end_date=end_date,
        user_id=os.getenv("INSULEARNER_TP_OBSERVED_USER_ID"),
        save_data=False,
    )

    ns_user = NightscoutUser()
    ns_api = NightscoutAPI(
        base_url=os.environ["INSULEARNER_NS_URL"],
        token=os.getenv("INSULEARNER_NS_TOKEN"),
        api_secret=os.getenv("INSULEARNER_NS_API_SECRET"),
    )
    ns_user.load_from_api(
        ns_api_obj=ns_api,
        start_date=start_date,
        end_date=end_date,
        save_data=False,
    )
    tp_user.analyze_duplicates(time_diff_thresh_sec=60 * 60)
    ns_user.analyze_duplicates(time_diff_thresh_sec=60 * 60)

    tp_insulin = tp_user.get_insulin_stats(start_date, end_date)
    ns_insulin = ns_user.get_insulin_stats(start_date, end_date)
    tp_carbs = tp_user.get_carb_stats(start_date, end_date)
    ns_carbs = ns_user.get_carb_stats(start_date, end_date)
    debug_summary = _parity_debug_summary(tp_user, ns_user, tp_insulin, ns_insulin, tp_carbs, ns_carbs)

    try:
        _assert_close("live total insulin", tp_insulin["total_insulin"], ns_insulin["total_insulin"], rel_tol=rel_tol, abs_tol=abs_tol)
        _assert_close("live total bolus", tp_insulin["total_bolus"], ns_insulin["total_bolus"], rel_tol=rel_tol, abs_tol=abs_tol)
        _assert_close("live total basal", tp_insulin["total_basal"], ns_insulin["total_basal"], rel_tol=rel_tol, abs_tol=abs_tol)
        _assert_close("live total carbs", tp_carbs["total_carbs"], ns_carbs["total_carbs"], rel_tol=rel_tol, abs_tol=abs_tol)
    except AssertionError as e:
        raise AssertionError(str(e) + debug_summary)

    # Skip model comparison when one source lacks enough windows with carbs+insulin.
    tp_windows = tp_user.compute_window_stats(start_date, end_date, use_circadian=False, window_size_hours=24, hop_size_hours=24)
    ns_windows = ns_user.compute_window_stats(start_date, end_date, use_circadian=False, window_size_hours=24, hop_size_hours=24)
    assert len(tp_windows) >= 3 and len(ns_windows) >= 3, "Need at least 3 valid daily windows to compare model output."

    K = float(os.getenv("INSULEARNER_COMPARE_K", "12.5"))
    tp_cir, _, _, _ = analyze_settings_lr(
        tp_user,
        data_start_date=start_date,
        data_end_date=end_date,
        K=K,
        do_plots=False,
        use_circadian_hour_estimate=False,
        agg_period_window_size_hours=24,
        agg_period_hop_size_hours=24,
    )
    ns_cir, _, _, _ = analyze_settings_lr(
        ns_user,
        data_start_date=start_date,
        data_end_date=end_date,
        K=K,
        do_plots=False,
        use_circadian_hour_estimate=False,
        agg_period_window_size_hours=24,
        agg_period_hop_size_hours=24,
    )

    # PyCharm's test runner can truncate assertion message bodies.
    # Log comparison details unconditionally so diagnostics are always visible.
    logger.warning("Live CIR compare: TP=%.6f NS=%.6f", tp_cir, ns_cir)
    logger.warning(_window_debug_summary(tp_windows, ns_windows))

    try:
        _assert_close("live CIR estimate", tp_cir, ns_cir, rel_tol=rel_tol, abs_tol=1.0)
    except AssertionError as e:
        raise AssertionError(str(e) + _window_debug_summary(tp_windows, ns_windows))
