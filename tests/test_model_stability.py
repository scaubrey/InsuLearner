import numpy as np
import pandas as pd
import pytest

from InsuLearner.insulearner import estimate_therapy_settings_from_window_stats_lr

pytestmark = pytest.mark.unit


def _make_df():
    return pd.DataFrame(
        {
            "total_carbs": [80, 120, 150, 200, 90, 160, 300],  # includes one outlier-ish day
            "total_insulin": [14, 18, 22, 27, 15, 24, 39],
            "cgm_geo_mean": [125, 130, 140, 145, 128, 150, 160],
            "cgm_percent_tir": [0.7, 0.75, 0.68, 0.65, 0.72, 0.61, 0.55],
            "cgm_percent_available": [0.95, 0.96, 0.94, 0.93, 0.97, 0.92, 0.90],
            "end_date": pd.to_datetime(
                ["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06", "2026-02-07"]
            ),
        }
    )


@pytest.mark.parametrize("weight_scheme", [None, "CGM Weighted", "Carb Uncertainty", "Recency"])
def test_model_estimation_returns_finite(weight_scheme):
    df = _make_df()
    cir, isf, basal, _, r2, kout = estimate_therapy_settings_from_window_stats_lr(
        aggregated_df=df,
        K=12.5,
        period_window_size_hours=24,
        target_bg=110,
        x="total_carbs",
        y="total_insulin",
        do_plots=False,
        weight_scheme=weight_scheme,
    )

    assert np.isfinite(cir)
    assert np.isfinite(isf)
    assert np.isfinite(basal)
    assert np.isfinite(r2)
    assert kout == 12.5


def test_model_estimation_unknown_weight_scheme_raises():
    df = _make_df()
    with pytest.raises(Exception, match="Unknown weight scheme"):
        estimate_therapy_settings_from_window_stats_lr(
            aggregated_df=df,
            K=12.5,
            period_window_size_hours=24,
            do_plots=False,
            weight_scheme="INVALID",
        )
