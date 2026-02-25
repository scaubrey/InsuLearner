"""
Tests
"""

from pathlib import Path

import pandas as pd
import pytest

from InsuLearner.carbohydrate_sensitivity_factor import estimate_csf
from InsuLearner.insulearner import estimate_therapy_settings_from_window_stats_lr

EPSILON = 1e-6
pytestmark = pytest.mark.unit


def test_nonregression_linear_model_estimation():
    """
    Run the linear model fitting on a static synthetic dataset for regression
    testing.

    Last updated April 3, 2023.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_aggregated_dataset.csv"
    window_df = pd.read_csv(data_path)

    K = 5
    agg_period_window_size_hours = 24
    do_plots = False

    settings = estimate_therapy_settings_from_window_stats_lr(window_df,
                                                              K,
                                                              period_window_size_hours=agg_period_window_size_hours,
                                                              target_bg=110,
                                                              x="total_carbs",
                                                              y="total_insulin",
                                                              do_plots=do_plots,
                                                              weight_scheme=None)

    cir_estimate, isf_estimate, basal_insulin_estimate, lr_model, lr_score, K_out = settings

    assert (cir_estimate - 21.439068470) < EPSILON
    assert (isf_estimate - 107.19534235) < EPSILON
    assert (basal_insulin_estimate - 18.666299498) < EPSILON
    assert (lr_score - 0.36454452) < EPSILON
    assert K_out == K


def test_carbohydrate_sensitivity_factor():
    """
    Check values for estimating CSF.
    """

    height_inches = 72
    weight_lbs = 200

    csf_male = estimate_csf(height_inches=height_inches, weight_lbs=weight_lbs, gender="male")

    csf_female = estimate_csf(height_inches=height_inches, weight_lbs=weight_lbs, gender="female")

    assert (csf_male - 3.98754307) < EPSILON
    assert (csf_female - 4.289579866) < EPSILON


def test_tidepool_data_download():
    """
    TODO
    """
