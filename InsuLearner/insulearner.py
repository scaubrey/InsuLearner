__author__ = "Cameron Summers"
"""
InsuLearner: Estimating Insulin Pump Settings via Machine Learning

Code underlying this blog post:
https://www.cameronsummers.com/how_I_calculate_my_sons_insulin_pump_settings_with_machine_learning
"""

import sys
import os
import argparse
import datetime as dt

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from sklearn.linear_model import LinearRegression

from InsuLearner.tidepool.tidepool_user_model import TidepoolUser
from InsuLearner.tidepool.tidepool_api import TidepoolAPI
from InsuLearner.util import get_logger

from InsuLearner.carbohydrate_sensitivity_factor import estimate_csf

logger = get_logger(__name__)


def compute_aace_pump_settings(weight_kg, prepump_tdd):
    """
    Get pump settings using the American Association of Clinical Endocrinologists/American College of
    Endocrinology.

    Other references:
    https://diabetesed.net/wp-content/uploads/2019/09/Insulin-Pump-Calculations-Sept-2019-slides.pdf

    Review of insulin dosing formulas
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4960276/

    Args:
        weight_kg (float): weight in kg
        prepump_tdd (float): units of insulin per day, (CS: presuming this is a life-time average)

    Returns:
        (float, float, float): basal rate, carb insulin ratio, insulin sensitivity factor
    """
    tdd_method1 = weight_kg * 0.5
    tdd_method2 = prepump_tdd * 0.75
    starting_pump_tdd = (tdd_method1 + tdd_method2) / 2

    basal_rate = starting_pump_tdd * 0.5 / 24
    cir = 450.0 / starting_pump_tdd
    isf = 1700.0 / starting_pump_tdd

    return basal_rate, cir, isf


def pd_1d_series_to_X(series):
    """helper function"""
    return np.array(series.values.tolist())[:, np.newaxis]


def estimate_therapy_settings_from_window_stats_lr(aggregated_df,
                                                   K,
                                                   period_window_size_hours,
                                                   target_bg=110,
                                                   x="total_carbs",
                                                   y="total_insulin",
                                                   do_plots=True,
                                                   trained_model=None,
                                                   weight_scheme=None):
    """
    Fit the linear model and estimate settings
    """
    X_carbs = pd_1d_series_to_X(aggregated_df[x])
    y_insulin = aggregated_df[y]

    if weight_scheme is None:
        sample_weights = None
    else:
        if weight_scheme == "CGM Weighted":
            sample_weights = 1.0 / np.array(np.maximum(1.0, abs(target_bg - aggregated_df["cgm_geo_mean"])) / aggregated_df["cgm_percent_tir"]) * aggregated_df["cgm_percent_available"]
        elif weight_scheme == "Carb Uncertainty":
            sample_weights = scipy.stats.norm.pdf(aggregated_df["total_carbs"], aggregated_df["total_carbs"].mean(), aggregated_df["total_carbs"].std())
        else:
            raise Exception("Unknown weight scheme {}.".format(weight_scheme))

        nan_mask = np.isnan(sample_weights)
        X_carbs = X_carbs[~nan_mask]
        y_insulin = y_insulin[~nan_mask]
        sample_weights = sample_weights[~nan_mask]

        if do_plots:
            plt.figure()
            plt.title("Sample Weight Distribution")
            plt.hist(sample_weights)

    if trained_model is None:
        lm_carb_to_insulin = LinearRegression()
        lm_carb_to_insulin.fit(X_carbs, y_insulin, sample_weight=sample_weights)
    else:
        lm_carb_to_insulin = trained_model

    basal_insulin_estimate = lm_carb_to_insulin.intercept_

    r2_fit = lm_carb_to_insulin.score(X_carbs, y_insulin)
    logger.info(f"Linear Model: Fit R^2 {np.round(r2_fit, 2)}. Intercept {np.round(lm_carb_to_insulin.intercept_, 2)}. Slope g/U {np.round(1.0 / lm_carb_to_insulin.coef_, 2)}")

    cir_estimate_slope = 1 / lm_carb_to_insulin.coef_[0]
    isf_estimate_slope = cir_estimate_slope * K

    basal_rate_estimate_hourly = basal_insulin_estimate / period_window_size_hours

    logger.info("Total Period Basal={:.2f}U. (Mean %Daily Total: {:.2f}%)".format(basal_insulin_estimate, np.nanmean(basal_insulin_estimate / aggregated_df[y]) * 100))

    logger.info("\n\n\tSettings Estimates:\n")

    logger.info(f"\tEstimated CIR={round(cir_estimate_slope, 2)} g/U.")
    logger.info(f"\tEstimated Hourly Basal={round(basal_rate_estimate_hourly, 3)} U/hr")
    logger.info(f"\tCSF={round(K, 2)} mg/dL / g")
    logger.info(f"\tEstimated ISF={round(isf_estimate_slope, 2)} mg/dL/ U")

    settings = (cir_estimate_slope, isf_estimate_slope, basal_insulin_estimate, lm_carb_to_insulin, r2_fit, K)
    if do_plots:
        plot_aggregated_scatter(aggregated_df,
                                period_window_size_hours,
                                lr_model=lm_carb_to_insulin,
                                settings=settings,
                                plot_aace=True,
                                weight_scheme=weight_scheme)

    return settings


def plot_aggregated_scatter(aggregated_df, period_window_size_hours, lr_model=None, settings=None, plot_aace=True, weight_scheme=None):
    """
    Plot the linear model on the data.
    """
    period_window_size_days = period_window_size_hours / 24

    fig, ax = plt.subplots(figsize=(8, 8))
    win_start_dates_str = aggregated_df["start_date"].dt.strftime("%Y-%m-%d")
    win_end_dates_str = aggregated_df["end_date"].dt.strftime("%Y-%m-%d")
    plt.title(f"Insulin Prediction Modeling, Aggr Period={period_window_size_days} Days, {win_start_dates_str.values[0]} to {win_end_dates_str.values[-1]}, {len(aggregated_df)} data points")

    hue_col = "cgm_geo_mean"
    # hue_col = None

    vars_to_plot = ["total_carbs", "total_insulin", "cgm_geo_mean"]
    scatter_df = aggregated_df[vars_to_plot]

    sns.scatterplot(data=scatter_df, x="total_carbs", y="total_insulin", hue=hue_col, ax=ax)
    plt.figure()

    ax.set_ylim(0, aggregated_df["total_insulin"].max() * 1.1)
    ax.set_xlim(0, aggregated_df["total_carbs"].max() * 1.1)

    if settings:

        cir_estimate_slope, isf_estimate_slope, basal_insulin_estimate, lm_carb_to_insulin, r2_fit, K = settings
        basal_glucose_lr = -basal_insulin_estimate / lr_model.coef_[0]

        x1, y1 = basal_glucose_lr, 0
        x2, y2 = aggregated_df["total_carbs"].max(), lr_model.predict([[aggregated_df["total_carbs"].max()]])
        ax.plot([x1, x2], [y1, y2[0]], label="Insulin Prediction LR Model (Weights: {})".format(weight_scheme))

        ax.set_xlabel("Total Exogenous Glucose in Period T")
        ax.set_ylabel("Total Insulin in Period T")

        # Equations and Settings
        ax.text(0.6, 0.25, "y={:.4f}*x + {:.2f}, (R^2={:.2f})".format(lr_model.coef_[0], lr_model.intercept_, r2_fit), ha="left", va="top", transform=ax.transAxes)
        ax.text(0.6, 0.2, "CIR={:.2f} g/U \nBasal Rate={:.2f}U/hr \nISF={:.2f} mg/dL/U (K={:.2f})".format(cir_estimate_slope,
                                                                                                   basal_insulin_estimate / period_window_size_hours,
                                                                                                   isf_estimate_slope,
                                                                                                   K),
                                                                                                   ha="left", va="top", transform=ax.transAxes)

        # Stars
        ax.plot(0, basal_insulin_estimate, marker="*", markersize=12, color="green", label="Basal Insulin LR Estimate")
        ax.plot(basal_glucose_lr, 0, marker="*", markersize=12, color="orange", label="Endogenous Glucose LR Estimate")

        mean_insulin = aggregated_df["total_insulin"].mean()
        mean_carbs = aggregated_df["total_carbs"].mean()
        ax.plot(mean_carbs, mean_insulin, marker="*", markersize=12, color="red", label="Mean Insulin/Carbs")

        # Shades
        ax.fill_between([0, x2], [basal_insulin_estimate, basal_insulin_estimate], color="blue", alpha=0.2, label="Endogenous")
        ax.fill_between([0, x2], [basal_insulin_estimate, basal_insulin_estimate], [basal_insulin_estimate, y2[0]],
                        color="orange", alpha=0.2, label="Exogenous")

        # AACE line if using 1-day windows
        if plot_aace and period_window_size_hours == 1:
            tdd_mean = aggregated_df["total_insulin"].mean()
            aace_basal_insulin_estimate = tdd_mean / 2
            cir_aace = 450 / tdd_mean
            aace_basal_glucose_estimate = -aace_basal_insulin_estimate / (1/cir_aace)
            x2 = aggregated_df["total_carbs"].max()

            x1, y1 = (aace_basal_glucose_estimate, 0)
            y2 = 1.0 / (cir_aace) * x2 + aace_basal_insulin_estimate
            star_description = "AACE Basal Estimate (mean(TDD) / 2)"
            line_description = "AACE (*mean(TDD) only)"

            ax.plot([x1, x2], [y1, y2], label=line_description, color="gray", linestyle="--")
            ax.plot(0, aace_basal_insulin_estimate, marker="*", markersize=12, color="gray", label=star_description)

        ax.legend()

    plt.show()


def analyze_settings_lr(user, data_start_date, data_end_date,
                        K=12.5,
                        do_plots=True,
                        use_circadian_hour_estimate=True,
                        agg_period_window_size_hours=24,
                        agg_period_hop_size_hours=24,
                        weight_scheme=None):
    """
    Aggregate the data and fit a linear model
    """
    window_stats = user.compute_window_stats(data_start_date, data_end_date,
                                             use_circadian=use_circadian_hour_estimate,
                                             window_size_hours=agg_period_window_size_hours,
                                             hop_size_hours=agg_period_hop_size_hours,
                                             plot_hop_raw=False)
    window_df = pd.DataFrame(window_stats)

    logger.debug(f'Mean of CGM Mean, {np.round(window_df["cgm_mean"].mean(), 2)}')
    logger.debug(f'Mean of CGM Geo Mean, {np.round(window_df["cgm_geo_mean"].mean(), 2)}')
    logger.debug(f'Total Period Insulin Mean: {np.round(window_df["total_insulin"].mean(), 2)}')
    logger.debug(f"{len(window_df)} Data Rows")

    settings = estimate_therapy_settings_from_window_stats_lr(window_df,
                                                              K,
                                                              period_window_size_hours=agg_period_window_size_hours,
                                                              target_bg=110,
                                                              x="total_carbs",
                                                              y="total_insulin",
                                                              do_plots=do_plots,
                                                              weight_scheme=weight_scheme)

    cir_estimate, isf_estimate, basal_insulin_estimate, lr_model, lr_score, K = settings

    return cir_estimate, basal_insulin_estimate, isf_estimate, lr_score


def load_user_data(username, password, data_start_date, data_end_date, estimation_window_size_days):
    """
    Load the Tidepool user_obj data for the given user_obj and parameters.

    Args:
        username: tidepool username
        password: tidepool password
        data_start_date: start date of analysis
        data_end_date: end date of analysis
        estimation_window_size_days: size of estimation window in days

    Returns:
        TidepoolUser object
    """

    tp_api_obj = TidepoolAPI(username, password)
    user = TidepoolUser()
    user.load_from_api(tp_api_obj, data_start_date, data_end_date,
                       user_id=None,  # For a user_obj that is sharing their Tidepool account with this one
                       save_data=False)

    total_basal_days = user.get_num_days_span(data_type="basal")
    total_bolus_days = user.get_num_days_span(data_type="bolus")
    total_cgm_days = user.get_num_days_span(data_type="cgm")
    total_food_days = user.get_num_days_span(data_type="food")

    if [total_basal_days, total_bolus_days, total_cgm_days, total_food_days] != [estimation_window_size_days]:
        logger.warning(f"*** Warning *** : Num data days span not the size of estimation window size of {estimation_window_size_days} days")
        logger.warning(f"Basal Days Span {total_basal_days}. Bolus Days Span {total_bolus_days}. CGM Days Span {total_cgm_days}. Food Days Span {total_food_days}")

    user.analyze_duplicates(time_diff_thresh_sec=60 * 60)

    return user


def main():

    parser = argparse.ArgumentParser("InsuLearner: Estimate Insulin Pump Settings with Linear Regression")

    parser.add_argument("tp_username", type=str, help="Email username for Tidepool Account")
    parser.add_argument("tp_password", type=str, help="Password for Tidepool Account")
    parser.add_argument("-ht", "--height_inches", type=float, help="Your height in inches")
    parser.add_argument("-wt", "--weight_lbs", type=float, help="Your weight in pounds")
    parser.add_argument("-g", "--gender", choices=["male", "female"])
    parser.add_argument("--num_days", type=int, help="Number of days in the past to analyze data", default=60)
    parser.add_argument("--CSF", type=float, help="If entered, will use this CSF instead of estimating it from height and weight.")
    parser.add_argument("-eb", "--estimate_agg_boundaries", "-eb", action="store_true", default=True,
                        help="Use an autocorrelation-like algorithm to estimate aggregation boundaries to denoise the fit.")

    parser.add_argument("-aw", "--agg_period_window_size_hours", type=int,
                        help="The size in hours of each period to aggregate for fitting the model.", default=24)
    parser.add_argument("-ah", "--agg_period_hop_size_hours", "-ah", type=int,
                        help="The size in hours to hop each period for aggregation.", default=24)

    args = parser.parse_args()

    tp_username = args.tp_username
    tp_password = args.tp_password
    estimation_window_size_days = args.num_days
    height_inches = args.height_inches
    weight_lbs = args.weight_lbs
    gender = args.gender
    CSF = args.CSF
    estimate_agg_boundaries = args.estimate_agg_boundaries
    agg_period_window_size_hours = args.agg_period_window_size_hours
    agg_period_hop_size_hours = args.agg_period_hop_size_hours

    logger.debug(f"Args:")
    logger.debug(f"estimation_window_size_days: {estimation_window_size_days}")
    logger.debug(f"height_inches: {height_inches}")
    logger.debug(f"weight_lbs: {weight_lbs}")
    logger.debug(f"gender: {gender}")
    logger.debug(f"CSF: {CSF}")
    logger.debug(f"estimate_agg_boundaries: {estimate_agg_boundaries}")
    logger.debug(f"agg_period_window_size_hours: {agg_period_window_size_hours}")
    logger.debug(f"agg_period_hop_size_hours: {agg_period_hop_size_hours}")

    K = CSF
    if CSF is None:
        # Estimate CSF from blood volume based on height and weight
        K = estimate_csf(height_inches, weight_lbs, gender, metabolism_efficiency_percentage=0.23)
        logger.info(f"CSF estimated to be {np.round(K, 2)} for height_inches {height_inches}, weight_lbs {weight_lbs}, and gender {gender}")
    else:
        logger.info(f"Provided CSF={K}")

    # Get date info
    today = dt.datetime.now()
    data_start_date = today - dt.timedelta(days=estimation_window_size_days + 1)
    data_end_date = today - dt.timedelta(days=1)

    # Uncomment if specific dates desired
    # data_start_date = dt.datetime(year=2022, month=12, day=1)
    # data_end_date = dt.datetime(year=2023, month=1, day=26)

    logger.info(f"Running for dates {data_start_date} to {data_end_date}")

    # Load user_obj data into an object
    user_obj = load_user_data(tp_username, tp_password, data_start_date, data_end_date, estimation_window_size_days)

    # Run settings analysis
    analyze_settings_lr(user_obj,
                        data_start_date=data_start_date,
                        data_end_date=data_end_date,
                        K=K,
                        do_plots=True,
                        use_circadian_hour_estimate=estimate_agg_boundaries,
                        agg_period_window_size_hours=agg_period_window_size_hours,
                        agg_period_hop_size_hours=agg_period_hop_size_hours,
                        )


if __name__ == "__main__":
    main()
