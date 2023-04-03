__author__ = "Cameron Summers"

import datetime as dt
import re

import logging
import numpy as np

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os
THIS_DIR = os.path.dirname(__file__)

PHI_DATA_DIR = os.path.join("./PHI/")


def get_logger(name):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

logger = get_logger(__name__)

class TidepoolAPIDateParsingException(Exception):
    pass


DATESTAMP_FORMAT = "%Y-%m-%d"
API_DATA_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def parse_tidepool_api_date_str(date_str):
    """
    Parse date strings in formats common to Tidepool API

    Args:
        date_str (str): date string

    Returns:
        dt.DateTime
    """
    common_timestamp_formats = [
        API_DATA_TIMESTAMP_FORMAT,
        "%Y-%m-%dT%H:%M:%SZ",
        DATESTAMP_FORMAT
    ]

    datetime_obj = None

    # Some devices have 7 zeros instead of six, which datetime can't handle.
    if len(date_str) == len('2021-03-24T14:05:29.0000000Z'):
        date_str = re.sub("\d{7}Z", "000000Z", date_str)
    elif len(date_str) == len('2021-03-24T14:05:29.00000000Z'):
        date_str = re.sub("\d{8}Z", "000000Z", date_str)
    elif len(date_str) == len('2021-03-24T14:05:29.000000000Z'):
        date_str = re.sub("\d{9}Z", "000000Z", date_str)

    try:
        datetime_obj = dt.datetime.fromisoformat(date_str)
    except ValueError:
        for format in common_timestamp_formats:

            try:
                datetime_obj = dt.datetime.strptime(date_str, format)
            except:
                pass

    if datetime_obj is None:
        raise TidepoolAPIDateParsingException("String '{}' could not be parsed.".format(date_str))

    # Notes have
    if datetime_obj.tzinfo is not None:
        offset = datetime_obj.utcoffset()
        datetime_obj = datetime_obj + offset
        datetime_obj = datetime_obj.replace(tzinfo=None)

    return datetime_obj


def get_user_group_data_dir(user_group_name):

    return os.path.join(PHI_DATA_DIR, "tidepool_user_groups", user_group_name)


def create_user_dir(user_id, start_date, end_date, user_group_name=""):
    """
    Create
    Args:
        start_date dt.DateTime: start date of data for user_obj
        end_date dt.DateTime: end date of data for user_obj
        user_id (str): user_obj id for user_obj

    Returns:
        str: dir_path for saving data
    """
    if not os.path.isdir(PHI_DATA_DIR):
        raise Exception("You are not saving to PHI folder. Check your path.")

    user_group_dir = get_user_group_data_dir(user_group_name)

    user_dir_name = "{}_{}_{}".format(user_id, start_date.strftime(DATESTAMP_FORMAT), end_date.strftime(DATESTAMP_FORMAT))
    user_dir_path = os.path.join(user_group_dir, user_dir_name)

    if not os.path.isdir(user_dir_path):
        os.makedirs(user_dir_path)

    return user_dir_path


def get_recursively(search_dict, keyword):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided or field in .

    ref: https://stackoverflow.com/questions/14962485/finding-a-key-recursively-in-a-dictionary
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == keyword:
            fields_found.append(value)

        if isinstance(search_dict[key], str) and keyword in search_dict[key]:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, keyword)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, keyword)
                    for another_result in more_results:
                        fields_found.append(another_result)


    return fields_found


def logger_describe_distribution(description, distribution):

    distribution = np.array(distribution)
    nan_mask = np.isnan(distribution)
    num_before = len(distribution)
    distribution = distribution[~nan_mask]
    num_after = len(distribution)

    logger.info("{} Before NanMask. {} After.".format(num_before, num_after))

    logger.info("{}: mean={:.2f}. min={:.2f}. q1={:.2f}. median={:.2f}. q3={:.2f}. max={:.2f}".format(
        description,
        np.nanmean(distribution),
        np.nanmin(distribution),
        np.nanpercentile(distribution, 25),
        np.nanmedian(distribution),
        np.nanpercentile(distribution, 75),
        np.nanmax(distribution),
    ))


def get_years_since_date(reference_date):

    years_since = (dt.datetime.today() - reference_date).total_seconds() / 60 / 60 / 24 / 365
    return years_since


def create_synthetic_dataset(num_days):

    data = []
    cir = 10
    target = 100
    br_total = 10
    np.random.seed(1234)
    for i in range(num_days):
        carb_trend = 1.0  #+ 0.2*np.cos(2*np.pi * i / num_days)
        total_carbs_true = np.random.normal(150, 30) * carb_trend

        # carb_estimation_std = total_carbs_true * 0.01
        carb_estimation_std = 20
        total_carbs = np.random.normal(total_carbs_true, carb_estimation_std)
        carb_diff = (total_carbs_true - total_carbs)

        insulin_sensitivity = 1.0 #+ 0.2*np.sin(2*np.pi * i / num_days)
        total_insulin = br_total + total_carbs_true / cir * insulin_sensitivity

        tir = 1.0 - abs(carb_diff) / 100
        day = {
            "date": dt.datetime(2021, 1, 1, 0, 0, 0) + dt.timedelta(days=i),
            "total_carbs_true": total_carbs_true,
            "total_carbs": total_carbs,
            "total_insulin": total_insulin,
            "cgm_geo_mean": target * insulin_sensitivity,
            "cgm_mean": target * insulin_sensitivity,
            "cgm_percent_below_54": 0,
            "cgm_percent_tir": tir,
            "carb_diff": carb_diff
        }
        data.append(day)

    df = pd.DataFrame(data)

    window_size_days = 60
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size_days)
    # df_agg = df.rolling(window=indexer, min_periods=1).mean()
    # df_agg = df_agg.iloc[:-window_size_days]
    # df_agg["date"] = df["date"].iloc[:-window_size_days]
    # df = df_agg

    g = sns.pairplot(df[["total_carbs_true", "total_carbs", "total_insulin", "cgm_percent_tir"]])

    df.to_csv("synthetic_aggregated_dataset.csv")

    plt.title('Fake Dataset')
    plt.show()

    return df


if __name__ == "__main__":
    create_synthetic_dataset(60)