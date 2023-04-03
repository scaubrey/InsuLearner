
import os
from collections import OrderedDict, defaultdict
import datetime as dt
import json
from operator import itemgetter

import numpy as np
from scipy.stats import gmean, gstd

import logging

from InsuLearner.util import (
    API_DATA_TIMESTAMP_FORMAT, parse_tidepool_api_date_str, create_user_dir, DATESTAMP_FORMAT, get_recursively, get_logger
)
from InsuLearner.visualize_user_data import (
    plot_raw_data, plot_daily_stats
)

logger = get_logger(__name__)


# ===============  Tidepool API Objects ================#

class TidepoolMeasurement(object):
    """
    Object to represent a measurement of something in the Tidepool system.
    """

    def __init__(self, value, units, mid, upload_id, source):
        if not isinstance(value, float):
            try:
                value = float(value)
            except Exception as e:
                logger.debug("Exception: TidepoolMeasurement")
                raise e

        self.value = value
        self.units = units
        self.id = mid
        self.upload_id = upload_id
        self.source = source

    def get_value(self):
        return self.value

    def get_units(self):
        return self.units

    def is_equal(self, other, value_diff_threshold):
        return self.__class__ == other.__class__ and \
               abs(self.value - other.value) < value_diff_threshold and \
               self.units == other.units


class TidepoolGlucoseMeasurement(TidepoolMeasurement):

    def __init__(self, value, units, mid, upload_id, source):
        super().__init__(value, units, mid, upload_id, source)

        if units == "mmol/L":
            self.value *= 18.0182
            self.units = "mg/dL"


class TidepoolManualGlucoseMeasurement(TidepoolGlucoseMeasurement):

    def __init__(self, value, units, mid, upload_id, source):

        super().__init__(value, units, mid, upload_id, source)


class TidepoolCGMGlucoseMeasurement(TidepoolGlucoseMeasurement):

    def __init__(self, value, units, mid, upload_id, source):

        super().__init__(value, units, mid, upload_id, source)


class TidepoolFood(TidepoolMeasurement):

    def __init__(self, value, units, mid, upload_id, source, duration_sec=None):
        super().__init__(value, units, mid, upload_id, source)
        self.duration_sec = duration_sec


class TidepoolBasal(TidepoolMeasurement):

    def __init__(self, value, units, duration_hours, delivery_type, mid, upload_id, source):
        super().__init__(value, units, mid, upload_id, source)

        self.duration_hours = duration_hours
        self.delivery_type = delivery_type

    def get_duration_hours(self):
        return self.duration_hours

    def is_equal(self, other, value_diff_threshold):

        super().is_equal(other, value_diff_threshold) and \
        self.duration_hours == other.duration_hours and \
        self.delivery_type == other.delivery_type


class TidepoolBolus(TidepoolMeasurement):

    def __init__(self, value, units, mid, upload_id, source):
        super().__init__(value, units, mid, upload_id, source)


class TidepoolSquareBolus(TidepoolBolus):

    def __init__(self, value, units, duration, mid, upload_id, source):
        super().__init__(value, units, mid, upload_id, source)

        self.duration = duration


class TidepoolTimeChange():

    def __init__(self, timechange_event, mid, upload_id, source):

        self.timechange_event = timechange_event


class TidepoolReservoirChange():

    def __init__(self, event_data, mid, upload_id, source):

        self.data = event_data


class TidepoolNote():
    """
    A Tidepool Note
    """

    def __init__(self):

        self.note_datetime = None
        self.created_datetime = None
        self.message = None

        self.sensor_change_algo_map = {
            "v1": self._is_sensor_change_v1
        }

    def from_raw(self, note):
        """
        Constructor from raw json.

        Args:
            note (dict): raw json from Tidepool API

        Returns:
            self
        """

        self.note_datetime = parse_tidepool_api_date_str(note["timestamp"])
        self.created_datetime = parse_tidepool_api_date_str(note["createdtime"])
        self.message = note["messagetext"]

        return self

    def is_type(self, note_type, algo_version="v1"):

        return self.sensor_change_algo_map[algo_version](note_type)

    def _is_sensor_change_v1(self, note_type):

        return note_type in self.message

    def has_tag(self, tag_string, case_match=False):

        if case_match:
            return tag_string in self.message
        else:
            return tag_string.lower() in self.message.lower()

    def get_tags(self):
        raise NotImplementedError


class TidepoolWizard():

    def __init__(self, wizard_event, mid, upload_id, source):

        self.raw_data = wizard_event
        self.id = mid
        self.upload_id = upload_id


# ========== Tidepool User ================

class TidepoolUser(object):
    """
    Class representing a Tidepool user_obj from their data.
    """
    def __init__(self, device_data_json=None, notes_json=None, api_version="v1"):
        """
        Args:
            device_data_json (list): list of event data of any kind in Tidepool API
            api_version (str): parser version to user_obj
        """
        self._notes_filename = "notes.json"
        self._event_data_filename = "event_data.json"
        self._creation_meta_filename = "creation_metadata.json"

        self.unparsed_events = defaultdict(int)

        self.device_data_json = device_data_json
        self.notes_json = notes_json
        self.api_version = api_version

        self.data_parser_map = {
            "v1": self.parse_data_json_v1
        }

        self.notes_parser_map = {
            "v1": self.parse_notes_json_v1
        }

        # Manage device data in separate timelines
        self.basal_timeline = OrderedDict()
        self.bolus_timeline = OrderedDict()
        self.food_timeline = OrderedDict()
        self.cgm_timeline = OrderedDict()
        self.wizard_timeline = OrderedDict()
        self.time_change_timeline = OrderedDict()
        self.pump_settings_timeline = OrderedDict()
        self.reservoir_change_timeline = OrderedDict()

        # Notes data
        self.note_timeline = OrderedDict()

        self.num_absorption_times = 0 # tmp for carb duration counts

        self._parse()

    def _parse(self):

        if self.device_data_json is not None and len(self.device_data_json) > 0:
            self.data_parser_map[self.api_version]()

        try:
            if self.notes_json is not None and len(self.notes_json) > 0:
                self.notes_parser_map[self.api_version]()
        except Exception as e:
            logger.warning("Failed to parse notes: {}".format(e))

    def load_from_api(self, tp_api_obj, start_date, end_date, user_id=None, save_data=True, user_group_name=""):
        """
        Use Tidepool API to download Tidepool user_obj data

        Args:
            tp_api_obj (TidepoolAPI): credentialed api object
            save_dir (str): directory where the user_obj data will be stored
            start_date (dt.DateTime): start date of data collection
            end_date dt.DateTime: end date of data collection
            user_id (str): Optional user_obj id if the login credentials are an observer
        """
        tp_api_obj.login()

        # Create directory based on user_obj id whose data this is
        user_id_of_data = user_id
        if user_id_of_data is None:
            user_id_of_data = tp_api_obj.get_login_user_id()

        if save_data:
            save_dir = create_user_dir(user_id_of_data, start_date, end_date, user_group_name=user_group_name)

        # Download and save events
        user_event_json = tp_api_obj.get_user_event_data(start_date, end_date, observed_user_id=user_id)
        if save_data:
            json.dump(user_event_json, open(os.path.join(save_dir, self._event_data_filename), "w"))

        # Download and save notes
        notes_json = tp_api_obj.get_notes(start_date, end_date, observed_user_id=user_id)
        if save_data:
            json.dump(notes_json, open(os.path.join(save_dir, self._notes_filename), "w"))

        # TODO: add profile metadata

        tp_api_obj.logout()

        # Document this operation and save
        creation_metadata = {
            "date_created": dt.datetime.now().isoformat(),
            "api_version": "v1",
            "data_start_date": start_date.strftime(DATESTAMP_FORMAT),
            "data_end_date": end_date.strftime(DATESTAMP_FORMAT),
            "user_id": user_id,
            "observer_id": tp_api_obj.get_login_user_id()
        }
        if save_data:
            json.dump(creation_metadata, open(os.path.join(save_dir, self._creation_meta_filename), "w"))

        self.api_version = "v1"
        self.device_data_json = user_event_json
        self.notes_json = notes_json

        self._parse()

    def load_from_dir(self, path_to_user_data_dir):
        """
        Load stored data and parse.
        """
        # TODO: require device data?
        event_data_json = json.load(open(os.path.join(path_to_user_data_dir, self._event_data_filename)))
        self.device_data_json = event_data_json

        try:
            notes_json = json.load(open(os.path.join(path_to_user_data_dir, self._notes_filename)))
            self.notes_json = notes_json
        except FileNotFoundError:
            logger.warning("No Notes data found.")
            self.notes_json = None

        try:
            creation_meta_json = json.load(open(os.path.join(path_to_user_data_dir, self._creation_meta_filename)))
            creation_meta_json["data_start_date"] = parse_tidepool_api_date_str(creation_meta_json["data_start_date"])
            creation_meta_json["data_end_date"] = parse_tidepool_api_date_str(creation_meta_json["data_end_date"])
            self.id = creation_meta_json["user_id"]
        except FileNotFoundError:
            logger.warning("No creation metadata found.")

            self.id = None

        self._parse()

    def add_event_to_timeline(self, timeline, event_datetime, event):

        # Enforce order of datetime keys
        if len(timeline) > 0:
            last_datetime_added = next(iter(timeline))
            if event_datetime > last_datetime_added:
                raise Exception("Times for events not in order")

        timeline[event_datetime] = event

    def parse_data_json_v1(self):
        """
        Parse the json list into different event types

        Assumes and asserts device data is already sorted, but checks for direction of sort.
        """
        # time example: "2020-01-02T23:15:12.611Z"

        first_event_datetime = parse_tidepool_api_date_str(self.device_data_json[0]["time"])
        last_event_datetime = parse_tidepool_api_date_str(self.device_data_json[-1]["time"])

        if first_event_datetime < last_event_datetime:
            self.device_data_json.reverse()

        for i, event in enumerate(self.device_data_json):

            try:

                event_type = event.get("type")

                mid = event.get("id")
                upload_id = event.get("uploadId")
                origin = event.get("origin")
                if origin is not None:
                    source_name = origin.get("name", "")
                else:
                    source_name = event.get("deviceId", "")

                time_str = event["time"]
                event_datetime = parse_tidepool_api_date_str(time_str)

                if event_type == "smbg":
                    smbg = TidepoolManualGlucoseMeasurement(event["value"], event["units"], mid, upload_id, source_name)

                elif event_type == "cbg":

                    cbg = TidepoolCGMGlucoseMeasurement(event["value"], event["units"], mid, upload_id, source_name)
                    self.add_event_to_timeline(self.cgm_timeline, event_datetime, cbg)

                elif event_type == "food":

                    carb_key = "carbohydrate"
                    if carb_key not in event["nutrition"]:
                        carb_key = "carbohydrates"

                    value = float(event["nutrition"][carb_key]["net"])
                    units = event["nutrition"][carb_key]["units"]

                    duration_sec = None
                    if event.get("payload") and event["payload"].get("com.loudnate.CarbKit.HKMetadataKey.AbsorptionTimeMinutes"):
                        duration_sec = int(event["payload"].get("com.loudnate.CarbKit.HKMetadataKey.AbsorptionTimeMinutes"))
                        self.num_absorption_times += 1

                    food = TidepoolFood(value, units, mid, upload_id, source_name, duration_sec=duration_sec)
                    self.add_event_to_timeline(self.food_timeline, event_datetime, food)

                elif event_type == "basal":

                    try:
                        duration_ms = float(event["duration"])
                    except Exception as e:
                        logger.debug("Can't convert basal duration to float.")
                        raise e

                    duration_hours = duration_ms / 1000.0 / 3600

                    if event.get("rate") in [None, ""] and event["deliveryType"] == "suspend":
                        basal = TidepoolBasal(0, "U/hr", duration_hours, "suspend", mid, upload_id, source_name)
                    else:
                        basal = TidepoolBasal(event["rate"], "U/hr", duration_hours, event["deliveryType"], mid, upload_id, source_name)

                    self.add_event_to_timeline(self.basal_timeline, event_datetime, basal)

                elif event_type == "bolus":

                    sub_type = event["subType"]

                    if sub_type == "square":
                        bolus = TidepoolSquareBolus(event["extended"], "Units", event["duration"], mid, upload_id, source_name)
                    else:
                        bolus = TidepoolBolus(event["normal"], "Units", mid, upload_id, source_name)

                    self.add_event_to_timeline(self.bolus_timeline, event_datetime, bolus)

                    # On pumps where carbs are entered and stored with bolus
                    # TODO: make sure this doesn't clash with food events
                    if event.get("carbInput") not in ["", None, "0.0"]:
                        logger.debug("Found 'stitched' bolus/wizard. Adding carb.")
                        carb_input = float(event["carbInput"])
                        food = TidepoolFood(carb_input, "g", mid, upload_id, source_name)
                        self.add_event_to_timeline(self.food_timeline, event_datetime, food)

                elif event_type == "insulin":

                    try:
                        dose_total = event["dose.total"]
                    except KeyError:
                        dose_total = event["dose"]["total"]

                    bolus = TidepoolBolus(dose_total, "Units", mid, upload_id, source_name)  # manual
                    self.bolus_timeline[event_datetime] = bolus

                    self.add_event_to_timeline(self.bolus_timeline, event_datetime, bolus)

                elif event_type == "wizard":

                    wizard_event = TidepoolWizard(event, mid, upload_id, source_name)
                    self.add_event_to_timeline(self.wizard_timeline, event_datetime, wizard_event)

                    if wizard_event.raw_data.get("carbInput") is not None:
                        logger.debug("Wizard Carb Event!")
                        food = TidepoolFood(wizard_event.raw_data["carbInput"], "g", mid, upload_id, source_name)
                        self.add_event_to_timeline(self.food_timeline, event_datetime, food)

                elif event_type == "deviceEvent":

                    sub_type = event.get("subType")

                    if sub_type == "status":
                        # Suspended pump
                        self.unparsed_events["{}-{}".format(event_type, sub_type)] += 1
                    elif sub_type == "prime":
                        # Prime pump
                        self.unparsed_events["{}-{}".format(event_type, sub_type)] += 1
                    elif sub_type == "timeChange":
                        time_change = TidepoolTimeChange(event, mid, upload_id, source_name)
                        self.add_event_to_timeline(self.time_change_timeline, event_datetime, time_change)
                    elif sub_type == "reservoirChange":
                        reservoir_change = TidepoolReservoirChange(event, mid, upload_id, source_name)
                        self.add_event_to_timeline(self.reservoir_change_timeline, event_datetime, reservoir_change)
                    else:
                        self.unparsed_events["{}-{}".format(event_type, sub_type)] += 1
                        logger.debug("Device event not parsed. Sub type: {}".format(sub_type))

                elif event_type == "pumpSettings":
                    self.add_event_to_timeline(self.pump_settings_timeline, event_datetime, event)

                elif event_type == "upload":
                    self.unparsed_events["{}".format(event_type)] += 1
                elif event_type == "cgmSettings":
                    self.unparsed_events["{}".format(event_type)] += 1
                elif event_type == "physicalActivity":
                    self.unparsed_events["{}".format(event_type)] += 1
                else:
                    raise Exception("Unknown event type")

            except Exception as e:
                logger.warning("Dropped an event of type {} due to error: {}".format(event_type, e))

        if len(self.unparsed_events) > 0:
            logger.warning("Unparsed events: {}".format(self.unparsed_events))

    def parse_notes_json_v1(self):
        """
        Parse the Tidepool notes json.
        """
        for note in self.notes_json.get("messages", []):

            note_obj = TidepoolNote().from_raw(note)
            self.note_timeline[note_obj.note_datetime] = note_obj

    def get_num_days_span(self, data_type="basal"):
        """
        Count the number of days between the beginning and end of an event timeline.
        """
        if data_type == "cgm":
            timeline = self.cgm_timeline
        elif data_type == "bolus":
            timeline = self.bolus_timeline
        elif data_type == "basal":
            timeline = self.basal_timeline
        elif data_type == "food":
            timeline = self.food_timeline
        else:
            raise Exception(f"Data type unknown: {data_type}")

        data_end_date = list(timeline.items())[0][0]
        data_start_date = list(timeline.items())[-1][0]

        total_days_in_data = int((data_end_date - data_start_date).total_seconds() / 3600 / 24)

        return total_days_in_data

    def analyze_duplicates(self, time_diff_thresh_sec=5):

        self.cgm_timeline, found_dupe_cgm = self.deduplicate_timeline(self.cgm_timeline,
                                                                          value_diff_min_thres=1.0,
                                                                          time_diff_thresh_sec=60 * 60,  # within one hour and of different source
                                                                          description="CBG")

        self.bolus_timeline, found_dupe_bolus = self.deduplicate_timeline(self.bolus_timeline,
                                    value_diff_min_thres=1e-6,
                                  time_diff_thresh_sec=time_diff_thresh_sec,
                                  description="Bolus")

        self.basal_timeline, found_dupe_basal = self.deduplicate_timeline(self.basal_timeline,
                                  value_diff_min_thres=1e-6,
                                  time_diff_thresh_sec=time_diff_thresh_sec,
                                  description="Basal")

        self.food_timeline, found_dupe_food = self.deduplicate_timeline(self.food_timeline,
                                  value_diff_min_thres=1e-6,
                                  time_diff_thresh_sec=time_diff_thresh_sec,
                                  description="Food")

    def deduplicate_timeline(self, timeline, value_diff_min_thres, time_diff_thresh_sec=60, description=None):
        """
        Assumes the events are in order and that duplicates are next to one another.
        """
        logger.warning("De-duplicating {} data...".format(description))

        duplicate_candidates = []
        deduped_timeline = OrderedDict()
        found_dupes = False

        for i, (candidate_dt, candidate_event) in enumerate(list(timeline.items())):

            if i == 0:
                base_event = candidate_event
                base_dt = candidate_dt
                deduped_timeline[candidate_dt] = candidate_event
                continue

            if self.is_duplicate_event_basic(candidate_event, candidate_dt, base_event, base_dt, value_diff_min_thres, time_diff_thresh_sec):
                duplicate_candidates.append((base_dt, base_event, candidate_dt, candidate_event))
            else:
                deduped_timeline[candidate_dt] = candidate_event

                base_event = candidate_event
                base_dt = candidate_dt

        if len(duplicate_candidates) > 0:
            found_dupes = True
            logger.info(f"In {description} Data: Total events {len(timeline)}. Num dupes {len(duplicate_candidates)}")
            # deduped_timeline = timeline  # TODO expose optionally don't dedupe after analysis
            logger.info(f"Num {description} events after dedupe: {len(deduped_timeline)}")
        else:
            logger.info(f"Zero duplicates found in {description} data.")

        return deduped_timeline, found_dupes

    def is_duplicate_event_basic(self, event1, event1_dt, event2, event2_dt, value_diff_thresh, time_diff_thresh_different_sources):
        """
        According to backend TP engineers, duplicates are likely due to similar events from multiple sources.

        So if an event is from a different source, is close in specified value, and close in specified time, it is
        considered a duplicate.
        """

        if TidepoolMeasurement in type(event1).__mro__:
            seconds_between = abs(event1_dt - event2_dt).total_seconds()

            is_same_source = event1.source == event2.source
            is_same_event = event1.is_equal(event2, value_diff_thresh)

            if is_same_source and is_same_event:
                is_same_time = seconds_between <= 5  # within 5 seconds on the same device
            elif not is_same_source and is_same_event:
                is_same_time = seconds_between <= time_diff_thresh_different_sources
            else:
                is_same_time = False

            is_dupe = is_same_event and is_same_time and is_same_source

        elif isinstance(event1, TidepoolWizard):
            pass
        else:
            raise Exception()

        return is_dupe

    def get_insulin_stats(self, start_date, end_date):
        """
        Get the sum of insulin with the two datetimes, inclusive.

        Args:
            start_date (dt.DateTime): start date
            end_date (dt.DateTime): end date

        Returns:
            (float, int, float, int): sum and counts of bolus and basal
        """

        insulin_stats = {
            "total_bolus": 0.0,
            "num_bolus_events": 0,
            "total_basal": 0.0,
            "num_basal_events": 0
        }

        for time, bolus in self.bolus_timeline.items():
            if start_date <= time <= end_date:
                insulin_stats["total_bolus"] += bolus.get_value()
                insulin_stats["num_bolus_events"] += 1

        for time, basal in self.basal_timeline.items():
            if start_date <= time <= end_date:
                rate = basal.get_value()
                amount_delivered = rate * basal.get_duration_hours()
                insulin_stats["total_basal"] += amount_delivered
                insulin_stats["num_basal_events"] += 1

        insulin_stats["total_insulin"] = insulin_stats["total_bolus"] + insulin_stats["total_basal"]

        if insulin_stats["num_bolus_events"] + insulin_stats["num_basal_events"] == 0:
            insulin_stats["total_bolus"] = np.nan
            insulin_stats["total_basal"] = np.nan

        return insulin_stats

    def get_carb_stats(self, start_date, end_date):
        """
        Get the sum of carbs with two datetimes, inclusive.
        Args:
            start_date (dt.DateTime): start date
            end_date (dt.DateTime): end date

        Returns:
            (float, int): total carbs and number of carb events
        """
        carb_stats = {
            "total_carbs": 0.0,
            "num_carb_events": 0
        }

        for time, food in self.food_timeline.items():
            if start_date <= time <= end_date:
                carb_stats["total_carbs"] += food.get_value()
                carb_stats["num_carb_events"] += 1

        return carb_stats

    def get_cgm_stats(self, start_date, end_date):
        """
        Compute cgm stats with dates

        Args:
            start_date (dt.DateTime): start date
            end_date (dt.DateTime): end date

        Returns:
            (float, float): geo mean and std
        """

        stats = {
            "cgm_geo_mean": np.nan,
            "cgm_geo_std": np.nan,
            "cgm_mean": np.nan,
            "cgm_std": np.nan,
            "cgm_percent_available": np.nan
        }
        cgm_values = []
        for time, cgm_event in self.cgm_timeline.items():
            if start_date <= time <= end_date:
                cgm_value = cgm_event.get_value()
                cgm_values.append(cgm_value)

        if len(cgm_values) > 1:

            seconds_per_cgm_value = (end_date - start_date).total_seconds() / 300

            stats.update({
                "cgm_geo_mean": gmean(cgm_values),
                "cgm_geo_std": gstd(cgm_values),
                "cgm_mean": np.nanmean(cgm_values),
                "cgm_std": np.nanstd(cgm_values),
                "cgm_percent_available": len(cgm_values) / seconds_per_cgm_value,
                "cgm_diff": cgm_values[0] - cgm_values[-1],
            })

        return stats

    def detect_circadian_hr(self, start_time=dt.datetime.min, end_time=dt.datetime.max, win_radius=3):
        """
        Count carb intake per hour and use the minimum as a likely cutoff for daily circadian
        boundary. Useful for window analysis.

        Args:
            start_time: datetime
                The start date of projects to use for detection

            end_time: datetime
                The end date of projects to use for detection

        Returns:
            int: hour of least user_obj activity
        """

        # Food Entries
        hour_ctr = defaultdict(int)
        if len(self.food_timeline) > 0:
            for dt, carb in self.food_timeline.items():
                if start_time <= dt <= end_time:

                    for radius in range(-win_radius, win_radius + 1):
                        associated_hour = (dt.hour + radius) % 24
                        hour_ctr[associated_hour] += 1

        # Wizard Entries
        if len(self.wizard_timeline) > 0:
            for dt, wiz in self.wizard_timeline.items():
                if start_time <= dt <= end_time:

                    for radius in range(-win_radius, win_radius + 1):
                        associated_hour = (dt.hour + radius) % 24
                        hour_ctr[associated_hour] += 1

        if len(hour_ctr) == 0:
            min_hr, min_count = 0, 0
        else:
            min_hr, min_count = min(hour_ctr.items(), key=itemgetter(1))

        return min_hr

    def detect_circadian_bg_velocity(self, start_time=dt.datetime.min, end_time=dt.datetime.max, win_radius=3):
        """
        Hour at which sum of positive bg velocity is the minimum, ie least likely time carb or insulin effects.
        """
        hour_ctr = defaultdict(int)
        if len(self.cgm_timeline) > 0:
            prev_val = None
            prev_dt = None
            for dt, cbg in self.cgm_timeline.items():
                if start_time <= dt <= end_time:

                    if prev_val:
                        velocity = (cbg.value - prev_val) / (dt - prev_dt).total_seconds()
                        if velocity > 0:
                            for radius in range(-win_radius, win_radius + 1):
                                associated_hour = (dt.hour + radius) % 24
                                hour_ctr[associated_hour] += velocity

                        prev_val = cbg.value
                        prev_dt = dt
                    else:
                        prev_val = cbg.value
                        prev_dt = dt

        if len(hour_ctr) == 0:
            min_hr, min_count = 0, 0
        else:
            min_hr, min_count = min(hour_ctr.items(), key=itemgetter(1))

        return min_hr

    def compute_window_stats(self, start_date, end_date, use_circadian=True, window_size_hours=24, hop_size_hours=24, plot_hop_raw=False):
        """
        Compute stats for a user_obj in a specified window.

        Args:
            start_date (dt.DateTime): start date
            end_date (dt.DateTime): end date
            use_circadian (bool): Use circadian hour instead of timestamp midnight for day boundary

        Returns:
            pd.DataFrame: rows are days, columns are stats
        """
        circadian_hour = 0
        if use_circadian:
            circadian_hour = self.detect_circadian_hr(win_radius=9)
            circadian_hour_bg = self.detect_circadian_bg_velocity(win_radius=5)

            logger.debug("Circadian Hour Activity {}. Circadian Hour BG Velocity {}".format(circadian_hour, circadian_hour_bg))
            circadian_hour = circadian_hour_bg

        dates = []

        start_datetime_withoffset = dt.datetime(year=start_date.year, month=start_date.month, day=start_date.day,
                                                hour=circadian_hour)

        all_window_stats = []
        segment = 0

        window_start_datetime = start_datetime_withoffset
        window_end_datetime = window_start_datetime + dt.timedelta(hours=window_size_hours)

        while True:

            dates.append(window_start_datetime.date())

            window_stats = {
                "segment_idx": segment,
                "start_date": window_start_datetime,
                "end_date": window_end_datetime
            }

            insulin_stats = self.get_insulin_stats(window_start_datetime, window_end_datetime)
            carb_stats = self.get_carb_stats(window_start_datetime, window_end_datetime)
            cgm_stats = self.get_cgm_stats(window_start_datetime, window_end_datetime)

            window_stats.update(insulin_stats)
            window_stats.update(carb_stats)
            window_stats.update(cgm_stats)

            all_window_stats.append(window_stats)

            if plot_hop_raw:
                plot_raw_data(self, window_start_datetime, window_end_datetime)

            # Update
            segment += 1
            window_start_datetime += dt.timedelta(hours=hop_size_hours)
            window_end_datetime = window_start_datetime + dt.timedelta(hours=window_size_hours)

            if window_end_datetime > end_date:
                break

        return all_window_stats

    def get_nearest_bg(self, lookup_time):
        pass

    def get_count_basal_event_types(self):
        counts = defaultdict(int)
        for _, event in self.basal_timeline.items():
            counts[event.delivery_type] += 1
        return counts

    def describe_timeline(self, timeline):

        num_events = len(timeline)
        event_dates = list(timeline.keys())
        num_days = 0
        num_events_per_day = 0
        max_events_any_day = 0
        min_events_any_day = 0
        if num_events > 1:

            dates = defaultdict(int)
            for event_datetime, event in timeline.items():
                dates[event_datetime.date()] += 1

            num_days = (event_dates[0] - event_dates[-1]).total_seconds() / 60 / 60 / 24
            num_events_per_day = num_events / num_days
            max_events_any_day = max(dates.values())
            min_events_any_day = min(dates.values())

        stats = {
            "num_events": num_events,
            "num_days": num_days,
            "num_events_per_day": num_events_per_day,
            "max_events_any_day": max_events_any_day,
            "min_events_any_day": min_events_any_day
        }
        return stats

    def describe(self):

        timelines_to_describe = [
            ("basal", self.basal_timeline),
            ("bolus", self.bolus_timeline),
            ("food", self.food_timeline),
            ("glucose", self.cgm_timeline),
            ("notes", self.note_timeline)
        ]

        stats_dict = dict()

        for name, timeline in timelines_to_describe:

            timeline_stats = self.describe_timeline(timeline)

            stats_dict[name] = timeline_stats

        stats_dict["glucose"]["cbg_percent_coverage"] = stats_dict["glucose"]["num_events_per_day"] / 288.0 * 100

        return stats_dict

    def is_keyword_in_data(self, keyword):

        events = []
        for event in self.device_data_json:
            field = get_recursively(event, keyword)
            if field:
                events.append((event, field))

        return events

