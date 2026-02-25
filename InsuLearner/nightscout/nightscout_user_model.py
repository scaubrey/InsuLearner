import datetime as dt
import json
import os
from collections import OrderedDict
from zoneinfo import ZoneInfo

from InsuLearner.tidepool.tidepool_user_model import (
    TidepoolBasal,
    TidepoolBolus,
    TidepoolCGMGlucoseMeasurement,
    TidepoolFood,
    TidepoolUser,
)
from InsuLearner.util import DATESTAMP_FORMAT, create_user_dir, get_logger, parse_tidepool_api_date_str

logger = get_logger(__name__)


class NightscoutUser(TidepoolUser):
    """
    Nightscout-backed user class that mirrors TidepoolUser's interface.
    """

    def __init__(self, entries_json=None, treatments_json=None, profiles_json=None, api_version="v1"):
        self._entries_filename = "entries.json"
        self._treatments_filename = "treatments.json"
        self._profiles_filename = "profiles.json"
        self._creation_meta_filename = "creation_metadata.json"

        self.entries_json = entries_json
        self.treatments_json = treatments_json
        self.profiles_json = profiles_json

        self.data_start_date = None
        self.data_end_date = None

        # Nightscout report-like preprocessed caches
        self._profile_records = []
        self._profile_treatments = []
        self._temp_basal_treatments = []
        self._combo_bolus_treatments = []
        self._latest_temp_basal_cache = None

        super().__init__(device_data_json=None, notes_json=None, api_version=api_version)

    def _parse(self):
        has_entries = self.entries_json is not None and len(self.entries_json) > 0
        has_treatments = self.treatments_json is not None and len(self.treatments_json) > 0
        has_profiles = self.profiles_json is not None and len(self.profiles_json) > 0

        if has_entries or has_treatments or has_profiles:
            self.data_parser_map[self.api_version]()

    def _normalize_user_id(self, user_id):
        return user_id.replace("https://", "").replace("http://", "").replace("/", "_")

    def _parse_nightscout_datetime(self, value):
        # Use the same parser/conversion path as the Tidepool model to keep
        # day-boundary behavior consistent in downstream window aggregation.
        return parse_tidepool_api_date_str(str(value).strip())

    def _to_millis(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if value > 1e12:
                return int(value)
            if value > 1e9:
                return int(value * 1000)
        try:
            parsed = self._parse_nightscout_datetime(value)
            return int(parsed.timestamp() * 1000)
        except Exception:
            return None

    def _get_event_datetime(self, event):
        if event.get("created_at"):
            return self._parse_nightscout_datetime(event["created_at"])
        if event.get("dateString"):
            return self._parse_nightscout_datetime(event["dateString"])
        if event.get("timestamp"):
            return self._parse_nightscout_datetime(event["timestamp"])
        if event.get("startDate"):
            return self._parse_nightscout_datetime(event["startDate"])
        if event.get("mills") is not None:
            return dt.datetime.utcfromtimestamp(float(event["mills"]) / 1000.0)
        if event.get("date") is not None:
            return dt.datetime.utcfromtimestamp(float(event["date"]) / 1000.0)

        raise ValueError("Unable to parse event datetime.")

    def _sorted_with_datetime(self, events):
        events_with_dt = []
        for event in events:
            try:
                event_datetime = self._get_event_datetime(event)
                events_with_dt.append((event_datetime, event))
            except Exception:
                self.dropped_events["unknown-datetime"] += 1
        events_with_dt.sort(key=lambda x: x[0], reverse=True)
        return events_with_dt

    def load_from_api(self, ns_api_obj, start_date, end_date, user_id=None, save_data=True, user_group_name=""):
        ns_api_obj.login()

        user_id_of_data = user_id if user_id is not None else ns_api_obj.get_login_user_id()
        safe_user_id = self._normalize_user_id(user_id_of_data)

        if save_data:
            save_dir = create_user_dir(safe_user_id, start_date, end_date, user_group_name=user_group_name)

        entries_json = ns_api_obj.get_entries(start_date, end_date)
        treatments_json = ns_api_obj.get_treatments(start_date, end_date)
        try:
            profiles_json = ns_api_obj.get_profiles(end_date=end_date)
        except Exception as e:
            logger.warning(f"Unable to load Nightscout profiles. Falling back to treatment-only basal parsing. Error: {e}")
            profiles_json = []

        if save_data:
            json.dump(entries_json, open(os.path.join(save_dir, self._entries_filename), "w"))
            json.dump(treatments_json, open(os.path.join(save_dir, self._treatments_filename), "w"))
            json.dump(profiles_json, open(os.path.join(save_dir, self._profiles_filename), "w"))

        ns_api_obj.logout()

        creation_metadata = {
            "date_created": dt.datetime.now().isoformat(),
            "api_version": self.api_version,
            "data_source": "nightscout",
            "data_start_date": start_date.strftime(DATESTAMP_FORMAT),
            "data_end_date": end_date.strftime(DATESTAMP_FORMAT),
            "user_id": safe_user_id,
            "observer_id": safe_user_id,
        }
        if save_data:
            json.dump(creation_metadata, open(os.path.join(save_dir, self._creation_meta_filename), "w"))

        self.data_start_date = start_date
        self.data_end_date = end_date
        self.device_data_json = [{"entries": entries_json, "treatments": treatments_json, "profiles": profiles_json}]
        self.entries_json = entries_json
        self.treatments_json = treatments_json
        self.profiles_json = profiles_json
        self._parse()

    def load_from_dir(self, path_to_user_data_dir):
        self.entries_json = json.load(open(os.path.join(path_to_user_data_dir, self._entries_filename)))
        self.treatments_json = json.load(open(os.path.join(path_to_user_data_dir, self._treatments_filename)))
        try:
            self.profiles_json = json.load(open(os.path.join(path_to_user_data_dir, self._profiles_filename)))
        except FileNotFoundError:
            logger.warning("No Nightscout profile data found.")
            self.profiles_json = []

        try:
            creation_meta_json = json.load(open(os.path.join(path_to_user_data_dir, self._creation_meta_filename)))
            self.data_start_date = parse_tidepool_api_date_str(creation_meta_json["data_start_date"])
            self.data_end_date = parse_tidepool_api_date_str(creation_meta_json["data_end_date"])
        except FileNotFoundError:
            logger.warning("No creation metadata found for Nightscout data.")
            self.data_start_date = None
            self.data_end_date = None

        self.device_data_json = [{"entries": self.entries_json, "treatments": self.treatments_json, "profiles": self.profiles_json}]
        self._parse()

    def _safe_float(self, value):
        try:
            return float(value)
        except Exception:
            return None

    def _duration_minutes(self, treatment):
        minutes = self._safe_float(treatment.get("duration"))
        if minutes is not None and minutes >= 0:
            return minutes

        duration_ms = self._safe_float(treatment.get("durationInMilliseconds"))
        if duration_ms is not None and duration_ms >= 0:
            return duration_ms / 1000.0 / 60.0

        return 0.0

    def _parse_time_as_seconds(self, time_str):
        try:
            hh, mm = [int(x) for x in str(time_str).split(":")[:2]]
            return hh * 3600 + mm * 60
        except Exception:
            return None

    def _preprocess_profile_store(self, container):
        if isinstance(container, dict):
            for _, value in list(container.items()):
                self._preprocess_profile_store(value)
        elif isinstance(container, list):
            for value in container:
                self._preprocess_profile_store(value)
        else:
            return

        if isinstance(container, dict) and container.get("time") and container.get("timeAsSeconds") is None:
            sec = self._parse_time_as_seconds(container.get("time"))
            if sec is not None:
                container["timeAsSeconds"] = sec

    def _prepare_profiles(self):
        profiles_raw = self.profiles_json if self.profiles_json is not None else []
        if isinstance(profiles_raw, dict):
            profiles = [profiles_raw]
        else:
            profiles = profiles_raw

        converted = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue

            prof = dict(profile)
            if not prof.get("defaultProfile"):
                start_date = prof.get("startDate", "1980-01-01")
                prof = {
                    "defaultProfile": "Default",
                    "store": {"Default": {k: v for k, v in prof.items() if k not in ["startDate", "_id", "created_at"]}},
                    "startDate": start_date,
                    "_id": profile.get("_id"),
                }

            start_mills = self._to_millis(prof.get("startDate"))
            if start_mills is None:
                start_mills = 0
            prof["mills"] = start_mills

            store = prof.get("store", {})
            if isinstance(store, dict):
                for _, profile_record in store.items():
                    self._preprocess_profile_store(profile_record)

            converted.append(prof)

        converted.sort(key=lambda x: x.get("mills", 0), reverse=True)
        self._profile_records = converted

    def _process_durations(self, treatments, keep_zero_duration):
        # Mirror Nightscout ddata.processDurations semantics.
        deduped = []
        seen_mills = set()
        for treatment in sorted(treatments, key=lambda x: x.get("mills", 0)):
            mills = treatment.get("mills")
            if mills in seen_mills:
                continue
            seen_mills.add(mills)
            deduped.append(treatment)

        end_events = [t for t in deduped if not t.get("duration")]

        def cut_if_in_interval(base, end):
            base_mills = base.get("mills", 0)
            base_dur_mins = self._safe_float(base.get("duration")) or 0.0
            end_mills = end.get("mills", 0)
            if base_mills < end_mills and base_mills + int(base_dur_mins * 60 * 1000) > end_mills:
                new_duration = (end_mills - base_mills) / 1000.0 / 60.0
                base["duration"] = new_duration
                if end.get("profile"):
                    base["cuttedby"] = end["profile"]
                    end["cutting"] = base.get("profile")

        for treatment in deduped:
            if treatment.get("duration"):
                for end_event in end_events:
                    cut_if_in_interval(treatment, end_event)

        for treatment in deduped:
            if treatment.get("duration"):
                for other in deduped:
                    cut_if_in_interval(treatment, other)

        if keep_zero_duration:
            return deduped
        return [t for t in deduped if t.get("duration")]

    def _prepare_treatments_for_basal_math(self):
        treatments = self.treatments_json if self.treatments_json is not None else []
        normalized = []
        for treatment in treatments:
            if not isinstance(treatment, dict):
                continue
            tr = dict(treatment)
            mills = self._to_millis(tr.get("mills"))
            if mills is None:
                mills = self._to_millis(tr.get("timestamp"))
            if mills is None:
                mills = self._to_millis(tr.get("created_at"))
            if mills is None:
                continue
            tr["mills"] = mills
            tr["duration"] = self._duration_minutes(tr)
            normalized.append(tr)

        profile_treatments = [t for t in normalized if t.get("eventType") == "Profile Switch"]
        combo_treatments = [t for t in normalized if t.get("eventType") == "Combo Bolus"]

        temp_treatments_raw = [t for t in normalized if t.get("eventType") and "Temp Basal" in t.get("eventType")]

        # Nightscout may emit paired temp basal records with identical mills
        # (for example, one record carries duration and another carries absolute/percent).
        # Merge only temp-basal records so other treatment types are not lost.
        temp_merged_by_mills = {}
        for tr in sorted(temp_treatments_raw, key=lambda x: x.get("mills", 0)):
            mills = tr.get("mills")
            if mills not in temp_merged_by_mills:
                temp_merged_by_mills[mills] = dict(tr)
            else:
                merged = temp_merged_by_mills[mills]
                for key, value in tr.items():
                    if value not in [None, ""] and merged.get(key) in [None, "", 0]:
                        merged[key] = value
                if (merged.get("duration") in [None, 0]) and tr.get("duration", 0) > 0:
                    merged["duration"] = tr["duration"]

        temp_treatments = list(temp_merged_by_mills.values())

        self._profile_treatments = self._process_durations(profile_treatments, keep_zero_duration=True)
        self._temp_basal_treatments = self._process_durations(temp_treatments, keep_zero_duration=False)
        self._combo_bolus_treatments = sorted(combo_treatments, key=lambda x: x.get("mills", 0))
        self._temp_basal_treatments.sort(key=lambda x: x.get("mills", 0))
        self._latest_temp_basal_cache = None

    def _infer_data_date_range_from_loaded_events(self):
        if self.data_start_date is not None and self.data_end_date is not None:
            return

        candidate_datetimes = []
        for source_events in [self.entries_json or [], self.treatments_json or []]:
            for event in source_events:
                try:
                    candidate_datetimes.append(self._get_event_datetime(event))
                except Exception:
                    continue

        if len(candidate_datetimes) == 0:
            return

        min_dt = min(candidate_datetimes)
        max_dt = max(candidate_datetimes)

        self.data_start_date = dt.datetime(min_dt.year, min_dt.month, min_dt.day, 0, 0, 0)
        self.data_end_date = dt.datetime(max_dt.year, max_dt.month, max_dt.day, 0, 0, 0) + dt.timedelta(days=1)

    def _profile_from_time(self, mills):
        if len(self._profile_records) == 0:
            return None
        for profile_record in self._profile_records:
            if mills >= profile_record.get("mills", 0):
                return profile_record
        return self._profile_records[0]

    def _active_profile_treatment_to_time(self, mills, profile_record):
        if profile_record is None:
            return None
        active = None
        for treatment in self._profile_treatments:
            if mills >= treatment.get("mills", 0) and treatment.get("mills", 0) >= profile_record.get("mills", 0):
                duration_mins = self._safe_float(treatment.get("duration")) or 0.0
                duration_ms = int(duration_mins * 60 * 1000)
                if duration_ms == 0:
                    active = treatment
                elif mills < treatment.get("mills", 0) + duration_ms:
                    active = treatment
        return active

    def _active_profile_name(self, mills, profile_record):
        if profile_record is None:
            return None

        active_name = profile_record.get("defaultProfile")
        treatment = self._active_profile_treatment_to_time(mills, profile_record)

        if treatment:
            treatment_profile = treatment.get("profile")
            store = profile_record.get("store", {})
            if treatment_profile and treatment_profile in store:
                active_name = treatment_profile
            elif treatment_profile and treatment.get("profileJson"):
                try:
                    profile_json = json.loads(treatment.get("profileJson"))
                    store[treatment_profile] = profile_json
                    self._preprocess_profile_store(store[treatment_profile])
                    active_name = treatment_profile
                except Exception:
                    pass

        return active_name

    def _seconds_from_midnight_for_profile_tz(self, mills, timezone_name):
        ts = dt.datetime.utcfromtimestamp(mills / 1000.0).replace(tzinfo=dt.timezone.utc)
        if timezone_name:
            try:
                ts = ts.astimezone(ZoneInfo(timezone_name))
            except Exception:
                pass
        return ts.hour * 3600 + ts.minute * 60 + ts.second

    def _get_scheduled_basal(self, mills):
        profile_record = self._profile_from_time(mills)
        if profile_record is None:
            return 0.0

        active_name = self._active_profile_name(mills, profile_record)
        profile_data = profile_record.get("store", {}).get(active_name, {})
        basal_entries = profile_data.get("basal", [])
        if not isinstance(basal_entries, list) or len(basal_entries) == 0:
            return 0.0

        timezone_name = profile_data.get("timezone")
        sec_since_midnight = self._seconds_from_midnight_for_profile_tz(mills, timezone_name)

        selected = None
        for basal_entry in basal_entries:
            if not isinstance(basal_entry, dict):
                continue
            sec = basal_entry.get("timeAsSeconds")
            if sec is None and basal_entry.get("time"):
                sec = self._parse_time_as_seconds(basal_entry.get("time"))
            try:
                sec = int(sec)
            except Exception:
                continue

            if sec_since_midnight >= sec:
                selected = self._safe_float(basal_entry.get("value"))

        if selected is None:
            selected = self._safe_float(basal_entries[-1].get("value"))
        if selected is None:
            selected = 0.0
        return float(selected)

    def _active_temp_basal_treatment(self, mills):
        cached = self._latest_temp_basal_cache
        if cached and cached.get("mills", 0) <= mills < cached.get("endmills", 0):
            return cached

        first = 0
        last = len(self._temp_basal_treatments) - 1
        while first <= last:
            idx = first + (last - first) // 2
            treatment = self._temp_basal_treatments[idx]
            start = treatment.get("mills", 0)
            end = treatment.get("endmills", start)

            if start <= mills < end:
                self._latest_temp_basal_cache = treatment
                return treatment
            if mills < start:
                last = idx - 1
            else:
                first = idx + 1

        return None

    def _active_combo_bolus_treatment(self, mills):
        active = None
        for treatment in self._combo_bolus_treatments:
            duration_mins = self._safe_float(treatment.get("duration")) or 0.0
            end_mills = treatment.get("mills", 0) + int(duration_mins * 60 * 1000)
            if treatment.get("mills", 0) < mills < end_mills:
                active = treatment
        return active

    def _get_temp_basal_value(self, mills):
        basal = self._get_scheduled_basal(mills)
        temp_basal = basal
        combo_bolus_basal = 0.0

        treatment = self._active_temp_basal_treatment(mills)
        combo_treatment = self._active_combo_bolus_treatment(mills)

        if treatment:
            absolute = self._safe_float(treatment.get("absolute"))
            if absolute is not None and (self._safe_float(treatment.get("duration")) or 0.0) > 0:
                temp_basal = absolute
            else:
                percent = self._safe_float(treatment.get("percent"))
                if percent is not None:
                    temp_basal = basal * (100.0 + percent) / 100.0

        if combo_treatment:
            relative = self._safe_float(combo_treatment.get("relative"))
            if relative is not None:
                combo_bolus_basal = relative

        return {
            "basal": basal,
            "tempbasal": temp_basal,
            "combobolusbasal": combo_bolus_basal,
            "totalbasal": temp_basal + combo_bolus_basal,
            "treatment": treatment,
            "combobolustreatment": combo_treatment,
        }

    def _rebuild_basal_timeline_from_nightscout_report_logic(self):
        if self.data_start_date is None or self.data_end_date is None:
            return OrderedDict()

        # Prepare treatment range helpers used by getTempBasal-like calculations.
        for treatment in self._temp_basal_treatments:
            duration_mins = self._safe_float(treatment.get("duration")) or 0.0
            treatment["endmills"] = treatment.get("mills", 0) + int(duration_mins * 60 * 1000)

        rebuilt = []
        base_basal_insulin = 0.0
        positive_temps = 0.0
        negative_temps = 0.0
        current = self.data_start_date
        while current < self.data_end_date:
            next_time = min(current + dt.timedelta(minutes=5), self.data_end_date)
            duration_hours = (next_time - current).total_seconds() / 3600.0

            basal_value = self._get_temp_basal_value(int(current.timestamp() * 1000))
            base_basal_insulin += basal_value["basal"] * duration_hours
            temp_part = (basal_value["tempbasal"] - basal_value["basal"]) * duration_hours
            if temp_part > 0:
                positive_temps += temp_part
            elif temp_part < 0:
                negative_temps += temp_part
            delivery_type = "temp" if basal_value["treatment"] else "scheduled"
            source = f"nightscout-report-{current.isoformat()}"

            basal_event = TidepoolBasal(
                basal_value["totalbasal"],
                "U/hr",
                duration_hours,
                delivery_type,
                None,
                None,
                source,
            )
            rebuilt.append((current, basal_event))
            current = next_time

        logger.info(
            "Nightscout basal breakdown: base=%.2fU temp_plus=%.2fU temp_minus=%.2fU total=%.2fU",
            base_basal_insulin,
            positive_temps,
            negative_temps,
            base_basal_insulin + positive_temps + negative_temps,
        )

        timeline = OrderedDict()
        for event_datetime, basal_event in sorted(rebuilt, key=lambda x: x[0], reverse=True):
            timeline[event_datetime] = basal_event

        return timeline

    def parse_data_json_v1(self):
        entries = self.entries_json if self.entries_json is not None else []
        treatments = self.treatments_json if self.treatments_json is not None else []

        # Needed for fixture/constructor-based usage where load_from_api is not called.
        self._infer_data_date_range_from_loaded_events()

        # Parse CGM
        for event_datetime, entry in self._sorted_with_datetime(entries):
            try:
                glucose_value = entry.get("sgv")
                if glucose_value is None:
                    glucose_value = entry.get("mbg")

                if glucose_value is None:
                    self.unparsed_events["entry-no-glucose"] += 1
                    continue

                units = entry.get("units", "mg/dL")
                source = entry.get("device", "nightscout")
                mid = entry.get("_id")
                upload_id = entry.get("sysTime", entry.get("dateString"))

                cgm = TidepoolCGMGlucoseMeasurement(glucose_value, units, mid, upload_id, source)
                self.add_event_to_timeline(self.cgm_timeline, event_datetime, cgm)
            except Exception:
                self.dropped_events["entry"] += 1

        # Parse food/bolus
        for event_datetime, treatment in self._sorted_with_datetime(treatments):
            event_type = treatment.get("eventType", "unknown")
            mid = treatment.get("_id")
            upload_id = treatment.get("identifier", treatment.get("created_at"))
            source = treatment.get("enteredBy", "nightscout")

            parsed_something = False

            carbs = treatment.get("carbs")
            if carbs not in [None, ""]:
                try:
                    food = TidepoolFood(carbs, "g", mid, upload_id, source)
                    self.add_event_to_timeline(self.food_timeline, event_datetime, food)
                    parsed_something = True
                except Exception:
                    self.dropped_events["food"] += 1

            insulin = treatment.get("insulin", treatment.get("enteredinsulin"))
            if insulin not in [None, ""]:
                try:
                    bolus = TidepoolBolus(insulin, "Units", mid, upload_id, source)
                    self.add_event_to_timeline(self.bolus_timeline, event_datetime, bolus)
                    parsed_something = True
                except Exception:
                    self.dropped_events["bolus"] += 1

            # Temp basal treatments are consumed by report-style basal math below.
            if treatment.get("eventType") and "Temp Basal" in treatment.get("eventType"):
                parsed_something = True

            if not parsed_something:
                self.unparsed_events[f"treatment-{event_type}"] += 1

        # Nightscout report-style basal calculation:
        # total basal = base scheduled basal + temp basal deltas (+ combo relative).
        self._prepare_profiles()
        self._prepare_treatments_for_basal_math()
        self.basal_timeline = self._rebuild_basal_timeline_from_nightscout_report_logic()

        if len(self.unparsed_events) > 0:
            logger.warning(f"Unparsed events: {self.unparsed_events}")

        if len(self.dropped_events) > 0:
            logger.warning(f"Dropped events: {self.dropped_events}")
