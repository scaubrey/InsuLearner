import datetime as dt
import hashlib

import requests

from InsuLearner.util import get_logger

logger = get_logger(__name__)


class NightscoutAPI(object):
    """
    Class for downloading entries/treatments from a Nightscout site.
    """

    def __init__(self, base_url, api_secret=None, token=None, timeout_sec=30):
        self.base_url = base_url.rstrip("/")
        self.entries_url = f"{self.base_url}/api/v1/entries.json"
        self.treatments_url = f"{self.base_url}/api/v1/treatments.json"
        self.profile_url = f"{self.base_url}/api/v1/profile.json"

        self.token = token
        self.timeout_sec = timeout_sec
        self._headers = {"Content-Type": "application/json"}
        self._api_secret_plain = api_secret
        self._api_secret_sha1 = hashlib.sha1(api_secret.encode("utf-8")).hexdigest() if api_secret else None

        if api_secret:
            # Most Nightscout deployments accept plaintext in `api-secret`.
            self._headers["api-secret"] = api_secret

    def login(self):
        """
        Nightscout does not require a login flow for token/api-secret auth.
        """
        return True

    def logout(self):
        return True

    def get_login_user_id(self):
        return self.base_url

    def _date_to_nightscout_str(self, date_obj):
        return date_obj.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def _normalize_day_range(self, start_date, end_date):
        # Match Tidepool behavior: query whole calendar days, not wall-clock instants.
        start = dt.datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)
        end = dt.datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)
        return start, end

    def _request_json_once(self, url, params, use_hashed_secret=False):
        request_params = dict(params)
        if self.token:
            request_params["token"] = self.token

        headers = dict(self._headers)
        if use_hashed_secret and self._api_secret_sha1:
            headers["api-secret"] = self._api_secret_sha1

        response = requests.get(url, headers=headers, params=request_params, timeout=self.timeout_sec)
        if response.status_code >= 400:
            body_excerpt = response.text[:500]
            raise requests.HTTPError(
                f"HTTP {response.status_code} for {response.url}. Response: {body_excerpt}",
                response=response,
            )
        return response.json()

    def _request_with_optional_auth_fallback(self, url, params):
        try:
            data = self._request_json_once(url, params, use_hashed_secret=False)
            return data, False
        except requests.HTTPError as e:
            status_code = None
            if getattr(e, "response", None) is not None:
                status_code = e.response.status_code

            # Retry once for auth-only failures with hashed api-secret.
            if self._api_secret_sha1 and status_code in [401, 403]:
                data = self._request_json_once(url, params, use_hashed_secret=True)
                return data, True
            raise

    def _fetch_all_pages(self, url, params, page_size=10000):
        first_page, used_hashed_secret = self._request_with_optional_auth_fallback(url, params)

        if not isinstance(first_page, list):
            return first_page

        all_rows = list(first_page)
        if len(first_page) < page_size:
            return all_rows

        skip = page_size
        while True:
            page_params = dict(params)
            page_params["skip"] = skip
            page = self._request_json_once(url, page_params, use_hashed_secret=used_hashed_secret)

            if not isinstance(page, list) or len(page) == 0:
                break

            all_rows.extend(page)
            if len(page) < page_size:
                break
            skip += page_size

        return all_rows

    def _fetch_single_page(self, url, params):
        data, _ = self._request_with_optional_auth_fallback(url, params)
        return data

    def _build_entries_params(self, start_date, end_date):
        start_date, end_date = self._normalize_day_range(start_date, end_date)
        start_iso = self._date_to_nightscout_str(start_date)
        end_iso = self._date_to_nightscout_str(end_date)
        return {
            "find[dateString][$gte]": start_iso,
            "find[dateString][$lte]": end_iso,
            "count": 10000,
        }

    def _build_treatments_params(self, start_date, end_date):
        start_date, end_date = self._normalize_day_range(start_date, end_date)
        start_iso = self._date_to_nightscout_str(start_date)
        end_iso = self._date_to_nightscout_str(end_date)
        return {
            "find[created_at][$gte]": start_iso,
            "find[created_at][$lte]": end_iso,
            "count": 10000,
        }

    def get_entries(self, start_date, end_date):
        return self._fetch_all_pages(
            self.entries_url,
            self._build_entries_params(start_date, end_date),
            page_size=10000,
        )

    def get_treatments(self, start_date, end_date):
        return self._fetch_all_pages(
            self.treatments_url,
            self._build_treatments_params(start_date, end_date),
            page_size=10000,
        )

    def _build_profiles_params(self, end_date):
        end_iso = self._date_to_nightscout_str(end_date)
        return {
            "find[startDate][$lte]": end_iso,
            "count": 200,
        }

    def get_profiles(self, end_date):
        # Many Nightscout deployments do not support stable pagination on profile.json.
        # Use a single request and let Nightscout return the latest relevant records.
        return self._fetch_single_page(
            self.profile_url,
            self._build_profiles_params(end_date),
        )
