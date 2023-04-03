__author__ = "Cameron Summers"

# -*- coding: utf-8 -*-
"""
Utilities for downloading projects from Tidepool API

Reference: https://developer.tidepool.org/tidepool-api/index/
"""

import os
import datetime as dt
import sys
import requests

import logging
from InsuLearner.util import DATESTAMP_FORMAT,  get_logger

logger = get_logger(__name__)


class TidepoolAPI(object):
    """
    Class representing a user_obj with a Tidepool account.

    # TODO: Add checks and enforcement for order of events
    # TODO: Add helper functions for getting earlier/latest data
    """

    def __init__(self, username, password):

        self.login_url = "https://api.tidepool.org/auth/login"

        self.user_data_url = "https://api.tidepool.org/data/{user_id}"
        self.logout_url = "https://api.tidepool.org/auth/logout"
        self.users_sharing_to_url = "https://api.tidepool.org/metadata/users/{user_id}/users"
        self.users_sharing_with_url = "https://api.tidepool.org/access/groups/{user_id}"
        self.invitations_url = "https://api.tidepool.org/confirm/invitations/{user_id}"
        self.accept_invitations_url = "https://api.tidepool.org/confirm/accept/invite/{observer_id}/{user_id}"
        self.user_notes_url = "https://api.tidepool.org/message/notes/{user_id}"

        self.username = username
        self.password = password

        self._login_user_id = None
        self._login_headers = None

    def _check_login(func):
        """
        Decorator for enforcing login.
        """
        def is_logged_in(self, *args, **kwargs):
            if self._login_headers is None or self._login_user_id is None:
                raise Exception("Not logged in.")
            return func(self, *args, **kwargs)
        return is_logged_in

    def _check_http_error(func):
        """
        Decorator to batch handle failed http requests.
        """
        def response_is_ok(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except requests.HTTPError as e:
                logger.info("Failed request. HTTPError: {}".format(e))
        return response_is_ok

    def login(self):
        """
        Login to Tidepool API
        """
        login_response = requests.post(self.login_url, auth=(self.username, self.password))

        xtoken = login_response.headers["x-tidepool-session-token"]
        user_id_master = login_response.json()["userid"]

        self._login_user_id = user_id_master
        self._login_headers = {
            "x-tidepool-session-token": xtoken,
            "Content-Type": "application/json"
        }

    @_check_http_error
    @_check_login
    def logout(self):
        """
        Logout of Tidepool API

        Args:
            auth:

        Returns:

        """
        logout_response = requests.post(self.logout_url, auth=(self.username, self.password))
        logout_response.raise_for_status()

    @_check_login
    def get_pending_observer_invitations(self):
        """
        Get pending invitations that have been sent to an observer.

        Args:
            user_id_observer:
            headers:

        Returns:
            list of invitation json objects
        """
        try:
            invitations_url = self.invitations_url.format(**{"user_id": self._login_user_id})
            invitations_response = requests.get(invitations_url, headers=self._login_headers)
            invitations_response.raise_for_status()

            pending_invitations_json = invitations_response.json()
        except requests.HTTPError:
            pending_invitations_json = []

        return pending_invitations_json

    @_check_login
    def accept_observer_invitations(self):
        """
        Get pending invitations sent to an observer and accept them.

        Args:
            user_id_observer:
            headers:

        Returns:
            (list, list)
            pending
        """
        pending_invitations_json = self.get_pending_observer_invitations()

        total_invitations = len(pending_invitations_json)
        logger.info("Num pending invitations {}".format(total_invitations))

        invitation_accept_failed = []

        for i, invitation in enumerate(pending_invitations_json):

            try:
                share_key = invitation["key"]
                user_id = invitation["creatorId"]
                accept_url = self.accept_invitations_url.format(**{"observer_id": self._login_user_id, "user_id": user_id})

                accept_response = requests.put(accept_url, headers=self._login_headers, json={"key": share_key})
                accept_response.raise_for_status()

            except requests.HTTPError as e:
                invitation_accept_failed.append((e, invitation))

            if i % 20 == 0:
                num_failed = len(invitation_accept_failed)
                logger.info("Accepted {}. Failed {}. Out of {}".format(i - num_failed, num_failed, total_invitations))

        return pending_invitations_json, invitation_accept_failed

    @_check_http_error
    @_check_login
    def get_user_event_data(self, start_date, end_date, observed_user_id=None):
        """
        Get health event data for user_obj. TODO: Make more flexible

        Args:
            start_date (dt.datetime): Start date of data, inclusive
            end_date (dt.datetime): End date of data, inclusive of entire day
            observed_user_id (str): Optional id of observed user_obj if login id is clinician/study

        Returns:
            list: List of events as objects
        """
        user_id = self._login_user_id
        if observed_user_id:
            user_id = observed_user_id

        start_date_str, end_date_str = self.get_date_filter_string(start_date, end_date)

        user_data_base_url = self.user_data_url.format(**{"user_id": user_id})
        user_data_url = "{url_base}?startDate={start_date}&endDate={end_date}&dexcom=true&medtronic=true&carelink=true".format(**{
            "url_base": user_data_base_url,
            "end_date": end_date_str,
            "start_date": start_date_str,
        })

        data_response = requests.get(user_data_url, headers=self._login_headers)
        data_response.raise_for_status()
        user_event_data = data_response.json()

        return user_event_data

    @_check_http_error
    @_check_login
    def get_users_sharing_to(self):
        """
        Get a list of users the login id is sharing data to. The login id is typically
        a patient and the user_obj list is clinicians or studies.

        Returns:
            list: List of users as objects
        """

        user_metadata_url = self.users_sharing_to_url.format(**{
            "user_id": self._login_user_id
        })

        metadata_response = requests.get(user_metadata_url, headers=self._login_headers)
        metadata_response.raise_for_status()
        users_sharing_to = metadata_response.json()

        return users_sharing_to

    @_check_http_error
    @_check_login
    def get_users_sharing_with(self):
        """
        Get a list of users the login id is observing. The login id is typically the
        clinician or study and the user_obj list is patients.

        Returns:
            dict: List of users as objects
        """

        users_sharing_with_url = self.users_sharing_with_url.format(**{
            "user_id": self._login_user_id
        })
        users_sharing_with_response = requests.get(users_sharing_with_url, headers=self._login_headers)
        users_sharing_with_response.raise_for_status()
        users_sharing_with_json = users_sharing_with_response.json()

        return users_sharing_with_json

    @_check_http_error
    @_check_login
    def get_user_metadata(self, observed_user_id):

        user_metadata_url = self.users_sharing_to_url.format(**{
            "user_id": self._login_user_id
        })
        metadata_response = requests.get(user_metadata_url, headers=self._login_headers)

        metadata_response.raise_for_status()
        user_metadata = metadata_response.json()

        return user_metadata

    @_check_http_error
    @_check_login
    def get_notes(self, start_date, end_date, observed_user_id=None):
        """
        Get notes for a user_obj.
        """
        user_id = self._login_user_id
        if observed_user_id:
            user_id = observed_user_id

        start_date_str, end_date_str = self.get_date_filter_string(start_date, end_date)

        base_notes_url = self.user_notes_url.format(**{"user_id": user_id})
        notes_url = "{url_base}?startDate={start_date}&endDate={end_date}".format(
            **{
                "url_base": base_notes_url,
                "end_date": end_date_str,
                "start_date": start_date_str,
            })
        notes_response = requests.get(notes_url, headers=self._login_headers)

        try:
            notes_response.raise_for_status()
            notes_data = notes_response.json()
        except requests.HTTPError:
            notes_data = []

        return notes_data

    def get_date_filter_string(self, start_date, end_date):
        """
        Get string representations for date filters.

        Args:
            start_date dt.DateTime: start date
            end_date dt.Datetime: end date

        Returns:
            (str, str): start and end date strings
        """

        start_date_str = start_date.strftime(DATESTAMP_FORMAT) + "T00:00:00.000Z"
        end_date_str = end_date.strftime(DATESTAMP_FORMAT) + "T23:59:59.999Z"

        return start_date_str, end_date_str

    @_check_login
    def get_login_user_id(self):
        """
        Get the id of the user_obj logged in.
        Returns:
            str: user_obj id
        """
        return self._login_user_id



