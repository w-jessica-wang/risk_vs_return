# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""HTTP helpers: one pooled, retrying requests.Session per thread."""
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_local = threading.local()
TIMEOUT = (10, 30)  # (connect, read) seconds
ATTEMPTS = 5


def _session():
    session = getattr(_local, "session", None)
    if session is None:
        retry = Retry(total=6, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        _local.session = session
    return session


def _get(url, params=None, timeout=TIMEOUT):
    for attempt in range(ATTEMPTS):
        try:
            response = _session().get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout):
            # The adapter's Retry covers failed connects and 5xx responses, but not a
            # response body that stalls mid-read, which surfaces here as ConnectionError.
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(min(30, 2 ** attempt))


def get_text(url, params=None, timeout=TIMEOUT):
    return _get(url, params, timeout).text


def get_bytes(url, timeout=TIMEOUT):
    return _get(url, None, timeout).content
