# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Network tests — a response that stalls mid-read (seen live from the S3 listing endpoint) is
retried, and a persistent failure still surfaces after the last attempt. No network.

Run:  pytest tests/ -q
"""
import pytest
import requests

from rvr import net


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.content = text.encode()

    def raise_for_status(self):
        pass


class FlakySession:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.ConnectionError("Read timed out.")
        return FakeResponse("ok")


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(net.time, "sleep", lambda seconds: None)


def test_mid_read_timeouts_are_retried(monkeypatch, no_sleep):
    session = FlakySession(failures=2)
    monkeypatch.setattr(net, "_session", lambda: session)
    assert net.get_text("https://example.test/listing") == "ok"
    assert session.calls == 3


def test_persistent_failures_surface_after_the_last_attempt(monkeypatch, no_sleep):
    session = FlakySession(failures=99)
    monkeypatch.setattr(net, "_session", lambda: session)
    with pytest.raises(requests.ConnectionError):
        net.get_bytes("https://example.test/archive.zip")
    assert session.calls == net.ATTEMPTS
