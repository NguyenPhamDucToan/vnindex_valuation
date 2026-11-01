"""One shared HTTP session for every source, with retries.

The VCI, VNDirect and SSI endpoints are the JSON APIs behind their public web
apps. None needs a key, but VCI answers a request without a browser-like
User-Agent/Referer with an empty body, and all three return the odd transient
5xx under load -- a retry seconds later succeeds -- so every call goes through
`get_json`/`post_json` here rather than a bare `requests.get`.
"""
from __future__ import annotations

import time

import requests

TIMEOUT_S = 30
ATTEMPTS = 3
_BACKOFF_S = (2.0, 6.0)          # waits before attempt 2 and 3
_RETRY_STATUS = {429, 500, 502, 503, 504}

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

# Origin/Referer per host. VCI checks them; the others merely tolerate them.
_SITE = {
    "vietcap.com.vn": "https://trading.vietcap.com.vn/",
    "vndirect.com.vn": "https://dstock.vndirect.com.vn/",
    "ssi.com.vn": "https://iboard.ssi.com.vn/",
}

_session: requests.Session | None = None


class SourceError(RuntimeError):
    """A source answered, but not with usable data."""


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": _UA, "Accept": "application/json, text/plain, */*",
                          "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8"})
        _session = s
    return _session


def _site_headers(url: str) -> dict:
    for host, site in _SITE.items():
        if host in url:
            return {"Referer": site, "Origin": site.rstrip("/")}
    return {}


def _request(method: str, url: str, **kw):
    headers = {**_site_headers(url), **kw.pop("headers", {})}
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(_BACKOFF_S[attempt - 1])
        try:
            r = _get_session().request(method, url, headers=headers, timeout=TIMEOUT_S, **kw)
        except requests.RequestException as e:
            last = e
            continue
        if r.status_code in _RETRY_STATUS:
            last = SourceError(f"{r.status_code} from {url}")
            continue
        if r.status_code != 200:
            raise SourceError(f"{r.status_code} from {url}")
        try:
            return r.json()
        except ValueError as e:
            last = SourceError(f"non-JSON reply from {url}: {e}")
    raise last if last else SourceError(f"no reply from {url}")


def get_json(url: str, params: dict | None = None, **kw):
    return _request("GET", url, params=params, **kw)


def post_json(url: str, payload: dict, **kw):
    return _request("POST", url, json=payload, **kw)
