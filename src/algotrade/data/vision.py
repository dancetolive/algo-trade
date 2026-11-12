"""Client for the public Binance bulk-data archive (data.binance.vision).

The archive is an S3 bucket fronted by a CDN. File listings come from the S3
REST endpoint; the files themselves are fetched from the CDN host. Everything
downloaded is treated as untrusted input: sizes and MD5 digests are checked
against the listing, archives are size-capped before extraction, and the CSV
payload is parsed with explicit column names and numeric dtypes.
"""

from __future__ import annotations

import hashlib
import html
import io
import re
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILE_URL = "https://data.binance.vision"

MARKET_ROOT = {
    "spot": "data/spot/monthly",
    "um": "data/futures/um/monthly",
}

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
KLINE_KEEP = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

# Refuse to extract anything implausibly large (a month of 1m klines is ~5 MB).
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024

_local = threading.local()


def _session() -> requests.Session:
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        session.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=2))
        session.headers["User-Agent"] = "market-data-research/0.1"
        _local.session = session
    return session


class NotFound(Exception):
    """The requested archive object does not exist."""


def _get(url: str, *, params: dict | None = None, timeout: float = 90, retries: int = 6) -> requests.Response:
    delay = 1.0
    last: Exception | None = None
    for _ in range(retries):
        try:
            response = _session().get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last = exc
        else:
            if response.status_code == 200:
                return response
            if response.status_code == 404:
                raise NotFound(url)
            last = RuntimeError(f"HTTP {response.status_code}")
        time.sleep(delay)
        delay = min(delay * 2, 30.0)
    raise RuntimeError(f"giving up on {url}: {last}")


@dataclass(frozen=True)
class RemoteFile:
    key: str
    size: int
    etag: str  # hex MD5 of the object unless it was a multipart upload ("...-N")

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]


_COMMON_RE = re.compile(r"<CommonPrefixes><Prefix>([^<]*)</Prefix></CommonPrefixes>")
_CONTENTS_RE = re.compile(r"<Contents>(.*?)</Contents>", re.S)
_KEY_RE = re.compile(r"<Key>([^<]*)</Key>")
_SIZE_RE = re.compile(r"<Size>(\d+)</Size>")
_ETAG_RE = re.compile(r"<ETag>([^<]*)</ETag>")
_NEXT_RE = re.compile(r"<NextMarker>([^<]*)</NextMarker>")


def _list_pages(prefix: str) -> Iterator[str]:
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        text = _get(LIST_URL, params=params).text
        yield text
        if "<IsTruncated>true</IsTruncated>" not in text:
            return
        match = _NEXT_RE.search(text)
        if match:
            marker = html.unescape(match.group(1))
            continue
        keys = _KEY_RE.findall(text)
        if not keys:
            return
        marker = html.unescape(keys[-1])


def list_dirs(prefix: str) -> list[str]:
    """Names of the sub-directories directly under ``prefix``."""
    if not prefix.endswith("/"):
        prefix += "/"
    names: list[str] = []
    for page in _list_pages(prefix):
        for raw in _COMMON_RE.findall(page):
            full = html.unescape(raw)
            names.append(full[len(prefix) :].strip("/"))
    return names


def list_files(prefix: str, suffix: str = ".zip") -> list[RemoteFile]:
    """Objects directly under ``prefix`` whose key ends with ``suffix``."""
    if not prefix.endswith("/"):
        prefix += "/"
    files: list[RemoteFile] = []
    for page in _list_pages(prefix):
        for block in _CONTENTS_RE.findall(page):
            key_m, size_m, etag_m = _KEY_RE.search(block), _SIZE_RE.search(block), _ETAG_RE.search(block)
            if not (key_m and size_m):
                continue
            key = html.unescape(key_m.group(1))
            if not key.endswith(suffix):
                continue
            etag = html.unescape(etag_m.group(1)).strip('"') if etag_m else ""
            files.append(RemoteFile(key=key, size=int(size_m.group(1)), etag=etag))
    return files


def klines_prefix(market: str, symbol: str, interval: str) -> str:
    return f"{MARKET_ROOT[market]}/klines/{symbol}/{interval}/"


def funding_prefix(symbol: str) -> str:
    return f"{MARKET_ROOT['um']}/fundingRate/{symbol}/"


def list_symbols(market: str) -> list[str]:
    return list_dirs(f"{MARKET_ROOT[market]}/klines/")


_SAFE_KEY = re.compile(r"^data/[A-Za-z0-9_./-]+$")


def download(remote: RemoteFile, raw_root: Path, attempts: int = 3) -> Path:
    """Fetch one archive object into ``raw_root`` (mirroring the bucket layout)."""
    if not _SAFE_KEY.match(remote.key) or ".." in remote.key:
        raise ValueError(f"unexpected object key: {remote.key!r}")
    dest = raw_root / remote.key
    if dest.exists() and dest.stat().st_size == remote.size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    problem = ""
    for _ in range(attempts):
        content = _get(f"{FILE_URL}/{remote.key}").content
        if len(content) != remote.size:
            problem = f"size {len(content)} != listed {remote.size}"
        elif remote.etag and "-" not in remote.etag and hashlib.md5(content).hexdigest() != remote.etag:
            problem = "MD5 does not match the listing"
        else:
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(content)
            tmp.replace(dest)
            return dest
        time.sleep(2.0)
    raise IOError(f"{remote.key}: {problem}")


def _read_single_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected one member, found {len(members)}")
        if members[0].file_size > MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"{path.name}: member is implausibly large")
        return archive.read(members[0])


def _has_header(raw: bytes) -> bool:
    head = raw[:32].lstrip()
    return bool(head) and not head[:1].isdigit()


def _epoch_to_utc(values: pd.Series) -> pd.Series:
    """Binance used milliseconds until 2025 and microseconds for spot afterwards."""
    values = values.astype("int64")
    micro = values > 10**14
    millis = values.where(~micro, values // 1000)
    return pd.to_datetime(millis, unit="ms", utc=True)


def read_kline_zip(path: Path) -> pd.DataFrame:
    raw = _read_single_member(path)
    frame = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        names=KLINE_COLUMNS,
        skiprows=1 if _has_header(raw) else 0,
        dtype={
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "close_time": "int64",
            "quote_volume": "float64",
            "trades": "int64",
            "taker_buy_volume": "float64",
            "taker_buy_quote_volume": "float64",
        },
    )
    frame["open_time"] = _epoch_to_utc(frame["open_time"])
    return frame[["open_time", *KLINE_KEEP]]


def build_klines(paths: list[Path]) -> pd.DataFrame:
    """Concatenate monthly kline archives into one frame indexed by UTC open time."""
    if not paths:
        return pd.DataFrame(columns=KLINE_KEEP, index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    frames = [read_kline_zip(p) for p in sorted(paths)]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates("open_time", keep="last").sort_values("open_time")
    return frame.set_index("open_time")


def read_funding_zip(path: Path) -> pd.DataFrame:
    raw = _read_single_member(path)
    frame = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        names=["calc_time", "funding_interval_hours", "funding_rate"],
        skiprows=1 if _has_header(raw) else 0,
        dtype={"calc_time": "int64", "funding_interval_hours": "int64", "funding_rate": "float64"},
    )
    frame["calc_time"] = _epoch_to_utc(frame["calc_time"])
    return frame


def build_funding(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(
            columns=["funding_interval_hours", "funding_rate"],
            index=pd.DatetimeIndex([], tz="UTC", name="calc_time"),
        )
    frame = pd.concat([read_funding_zip(p) for p in sorted(paths)], ignore_index=True)
    frame = frame.drop_duplicates("calc_time", keep="last").sort_values("calc_time")
    return frame.set_index("calc_time")
