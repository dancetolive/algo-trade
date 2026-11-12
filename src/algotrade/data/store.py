"""Read access to the local parquet store, with a hold-out lock.

Layout: ``<root>/<market>/<interval>/<SYMBOL>.parquet`` where market is ``spot``
or ``um`` (USDT-margined perpetuals) and interval is ``1d``, ``1h``, ``1m`` or
``funding``. Bars are indexed by their UTC *open* time.

Instrument segments
-------------------
A ticker is not always one instrument: perpetuals are delisted and relisted
under the same name, coins are redenominated (1 old = 1000 new) and tickers are
reused for unrelated assets. The build step therefore cuts a symbol's history
wherever it stops for more than a week or resumes at a price more than five
times away. The latest segment keeps the plain name (``LUNAUSDT``); earlier ones
are stored as separate instruments (``LUNAUSDT__1``, ``LUNAUSDT__2``, ...), so no
rolling window, universe age or position ever spans such a break.

Hold-out discipline
-------------------
Everything from ``HOLDOUT_START`` onwards is reserved for a single final
evaluation. Loaders silently stop at the cut-off unless the process was started
with ``ALGOTRADE_ALLOW_HOLDOUT=1``. Strategy research must never set that flag.
A bar is returned only if it CLOSES by the cut-off.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DESIGN_END = pd.Timestamp("2024-01-01", tz="UTC")  # in-sample: data strictly before this
HOLDOUT_START = pd.Timestamp("2025-07-01", tz="UTC")  # validation: [DESIGN_END, HOLDOUT_START)

PANEL_FIELDS = ("open", "high", "low", "close", "quote_volume")


def data_root() -> Path:
    override = os.environ.get("ALGOTRADE_DATA")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "data" / "parquet"


def holdout_allowed() -> bool:
    return os.environ.get("ALGOTRADE_ALLOW_HOLDOUT") == "1"


_SEGMENT = re.compile(r"__\d+$")
_INTERVAL = re.compile(r"^(\d+)(m|h|d|w)$")


def base_symbol(name: str) -> str:
    """The exchange ticker of an instrument segment (``LUNAUSDT__1`` -> ``LUNAUSDT``)."""
    return _SEGMENT.sub("", name)


def bar_length(interval: str) -> pd.Timedelta:
    """Length of a bar of the given interval name (1m, 1h, 4h, 1d, 1w, ...)."""
    match = _INTERVAL.match(interval)
    if not match:
        raise ValueError(f"unsupported interval {interval!r}: bars must have a fixed length")
    unit = {"m": "min", "h": "h", "d": "D", "w": "W"}[match.group(2)]
    return pd.Timedelta(int(match.group(1)), unit=unit)


def _as_utc(value) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _window(start, end) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    """Resolve [start, end) and the hold-out lock (``None`` when the hold-out is open)."""
    start, end = _as_utc(start), _as_utc(end)
    lock = None if holdout_allowed() else HOLDOUT_START
    return start, end, lock


def _clip(frame: pd.DataFrame, start, end, lock, bar: pd.Timedelta) -> pd.DataFrame:
    """Bars that open in [start, end) and, under the lock, close by the cut-off."""
    if start is not None:
        frame = frame[frame.index >= start]
    if end is not None:
        frame = frame[frame.index < end]
    if lock is not None:
        frame = frame[frame.index + bar <= lock]
    return frame


def symbols(market: str, interval: str) -> list[str]:
    folder = data_root() / market / interval
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.parquet"))


def load_klines(
    market: str,
    interval: str,
    symbol: str,
    start=None,
    end=None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One symbol's bars in [start, end), indexed by UTC open time."""
    start, end, lock = _window(start, end)
    path = data_root() / market / interval / f"{symbol}.parquet"
    frame = pd.read_parquet(path, columns=list(columns) if columns else None)
    return _clip(frame, start, end, lock, bar_length(interval))


def load_panel(
    market: str,
    interval: str,
    symbols_: Iterable[str] | None = None,
    fields: Sequence[str] = PANEL_FIELDS,
    start=None,
    end=None,
) -> dict[str, pd.DataFrame]:
    """Wide frames (time x symbol), one per field, on the union of all bar times.

    Cells are NaN where a symbol had no bar (not yet listed, delisted, halted).
    """
    names = sorted(symbols_) if symbols_ is not None else symbols(market, interval)
    per_field: dict[str, dict[str, pd.Series]] = {field: {} for field in fields}
    for name in names:
        try:
            frame = load_klines(market, interval, name, start, end, columns=fields)
        except FileNotFoundError:
            continue
        if frame.empty:
            continue
        for field in fields:
            per_field[field][name] = frame[field]
    panel: dict[str, pd.DataFrame] = {}
    for field in fields:
        wide = pd.DataFrame(per_field[field])
        wide = wide.sort_index()
        wide.index.name = "open_time"
        panel[field] = wide
    return panel


def load_funding(symbols_: Iterable[str], start=None, end=None) -> pd.DataFrame:
    """Funding rates (time x symbol). Times are rounded to the minute.

    A positive rate means longs pay shorts ``rate * notional`` at that timestamp.
    A payment stamped exactly at ``end`` (or at the hold-out cut-off) is included:
    it is paid by the position carried through the last bar before that instant.
    Instrument segments (``XUSDT__1``) read the funding file of their ticker;
    symbols without a funding file are simply absent from the result.
    """
    start, end, lock = _window(start, end)
    limit = end if lock is None else (lock if end is None else min(end, lock))
    series: dict[str, pd.Series] = {}
    for name in sorted(symbols_):
        path = data_root() / "um" / "funding" / f"{base_symbol(name)}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["funding_rate"])
        rates = frame["funding_rate"]
        rates.index = rates.index.round("min")
        rates = rates[~rates.index.duplicated(keep="last")]
        if start is not None:
            rates = rates[rates.index >= start]
        if limit is not None:
            rates = rates[rates.index <= limit]
        if not rates.empty:
            series[name] = rates
    wide = pd.DataFrame(series).sort_index()
    wide.index.name = "calc_time"
    return wide


def resample_bars(frame: pd.DataFrame, rule: str, offset: str | None = None) -> pd.DataFrame:
    """Aggregate OHLCV bars to a coarser, fixed-length interval (labelled by open time).

    Buckets sit on a fixed grid counted from the Unix epoch, shifted by ``offset``
    (for example rule "1D" with offset "8h" gives days that run 08:00 to 08:00),
    so the same market always gets the same buckets whatever its first row is.
    A coarse bar's close is the close of the last fine bar inside it; buckets
    with no fine bars are dropped. The LAST bucket may be incomplete if the data
    ends inside it; remember that a coarse bar is known only at its close (see
    :func:`algotrade.backtest.decisions_on`).
    """
    how = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trades": "sum",
        "taker_buy_volume": "sum",
        "taker_buy_quote_volume": "sum",
    }
    how = {column: rule_ for column, rule_ in how.items() if column in frame.columns}
    out = frame.resample(pd.Timedelta(rule), label="left", closed="left", origin="epoch", offset=offset).agg(how)
    return out[np.isfinite(out["close"])] if "close" in out.columns else out.dropna(how="all")
