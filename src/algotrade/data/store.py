"""Read access to the local parquet store, with a hold-out lock.

Layout: ``<root>/<market>/<interval>/<SYMBOL>.parquet`` where market is ``spot``
or ``um`` (USDT-margined perpetuals) and interval is ``1d``, ``1h``, ``1m`` or
``funding``. Bars are indexed by their UTC *open* time.

Hold-out discipline
-------------------
Everything from ``HOLDOUT_START`` onwards is reserved for a single final
evaluation. Loaders silently stop at the cut-off unless the process was started
with ``ALGOTRADE_ALLOW_HOLDOUT=1``. Strategy research must never set that flag.
"""

from __future__ import annotations

import os
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


def _as_utc(value) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _window(start, end) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Resolve [start, end) and apply the hold-out lock to ``end``."""
    start, end = _as_utc(start), _as_utc(end)
    if not holdout_allowed():
        end = HOLDOUT_START if end is None else min(end, HOLDOUT_START)
    return start, end


def _clip(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is not None:
        frame = frame[frame.index >= start]
    if end is not None:
        frame = frame[frame.index < end]
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
    start, end = _window(start, end)
    path = data_root() / market / interval / f"{symbol}.parquet"
    frame = pd.read_parquet(path, columns=list(columns) if columns else None)
    return _clip(frame, start, end)


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
    """
    start, end = _window(start, end)
    series: dict[str, pd.Series] = {}
    for name in sorted(symbols_):
        path = data_root() / "um" / "funding" / f"{name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["funding_rate"])
        rates = frame["funding_rate"]
        rates.index = rates.index.round("min")
        rates = rates[~rates.index.duplicated(keep="last")]
        rates = _clip(rates.to_frame(), start, end)["funding_rate"]
        if not rates.empty:
            series[name] = rates
    wide = pd.DataFrame(series).sort_index()
    wide.index.name = "calc_time"
    return wide


def resample_bars(frame: pd.DataFrame, rule: str, offset: str | None = None) -> pd.DataFrame:
    """Aggregate OHLCV bars to a coarser interval (labelled by open time).

    Only complete information is used: a coarse bar's close is the close of the
    last fine bar inside it. Buckets with no fine bars are dropped.
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
    out = frame.resample(rule, label="left", closed="left", offset=offset).agg(how)
    return out[np.isfinite(out["close"])] if "close" in out.columns else out.dropna(how="all")
