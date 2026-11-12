"""Point-in-time tradable universe.

A universe must be decided with information available at the time, and must
include coins that were later delisted; otherwise every cross-sectional result
is biased upwards (survivorship bias).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

QUOTE = "USDT"

# Stablecoins, fiat and other assets pegged to a currency: no directional edge to trade.
STABLE_OR_FIAT = {
    "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "PAX", "UST", "USTC", "SUSD", "USDS", "USDSB",
    "PYUSD", "EURI", "AEUR", "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "UAH", "NGN", "BIDR", "IDRT",
    "BKRW", "VAI", "XUSD", "USD1", "BFUSD", "USDE", "RLUSD", "EURC", "USDD", "GUSD", "LUSD", "RWUSD", "KGST",
    "BVND", "ARS", "PLN", "RON", "ZAR", "JPY", "MXN", "CZK", "COP", "U",
}

# Tokens that track another asset; trade the underlying instead.
PEGGED = {"WBTC", "WBETH", "BETH", "BNSOL", "STETH", "WSTETH", "CBBTC", "TBTC", "RENBTC", "HBTC"}

# Tokenised gold is handled as its own asset class, not as a member of the coin universe.
GOLD = {"PAXG", "XAUT", "XAU"}

# FRAX is deliberately absent from the stable list: the only FRAXUSDT market is the
# renamed FXS token (from January 2026), which is volatile.


@lru_cache(maxsize=1)
def _listed_kinds() -> dict[str, dict[str, frozenset[str]]]:
    """Instruments that are not crypto at all, by market (see scripts/classify_instruments.py)."""
    path = Path(__file__).resolve().parent / "data" / "instrument_kinds.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {kind: {market: frozenset(names) for market, names in raw[kind].items()} for kind in ("tradfi", "index") if kind in raw}

_MULTIPLIER = re.compile(r"^(1000000|1000|1M)(?=[A-Z0-9])")


_SEGMENT = re.compile(r"__\d+$")


def base_asset(symbol: str) -> str:
    """Base asset of a USDT symbol, without the futures size prefix (1000PEPE -> PEPE).

    Earlier instrument segments of a ticker (``LUNAUSDT__1``) share its base asset.
    """
    symbol = _SEGMENT.sub("", symbol)
    if not symbol.endswith(QUOTE):
        raise ValueError(f"not a {QUOTE} symbol: {symbol}")
    base = symbol[: -len(QUOTE)]
    return _MULTIPLIER.sub("", base) or base


def is_leveraged_token(base: str, all_bases: set[str]) -> bool:
    """Binance leveraged tokens: <COIN>UP / <COIN>DOWN / <COIN>BULL / <COIN>BEAR."""
    for suffix in ("DOWN", "BULL", "BEAR", "UP"):
        if base.endswith(suffix):
            stem = base[: -len(suffix)]
            if stem == "" and suffix in ("BULL", "BEAR"):
                return True
            if stem in all_bases and stem != "":
                return True
    return False


def classify(symbols: list[str], market: str | None = None) -> pd.DataFrame:
    """One row per symbol with its base asset and category.

    Categories: ``coin`` (the only one that belongs in a crypto universe),
    ``stable``, ``gold``, ``pegged``, ``leveraged``, ``tradfi`` (perpetuals and
    tokens on stocks, ETFs, commodities and currencies) and ``index``. The last
    two come from a list keyed by market; pass ``market`` ("spot" or "um") to
    apply only that market's names, or leave it out to apply both.

    A ticker that was reused keeps its earlier life as a separate instrument
    (``MDTUSDT__1``); only the plain, latest name is matched against the list.
    """
    listed = _listed_kinds()
    markets = [market] if market else ["spot", "um"]
    special = {
        kind: frozenset().union(*(listed.get(kind, {}).get(m, frozenset()) for m in markets)) for kind in ("tradfi", "index")
    }
    bases = {s: base_asset(s) for s in symbols}
    all_bases = set(bases.values())
    rows = []
    for symbol, base in bases.items():
        if symbol in special["tradfi"]:
            kind = "tradfi"
        elif symbol in special["index"] or _SEGMENT.sub("", symbol) in special["index"]:
            kind = "index"
        elif base in STABLE_OR_FIAT:
            kind = "stable"
        elif base in GOLD:
            kind = "gold"
        elif base in PEGGED:
            kind = "pegged"
        elif is_leveraged_token(base, all_bases):
            kind = "leveraged"
        else:
            kind = "coin"
        rows.append({"symbol": symbol, "base": base, "kind": kind})
    return pd.DataFrame(rows).set_index("symbol")


def coin_symbols(symbols: list[str], market: str | None = None) -> list[str]:
    table = classify(symbols, market)
    return sorted(table.index[table["kind"] == "coin"])


def liquidity_score(quote_volume: pd.DataFrame, lookback: int = 30, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing median traded value per bar. Row t uses bars up to and including t."""
    if min_periods is None:
        min_periods = lookback
    return quote_volume.rolling(lookback, min_periods=min_periods).median()


def top_liquid(
    quote_volume: pd.DataFrame,
    top_n: int,
    lookback: int = 30,
    min_history: int = 60,
    exit_rank: int | None = None,
) -> pd.DataFrame:
    """Boolean membership (time x symbol) of the ``top_n`` most liquid symbols.

    * Row t is decided from bars up to and including t, so it may be used for a
      decision taken at the close of bar t.
    * A symbol needs ``min_history`` bars since its first bar before it can enter
      (new listings are chaotic and often untradeable in size).
    * Hysteresis: a symbol enters when its rank is within ``top_n`` and stays
      until its rank falls below ``exit_rank`` (default 1.25 x top_n), which
      avoids churning at the boundary. The universe therefore holds between
      ``top_n`` and ``exit_rank`` symbols.
    * A symbol with no bar at t (delisted or halted) is never a member at t.
    """
    if exit_rank is None:
        exit_rank = int(np.ceil(top_n * 1.25))
    score = liquidity_score(quote_volume, lookback)
    age = quote_volume.notna().cumsum()
    eligible = (age >= min_history) & quote_volume.notna() & score.notna()
    score = score.where(eligible)
    rank = score.rank(axis=1, ascending=False, method="first")

    rank_values = rank.to_numpy()
    member = np.zeros(rank_values.shape, dtype=bool)
    previous = np.zeros(rank_values.shape[1], dtype=bool)
    for i in range(rank_values.shape[0]):
        row = rank_values[i]
        finite = np.isfinite(row)
        enter = finite & (row <= top_n)
        stay = previous & finite & (row <= exit_rank)
        current = enter | stay
        member[i] = current
        previous = current
    return pd.DataFrame(member, index=quote_volume.index, columns=quote_volume.columns)
