"""Building blocks for strategies. Every function is strictly backward-looking:
row t of an output depends only on rows <= t of its inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()


def ewm_volatility(close: pd.DataFrame, halflife: float, periods: float, min_periods: int | None = None) -> pd.DataFrame:
    """Annualised EWMA volatility of log returns."""
    if min_periods is None:
        min_periods = int(max(5, halflife))
    variance = (log_returns(close) ** 2).ewm(halflife=halflife, min_periods=min_periods).mean()
    return np.sqrt(variance * periods)


def past_return(close: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Return from t - lookback - skip to t - skip (skip the most recent bars).

    Rows are bars of the panel's index: a symbol without a bar on either end of
    the window gets NaN.
    """
    if lookback < 1 or skip < 0:
        raise ValueError("lookback must be >= 1 and skip >= 0 (a negative skip would read the future)")
    return close.shift(skip) / close.shift(skip + lookback) - 1.0


def ema_trend(close: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """+1 when the fast EMA is above the slow EMA, -1 when below, NaN until warmed up."""
    fast_ema = close.ewm(span=fast, min_periods=slow).mean()
    slow_ema = close.ewm(span=slow, min_periods=slow).mean()
    return np.sign(fast_ema - slow_ema).where(fast_ema.notna() & slow_ema.notna())


def breakout_state(close: pd.DataFrame, entry: int, exit_: int | None = None) -> pd.DataFrame:
    """Donchian-style state machine on closes.

    Goes long (+1) when the close exceeds the highest close of the previous
    ``entry`` bars, short (-1) when below the lowest; returns to flat (0) when the
    close crosses the opposite ``exit_``-bar extreme (default entry // 2).
    """
    if exit_ is None:
        exit_ = max(1, entry // 2)
    prior = close.shift(1)
    upper, lower = prior.rolling(entry).max(), prior.rolling(entry).min()
    exit_low, exit_high = prior.rolling(exit_).min(), prior.rolling(exit_).max()
    c = close.to_numpy(dtype=float)
    up, lo = upper.to_numpy(dtype=float), lower.to_numpy(dtype=float)
    xl, xh = exit_low.to_numpy(dtype=float), exit_high.to_numpy(dtype=float)
    state = np.zeros(c.shape)
    current = np.zeros(c.shape[1])
    for i in range(c.shape[0]):
        row = c[i]
        valid = np.isfinite(row) & np.isfinite(up[i]) & np.isfinite(lo[i])
        current = np.where(~np.isfinite(row), 0.0, current)
        go_long = valid & (row > up[i])
        go_short = valid & (row < lo[i])
        exit_long = (current > 0) & np.isfinite(xl[i]) & (row < xl[i])
        exit_short = (current < 0) & np.isfinite(xh[i]) & (row > xh[i])
        current = np.where(exit_long | exit_short, 0.0, current)
        current = np.where(go_long, 1.0, current)
        current = np.where(go_short, -1.0, current)
        state[i] = current
    return pd.DataFrame(state, index=close.index, columns=close.columns)


def volatility_target(
    signal: pd.DataFrame,
    volatility: pd.DataFrame,
    target_vol: float,
    max_weight: float,
) -> pd.DataFrame:
    """Scale each position so it contributes ``target_vol`` of annualised risk."""
    raw = signal * (target_vol / volatility.where(volatility > 0))
    return raw.clip(lower=-max_weight, upper=max_weight)


def cross_sectional_rank(score: pd.DataFrame, member: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank in (0, 1] across the row (1 = highest score), among members only."""
    if member is not None:
        score = score.where(member)
    return score.rank(axis=1, pct=True)
