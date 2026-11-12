"""Cross-sectional relative strength on Binance spot, long only, daily bars.

Rule
----
At the close of a rebalance bar, rank the point-in-time liquid coins by their
past ``lookback``-day return (optionally skipping the most recent day), hold the
top ``top_k`` (momentum) or the bottom ``top_k`` (reversal), equal or
inverse-volatility weighted, fully invested. Optionally hold cash (USDT) instead
while BTC closes below its ``market_filter``-day simple average. The decision is
filled by the simulator at the next bar's open.

Inputs
------
``panel`` is a dict of frames on one daily bar index (UTC bar-open times):

* ``close``                         closes, NaN where a market has no bar;
* ``member30`` / ``member50`` ...   point-in-time universe membership from
                                    ``research.liquid_membership`` (True/1 = member
                                    at the close of that bar), on the same index
                                    and columns as ``close``.

Every row t of the output depends only on rows <= t of the inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algotrade import signals

# A member becomes a ranking candidate once it has this many daily bars: the longest
# lookback (90) + the skipped day (1) + the bar itself. Every lookback therefore ranks
# exactly the same candidates, and the first chaotic weeks of a listing are left out.
HISTORY_BARS = 92
MIN_NAMES = 10  # hold nothing while the universe is thinner than this
VOL_HALFLIFE = 20  # days, for inverse-volatility weights (fixed, not tuned)
BTC = "BTCUSDT"


def candidates(panel, universe: int | str = 30, exclude=()) -> pd.DataFrame:
    """Boolean frame: coins that may be ranked at the close of each bar."""
    close = panel["close"]
    member = panel[f"member{universe}"].reindex(index=close.index, columns=close.columns)
    member = member.fillna(0).astype(float) > 0.5
    seasoned = close.shift(HISTORY_BARS - 1).notna()
    out = member & seasoned & close.notna()
    for name in exclude:
        if name in out.columns:
            out[name] = False
    return out


def market_on(panel, market_filter: int) -> pd.Series:
    """True where BTC's close is above its ``market_filter``-day simple average (always True for 0)."""
    close = panel["close"]
    if not market_filter:
        return pd.Series(True, index=close.index)
    btc = close[BTC]
    average = btc.rolling(int(market_filter)).mean()
    return (btc > average) & average.notna()


def rebalance_rows(index: pd.DatetimeIndex, schedule: str = "weekly", weekday: int = 6) -> np.ndarray:
    """Rows on which a decision is taken. Weekly: bars of ``weekday`` (Monday = 0 ... Sunday = 6).

    The Sunday bar closes at Monday 00:00 UTC, so weekday 6 trades at Monday's open.
    """
    if schedule == "daily":
        return np.ones(len(index), dtype=bool)
    if schedule == "weekly":
        return np.asarray(index.weekday == weekday)
    raise ValueError(f"unknown schedule {schedule!r}")


def build_weights(
    panel,
    universe: int | str = 30,
    lookback: int = 28,
    skip: int = 0,
    top_k: int | None = 5,
    weighting: str = "equal",
    direction: str = "momentum",
    schedule: str = "weekly",
    weekday: int = 6,
    market_filter: int = 0,
    exclude=(),
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """Target weights (bars x symbols). Rows without a decision are entirely NaN.

    ``top_k=None`` holds every candidate (the equal-weight basket benchmark).
    """
    close = panel["close"]
    eligible = candidates(panel, universe, exclude)

    if top_k is None:
        selected = eligible
    else:
        score = signals.past_return(close, int(lookback), int(skip)).where(eligible)
        if direction == "momentum":
            order = score.rank(axis=1, ascending=False, method="first")
        elif direction == "reversal":
            order = score.rank(axis=1, ascending=True, method="first")
        else:
            raise ValueError(f"unknown direction {direction!r}")
        selected = order <= int(top_k)  # NaN ranks compare False

    count = selected.sum(axis=1)
    if weighting == "equal":
        slots = count if top_k is None else pd.Series(float(top_k), index=close.index)
        weights = selected.astype(float).div(slots.where(slots > 0), axis=0)
    elif weighting == "inverse_vol":
        vol = signals.ewm_volatility(close, VOL_HALFLIFE, 365.25)
        inverse = (1.0 / vol.where(vol > 0)).where(selected)
        # a selected coin without a volatility estimate gets the average of the others
        inverse = inverse.where(~(selected & inverse.isna()), inverse.mean(axis=1), axis=0)
        total = inverse.sum(axis=1)
        weights = inverse.div(total.where(total > 0), axis=0)
        if top_k is not None:  # fewer than K names selected: leave the missing slots in cash
            weights = weights.mul((count / float(top_k)).clip(upper=1.0), axis=0)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    weights = weights.fillna(0.0)

    enough = eligible.sum(axis=1) >= int(min_names)
    invested = enough & market_on(panel, market_filter)
    weights = weights.where(invested, 0.0, axis=0)

    decide = rebalance_rows(close.index, schedule, weekday)
    weights = weights.where(pd.Series(decide, index=close.index), np.nan, axis=0)
    return weights


def survivor_membership(panel, universe: int | str, as_of: pd.Timestamp) -> pd.DataFrame:
    """DELIBERATELY BIASED universe, for the survivorship warning only.

    The members of the point-in-time universe on the last bar at or before ``as_of`` are treated as
    members on every earlier day on which they have a bar. This is the classic
    mistake of backtesting today's liquid coins over their own history.
    """
    close = panel["close"]
    member = panel[f"member{universe}"].reindex(index=close.index, columns=close.columns).fillna(0).astype(float) > 0.5
    last = member.loc[:as_of].iloc[-1]
    names = list(last.index[last])
    out = pd.DataFrame(False, index=close.index, columns=close.columns)
    out[names] = close[names].notna()
    return out
