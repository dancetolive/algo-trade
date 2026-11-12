"""Independent re-implementation for the audit of research/micro_1m/price_action.

Nothing here imports the researcher's modules. Everything is written from the
definitions in the brief and in REPORT.md:

* event detected at the CLOSE of bar t from bars <= t; earliest fill = OPEN of t+1;
  forward return over h minutes = open[t+1+h] / open[t+1] - 1;
* a return is dropped when any bar t .. t+1+h is missing; an event is dropped
  when any of the 1,500 bars before it is missing (researcher's rule);
* baseline = mean forward return of all bars of the same coin in the same
  calendar year x hour of day x 60-minute realised-volatility quintile.

The hold-out is never touched: the loader stops at 2025-07-01 on its own and
this module refuses to run if the unlock flag is set.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from algotrade.data import store

assert os.environ.get("ALGOTRADE_ALLOW_HOLDOUT") is None, "hold-out flag must not be set"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PERPS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
HS = (1, 5, 15, 60, 240, 1440)
CONTEXT = 1500
T_DESIGN_END = pd.Timestamp("2024-01-01", tz="UTC")
T_HOLDOUT = pd.Timestamp("2025-07-01", tz="UTC")


# ----------------------------------------------------------------------------- data
def load_grid(market: str, symbol: str, segment: str) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Bars of a segment on a complete 1-minute grid (missing minutes = NaN rows).

    design     : everything before 2024-01-01
    validation : 2023-12-25 .. 2025-06-30 loaded, events only from 2024-01-01 (one week of warm-up)
    all        : design + validation in one piece (for by-year work); still stops at the hold-out
    """
    if segment == "design":
        start, end, first_event = None, T_DESIGN_END, None
    elif segment == "validation":
        start, end, first_event = pd.Timestamp("2023-12-25", tz="UTC"), None, T_DESIGN_END
    elif segment == "all":
        start, end, first_event = None, None, None
    else:
        raise ValueError(segment)
    df = store.load_klines(market, "1m", symbol, start=start, end=end,
                           columns=["open", "high", "low", "close", "volume"])
    assert len(df) and df.index.max() < T_HOLDOUT
    assert df.index.is_monotonic_increasing and df.index.is_unique
    full = pd.date_range(df.index[0], df.index[-1], freq="1min")
    g = df.reindex(full)
    assert int(g["open"].notna().sum()) == int(df["open"].notna().sum()), "reindex lost rows"
    return g, first_event


def arrays(g: pd.DataFrame):
    o, h, l, c = (g[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    present = np.isfinite(o) & np.isfinite(c)
    return o, h, l, c, present


# ----------------------------------------------------------------------------- building blocks
def prior_extreme(x: np.ndarray, n: int, kind: str) -> np.ndarray:
    """Value at t = max (min) of x[t-n .. t-1]; NaN if the window is short or holds a NaN."""
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    w = sliding_window_view(x, n)
    ext = w.max(axis=1) if kind == "max" else w.min(axis=1)  # NaN-propagating
    out[n:] = ext[:-1]
    return out


def sweep_signal(h, l, c, n: int, x_bp: float, k: int) -> np.ndarray:
    """+1 = sweep of the prior n-bar LOW and close back above it (long), -1 = sweep of the HIGH (short).

    Breach bar s: trades >= x bp beyond the level (the previous close was not beyond it).
    Event: the first close back inside the level among bars s .. s+k-1. Written
    forward from the breach bar (scatter), unlike the researcher's backward shifts.
    """
    n_bars = len(c)
    ph, pl = prior_extreme(h, n, "max"), prior_extreme(l, n, "min")
    c_prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(invalid="ignore"):
        breach_hi = (h >= ph * (1.0 + x_bp * 1e-4)) & (c_prev <= ph)
        breach_lo = (l <= pl * (1.0 - x_bp * 1e-4)) & (c_prev >= pl)
    short_evt = np.zeros(n_bars, dtype=bool)
    long_evt = np.zeros(n_bars, dtype=bool)
    for is_high, breach, level, target in ((True, breach_hi, ph, short_evt), (False, breach_lo, pl, long_evt)):
        s = np.flatnonzero(breach)
        lev = level[s]
        waiting = np.ones(len(s), dtype=bool)
        for j in range(k):
            t = s + j
            inside_grid = t < n_bars
            cc = np.where(inside_grid, c[np.minimum(t, n_bars - 1)], np.nan)
            with np.errstate(invalid="ignore"):
                back_inside = (cc < lev) if is_high else (cc > lev)
                still_beyond = (cc >= lev) if is_high else (cc <= lev)
            hit = waiting & inside_grid & back_inside
            target[t[hit]] = True
            waiting = waiting & still_beyond & inside_grid
    return long_evt.astype(np.int8) - short_evt.astype(np.int8)  # a bar that is both cancels


def context_mask(present: np.ndarray, lookback: int = CONTEXT) -> np.ndarray:
    """True at t when bars t-lookback .. t all exist."""
    run = pd.Series(present.astype(np.float64)).rolling(lookback + 1, min_periods=lookback + 1).sum().to_numpy()
    return run == lookback + 1


def forward_return(o: np.ndarray, present: np.ndarray, h: int, delay: int = 1) -> np.ndarray:
    """bp return from the open of t+delay to the open of t+delay+h; NaN unless bars t .. t+delay+h all exist."""
    n = len(o)
    out = np.full(n, np.nan)
    m = n - delay - h
    if m <= 0:
        return out
    cs = np.concatenate([[0], np.cumsum(present)])
    t = np.arange(m)
    complete = (cs[t + delay + h + 1] - cs[t]) == (delay + h + 1)
    r = (o[delay + h: delay + h + m] / o[delay: delay + m] - 1.0) * 1e4
    out[:m] = np.where(complete, r, np.nan)
    return out


def realised_vol(c: np.ndarray, window: int = 60) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.diff(np.log(c), prepend=np.nan)
    return np.sqrt(pd.Series(r * r).rolling(window, min_periods=window).sum().to_numpy())


def baseline_key(index: pd.DatetimeIndex, rv: np.ndarray, first_event: pd.Timestamp | None, n_vol: int = 5) -> np.ndarray:
    """Cell key = year x hour x volatility quantile (edges inside each calendar year); -1 = undefined."""
    year = index.year.to_numpy()
    hour = index.hour.to_numpy()
    q = np.full(len(index), -1, dtype=np.int64)
    probs = np.linspace(0, 1, n_vol + 1)[1:-1]
    for y in np.unique(year):
        sel = (year == y) & np.isfinite(rv)
        if sel.sum() < n_vol * 100:
            continue
        edges = np.quantile(rv[sel], probs)
        q[sel] = np.digitize(rv[sel], edges)  # edges[i-1] <= x < edges[i]
    key = year.astype(np.int64) * 10000 + hour * 100 + q
    key[q < 0] = -1
    if first_event is not None:
        key[np.asarray(index < first_event)] = -1
    return key


def cell_mean(f: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Mean of f over all bars with the same key (finite f only); NaN where key < 0."""
    s = pd.Series(np.where(key >= 0, f, np.nan))
    m = s.groupby(key).transform("mean").to_numpy(copy=True)
    m[key < 0] = np.nan
    return m


# ----------------------------------------------------------------------------- inference
def cluster_stats(value: np.ndarray, cluster: np.ndarray, adjacent: bool = False) -> dict:
    """Event-weighted mean with a cluster-robust standard error.

    ``adjacent``: add the covariance of clusters whose ids differ by exactly 1
    (only if it increases the variance), as the researcher does for h >= 240.
    """
    value = np.asarray(value, dtype=float)
    keep = np.isfinite(value)
    value, cluster = value[keep], np.asarray(cluster)[keep]
    n = len(value)
    if n == 0:
        return dict(n=0, clusters=0, mean=np.nan, se=np.nan, t=np.nan, t_naive=np.nan)
    frame = pd.DataFrame({"v": value, "g": cluster}).groupby("g")["v"].agg(["sum", "count"]).sort_index()
    tot, cnt, ids = frame["sum"].to_numpy(), frame["count"].to_numpy(dtype=float), frame.index.to_numpy()
    groups = len(tot)
    mean = tot.sum() / n
    resid = tot - cnt * mean
    meat = float((resid ** 2).sum())
    if adjacent and groups > 1:
        nxt = np.flatnonzero(np.diff(ids) == 1)
        extra = 2.0 * float((resid[nxt] * resid[nxt + 1]).sum())
        meat = max(meat, meat + extra)
    se = np.sqrt(meat / n ** 2 * groups / (groups - 1.0)) if groups > 1 else np.nan
    sd = value.std(ddof=0)
    return dict(n=int(n), clusters=int(groups), mean=float(mean), se=float(se), t=float(mean / se) if se and se > 0 else np.nan,
                t_naive=float(mean / (sd / np.sqrt(n))) if sd > 0 else np.nan)


def day_block_bootstrap(value: np.ndarray, day: np.ndarray, block: int, n_boot: int = 4000, seed: int = 12345) -> dict:
    """Moving-block bootstrap over the calendar (blocks of `block` consecutive days) of the event-weighted mean."""
    value = np.asarray(value, dtype=float)
    keep = np.isfinite(value)
    value, day = value[keep], np.asarray(day)[keep]
    first, last = int(day.min()), int(day.max())
    span = last - first + 1
    s = np.bincount(day - first, weights=value, minlength=span)
    c = np.bincount(day - first, minlength=span).astype(float)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(span / block))
    starts = rng.integers(0, span - block + 1, size=(n_boot, n_blocks))
    pick = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :span]
    bs, bc = s[pick].sum(axis=1), c[pick].sum(axis=1)
    boot = bs[bc > 0] / bc[bc > 0]
    mean = s.sum() / c.sum()
    return dict(mean=float(mean), se=float(boot.std(ddof=1)), lo=float(np.quantile(boot, 0.025)), hi=float(np.quantile(boot, 0.975)),
                p_le_0=float(np.mean(boot <= 0)), t=float(mean / boot.std(ddof=1)))


# ----------------------------------------------------------------------------- costs (brief)
def slip_bp(symbol: str) -> float:
    return 1.0 if symbol in ("BTCUSDT", "ETHUSDT") else 2.0


def taker_round_trip_bp(symbol: str, market: str = "um") -> float:
    fee = 5.0 if market == "um" else 10.0
    return 2.0 * (fee + slip_bp(symbol))


MAKER_ROUND_TRIP_BP = 4.0  # 2 bp fee per side, no slippage: optimistic bound
