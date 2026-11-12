"""Auditor's own event-study primitives (written from the study's verbal description).

Nothing here imports the researcher's modules. Conventions re-stated:

* bars are indexed by UTC open time; an event is known at the CLOSE of bar t (bars <= t only);
* the fill is the OPEN of bar t+1; the h-minute forward return is open[t+1+h] / open[t+1] - 1;
* a forward return is dropped when its window t+1 .. t+1+h touches a missing bar or a halt
  (a run of >= 5 consecutive zero-volume bars), or when the entry or exit bar did not trade.

Rolling statistics are done with cumulative sums in numpy (not pandas.rolling) so that a quirk of one
implementation cannot be shared with the study's code.
"""

from __future__ import annotations

import os

assert os.environ.get("ALGOTRADE_ALLOW_HOLDOUT") != "1", "the hold-out must stay locked"

import numpy as np
import pandas as pd

from algotrade.data import store

MIN_NS = 60 * 10**9
DAY_NS = 1440 * MIN_NS
DESIGN_END = pd.Timestamp("2024-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-07-01", tz="UTC")
PERPS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
HALT_RUN = 5
SEGMENTS = {
    # name: (load start, load end, first event time, end of events)
    "design": (None, DESIGN_END, pd.Timestamp("2020-01-01", tz="UTC"), DESIGN_END),
    "validation": (pd.Timestamp("2023-12-25", tz="UTC"), HOLDOUT_START, DESIGN_END, HOLDOUT_START),
}


def taker_round_trip_bp(symbol: str) -> float:
    return 2.0 * (5.0 + (1.0 if symbol in ("BTCUSDT", "ETHUSDT") else 2.0))


def spot_round_trip_bp(symbol: str) -> float:
    return 2.0 * (10.0 + (1.0 if symbol in ("BTCUSDT", "ETHUSDT") else 2.0))


MAKER_ROUND_TRIP_BP = 4.0


class Bars:
    """One symbol on a complete 1-minute grid; NaN where the bar is missing."""

    def __init__(self, market: str, symbol: str, segment: str | None = None, start=None, end=None, frame: pd.DataFrame | None = None):
        if segment is not None:
            start, end, self.first, self.last = SEGMENTS[segment]
        else:
            self.first, self.last = None, None
        if frame is None:
            frame = store.load_klines(market, "1m", symbol, start=start, end=end)
        assert len(frame), (market, symbol)
        assert frame.index[-1] < HOLDOUT_START
        if segment == "design":
            assert frame.index[-1] < DESIGN_END
        self.market, self.symbol, self.segment = market, symbol, segment
        stamps = frame.index.as_unit("ns").asi8
        assert (np.diff(stamps) > 0).all() and (stamps % MIN_NS == 0).all()
        pos = (stamps - stamps[0]) // MIN_NS
        n = int(pos[-1]) + 1
        self.n = n
        self.ns = stamps[0] + np.arange(n, dtype=np.int64) * MIN_NS
        self.index = pd.DatetimeIndex(self.ns, tz="UTC")
        self.present = np.zeros(n, dtype=bool)
        self.present[pos] = True

        def col(name):
            out = np.full(n, np.nan)
            out[pos] = frame[name].to_numpy(dtype=float)
            return out

        self.open, self.high, self.low, self.close = col("open"), col("high"), col("low"), col("close")
        self.vol, self.qv, self.trades, self.tbqv = col("volume"), col("quote_volume"), col("trades"), col("taker_buy_quote_volume")
        self.day = self.ns // DAY_NS  # days since epoch
        self.minute_of_day = (self.ns % DAY_NS) // MIN_NS
        self.year = self.index.year.to_numpy()
        self.hour = self.minute_of_day // 60
        self.inside = np.ones(n, dtype=bool) if self.first is None else ((self.ns >= self.first.value) & (self.ns < self.last.value))

        # zero-volume runs: position inside the run (causal) and the total run length (not causal)
        zero = self.present & (self.qv <= 0)
        change = np.flatnonzero(np.diff(np.concatenate([[False], zero, [False]]).astype(np.int8)))
        starts, ends = change[0::2], change[1::2]  # run = [start, end)
        so_far = np.zeros(n, dtype=np.int64)
        total = np.zeros(n, dtype=np.int64)
        for a, b in zip(starts, ends):
            so_far[a:b] = np.arange(1, b - a + 1)
            total[a:b] = b - a
        self.zero = zero
        #: knowable at the bar's own close: the bar is missing, or the halt it sits in has already lasted HALT_RUN bars
        self.stale = zero & (so_far >= HALT_RUN)
        self.known_ok = self.present & ~self.stale
        #: uses the whole run (future information): only ever used to decide whether a forward return is valid
        self.halt = ~self.present | (zero & (total >= HALT_RUN))

        # prices as the signal sees them (stale bars blanked)
        self.c = np.where(self.known_ok, self.close, np.nan)
        self.o = np.where(self.known_ok, self.open, np.nan)
        self.h = np.where(self.known_ok, self.high, np.nan)
        self.l = np.where(self.known_ok, self.low, np.nan)
        self.q = np.where(self.known_ok, self.qv, np.nan)
        self.tr = np.where(self.known_ok, self.trades, np.nan)
        self.tb = np.where(self.known_ok, self.tbqv, np.nan)

    # ---- forward returns -------------------------------------------------------------
    def fwd(self, h: int, lag: int = 1, entry: str = "open", exit_: str = "open", halt_rule: bool = True) -> np.ndarray:
        """Return of entering at bar t+lag (field ``entry``) and leaving h bars later (field ``exit_``).

        Valid only if every bar t+1 .. t+lag+h is present (and, with ``halt_rule``, not in a halt) and the entry
        and exit bars traded.
        """
        assert lag >= 1
        n = self.n
        fields = {"open": self.open, "high": self.high, "low": self.low, "close": self.close,
                  "vwap": np.where(self.vol > 0, self.qv / np.where(self.vol > 0, self.vol, 1.0), np.nan)}
        bad = self.halt if halt_rule else ~self.present
        nbad = np.concatenate([[0], np.cumsum(bad)])  # number of bad bars in [0, i)
        out = np.full(n, np.nan)
        t = np.arange(0, n - lag - h)
        if len(t) == 0:
            return out
        e, x = t + lag, t + lag + h
        clean = (nbad[x + 1] - nbad[t + 1]) == 0
        traded = (self.qv[e] > 0) & (self.qv[x] > 0) if halt_rule else np.ones(len(t), dtype=bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = fields[exit_][x] / fields[entry][e] - 1.0
        out[t] = np.where(clean & traded, r, np.nan)
        return out


# ---- rolling helpers (cumulative sums, NaN-aware) -------------------------------------


def _csum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x)])


def roll_sum_count(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of the finite values among x[t-w+1 .. t]."""
    ok = np.isfinite(x)
    cs, cn = _csum(np.where(ok, x, 0.0)), _csum(ok.astype(float))
    i = np.arange(len(x))
    lo = np.maximum(i - w + 1, 0)
    return cs[i + 1] - cs[lo], cn[i + 1] - cn[lo]


def roll_std(x: np.ndarray, w: int, min_count: int) -> np.ndarray:
    """Sample standard deviation (ddof = 1) of the finite values among x[t-w+1 .. t]."""
    centre = np.nanmean(x)  # constant shift for numerical stability only (does not change a standard deviation)
    y = x - centre
    s1, c = roll_sum_count(y, w)
    s2, _ = roll_sum_count(y * y, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / c) / (c - 1.0)
    var[(c < min_count) | (c < 2)] = np.nan
    return np.sqrt(np.maximum(var, 0.0))


def roll_rms(x: np.ndarray, w: int, min_count: int) -> np.ndarray:
    s2, c = roll_sum_count(x * x, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(s2 / c)
    out[c < min_count] = np.nan
    return out


def lagged(x: np.ndarray, k: int) -> np.ndarray:
    """x[t-k] (NaN for the first k positions). k must be positive: never looks forward."""
    assert k >= 1
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


def thin(mask: np.ndarray, w: int) -> np.ndarray:
    """Greedy causal thinning: keep a flagged bar only if no KEPT bar lies in the previous w-1 bars."""
    out = np.zeros(len(mask), dtype=bool)
    last = -10**9
    for t in np.flatnonzero(mask):
        if t - last >= w:
            out[t] = True
            last = t
    return out


# ---- the three selected definitions ----------------------------------------------------


def ret_k(b: Bars, k: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return b.c / lagged(b.c, k) - 1.0


def flag_move5_abs2pct(b: Bars) -> tuple[np.ndarray, np.ndarray]:
    """(falls, rises): first bar of a |5-minute close-to-close return| >= 2% crossing (either direction)."""
    r5 = ret_k(b, 5)
    with np.errstate(invalid="ignore"):
        cross = b.known_ok & (np.abs(r5) >= 0.02)
    kept = thin(cross, 5)
    return kept & (r5 < 0), kept & (r5 > 0)


def z1(b: Bars) -> np.ndarray:
    r1 = ret_k(b, 1)
    sigma = roll_std(r1, 1440, 1296)  # the day of 1-minute returns ending at t
    scale = lagged(sigma, 1)  # ... ending at t-1: strictly before the bar being judged
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scale > 0, r1 / scale, np.nan)


def flag_move1_z3(b: Bars) -> tuple[np.ndarray, np.ndarray]:
    z = z1(b)
    with np.errstate(invalid="ignore"):
        return b.known_ok & (z < -3.0), b.known_ok & (z > 3.0)


def funding_series(symbol: str, segment: str | None) -> pd.Series:
    """Own read of the funding prints (stamp rounded to the minute), through the loader's hold-out window."""
    start, end = (None, HOLDOUT_START) if segment is None else (SEGMENTS[segment][0], SEGMENTS[segment][1])
    wide = store.load_funding([symbol], start=None if start is None else start - pd.Timedelta(days=2), end=end)
    return wide[symbol].dropna().sort_index()


def flag_fund_post5_big(b: Bars, funding: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(mask, side): the last bar before a 00:00/08:00/16:00 stamp when the last ALREADY PAID rate has |rate| >= 3 bp."""
    stamps = funding.index.as_unit("ns").asi8
    rates = funding.to_numpy(dtype=float)
    k = np.searchsorted(stamps, b.ns, side="right") - 1  # last print at or before the bar's open
    rate = np.where(k >= 0, rates[np.maximum(k, 0)], np.nan)
    at_stamp = np.isin(b.minute_of_day, [1439, 479, 959])
    with np.errstate(invalid="ignore"):
        mask = b.known_ok & at_stamp & (np.abs(rate) >= 3e-4)
    return mask, np.sign(np.nan_to_num(rate))


# ---- baseline ---------------------------------------------------------------------------


def vol_state(b: Bars) -> np.ndarray:
    r1 = ret_k(b, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = roll_rms(r1, 60, 54) / roll_rms(r1, 1440, 1296)
    state = np.full(b.n, -1, dtype=np.int64)
    ok = np.isfinite(ratio)
    state[ok] = (ratio[ok] >= 0.7).astype(np.int64) + (ratio[ok] >= 1.0).astype(np.int64)
    return state


def matched_baseline(b: Bars, f: np.ndarray, by_hour: bool = True) -> np.ndarray:
    """Mean forward return of the in-segment bars of the same year x (hour) x volatility state."""
    key = (b.year - b.year.min()) * 100 + (b.hour if by_hour else 0)
    key = key * 4 + (vol_state(b) + 1)
    ok = np.isfinite(f) & b.inside
    table = pd.Series(f[ok]).groupby(key[ok]).mean()
    return pd.Series(key).map(table).to_numpy()


# ---- inference ---------------------------------------------------------------------------


def cluster_stat(x: np.ndarray, cluster: np.ndarray) -> dict:
    """Mean of x and its cluster-robust standard error (clusters = any integer label, e.g. calendar day)."""
    ok = np.isfinite(x)
    x, cluster = np.asarray(x)[ok], np.asarray(cluster)[ok]
    n = len(x)
    if n == 0:
        return {"n": 0, "clusters": 0, "mean": np.nan, "se": np.nan, "t": np.nan}
    mean = x.mean()
    sums = pd.Series(x - mean).groupby(cluster).sum().to_numpy()
    g = len(sums)
    if g < 2:
        return {"n": n, "clusters": g, "mean": mean, "se": np.nan, "t": np.nan}
    se = np.sqrt((sums**2).sum() * g / (g - 1.0)) / n
    return {"n": n, "clusters": g, "mean": float(mean), "se": float(se), "t": float(mean / se) if se > 0 else float("nan")}


def naive_t(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def day_bootstrap(x: np.ndarray, day: np.ndarray, n_boot: int = 4000, block: int = 1, seed: int = 7, n_days_total: int | None = None, day0: int | None = None) -> dict:
    """Bootstrap of the event mean, resampling calendar days (or blocks of consecutive days) with replacement.

    Days are laid on the full calendar (``n_days_total`` consecutive days from ``day0``; default: first to last event
    day), so blocks are blocks of consecutive calendar days and empty days are resampled too.
    """
    ok = np.isfinite(x)
    x, day = np.asarray(x)[ok], np.asarray(day)[ok].astype(np.int64)
    d0 = int(day.min()) if day0 is None else int(day0)
    nd = int(day.max() - d0 + 1) if n_days_total is None else n_days_total
    assert day.min() >= d0 and day.max() < d0 + nd
    s = np.bincount(day - d0, weights=x, minlength=nd)
    c = np.bincount(day - d0, minlength=nd).astype(float)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(nd / block))
    starts = rng.integers(0, nd - block + 1, size=(n_boot, nb))
    picks = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :nd]
    num, den = s[picks].sum(axis=1), c[picks].sum(axis=1)
    means = num[den > 0] / den[den > 0]
    mean = x.mean()
    return {"mean": float(mean), "se": float(means.std(ddof=1)), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)),
            "p_le_0": float((means <= 0).mean())}
