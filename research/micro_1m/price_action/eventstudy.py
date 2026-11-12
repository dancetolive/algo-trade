"""Event-study machinery for structures detected on 1-minute bars.

Conventions (all enforced here, all covered by test_eventstudy.py)
------------------------------------------------------------------
* Bars live on a COMPLETE 1-minute grid. Minutes for which the exchange
  delivered no bar are NaN rows, so a missing bar can never be skipped over
  silently by a shift or a rolling window.
* An event is detected at the CLOSE of bar ``t`` from bars ``<= t``.
  The earliest fill is the OPEN of bar ``t + 1``.
* The forward return over ``h`` minutes runs from the open of ``t + 1`` to the
  open of ``t + 1 + h``. It is NaN unless every bar ``t .. t + 1 + h`` exists.
* Returns are in basis points and are signed in the direction of the
  hypothesis (+1 = long, -1 = short).
* Inference is clustered by UTC calendar day across all symbols: the sufficient
  statistics are per-day sums, which is what :func:`aggregate_daily` stores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 15, 60, 240, 1440)
MINUTES_PER_DAY = 1440
STAT_FIELDS = ("n", "sg", "sb", "sw", "sgg", "see")


# --------------------------------------------------------------------------- #
# Costs (basis points)
# --------------------------------------------------------------------------- #

def slippage_bp(symbol: str) -> float:
    return 1.0 if symbol in ("BTCUSDT", "ETHUSDT") else 2.0


def round_trip_cost_bp(symbol: str, venue: str) -> float:
    """Cost of entering and leaving once.

    perp_taker : 5 bp fee + slippage per side.
    perp_maker : 2 bp fee per side and NO slippage. Optimistic bound: it assumes
                 the resting order is filled at the same price as the taker
                 would have got, i.e. it ignores non-fills and adverse selection.
    spot_taker : 10 bp fee + slippage per side.
    """
    slip = slippage_bp(symbol)
    if venue == "perp_taker":
        return 2.0 * (5.0 + slip)
    if venue == "perp_maker":
        return 2.0 * 2.0
    if venue == "spot_taker":
        return 2.0 * (10.0 + slip)
    raise ValueError(venue)


# --------------------------------------------------------------------------- #
# Bars on a complete grid
# --------------------------------------------------------------------------- #

@dataclass
class Bars:
    symbol: str
    market: str
    minute: np.ndarray  # int64 minutes since the Unix epoch (bar OPEN time, UTC), consecutive
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    valid: np.ndarray  # False where the exchange delivered no bar

    @property
    def n(self) -> int:
        return len(self.minute)

    @property
    def day(self) -> np.ndarray:
        return self.minute // MINUTES_PER_DAY

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.minute * 60, unit="s", utc=True)

    def cut(self, stop: int) -> "Bars":
        """The first ``stop`` bars (for truncation tests)."""
        return Bars(self.symbol, self.market, self.minute[:stop], self.o[:stop], self.h[:stop], self.l[:stop],
                    self.c[:stop], self.v[:stop], self.valid[:stop])


def bars_from_frame(frame: pd.DataFrame, symbol: str = "", market: str = "") -> Bars:
    """Put a kline frame (UTC open-time index) on a complete 1-minute grid."""
    if frame.empty:
        raise ValueError("no bars")
    index = frame.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    minute = (index.as_unit("s").asi8 // 60).astype(np.int64)
    if not (np.diff(minute) > 0).all():
        raise ValueError("bar times must be strictly increasing")
    first, last = int(minute[0]), int(minute[-1])
    n = last - first + 1
    pos = minute - first
    out = {}
    for name in ("open", "high", "low", "close", "volume"):
        full = np.full(n, np.nan)
        full[pos] = frame[name].to_numpy(dtype=float)
        out[name] = full
    valid = np.zeros(n, dtype=bool)
    valid[pos] = True
    valid &= np.isfinite(out["open"]) & np.isfinite(out["close"])
    return Bars(symbol, market, first + np.arange(n, dtype=np.int64), out["open"], out["high"], out["low"],
                out["close"], out["volume"], valid)


def load_bars(market: str, symbol: str, start=None, end=None) -> Bars:
    from algotrade.data import store

    frame = store.load_klines(market, "1m", symbol, start=start, end=end,
                              columns=["open", "high", "low", "close", "volume"])
    return bars_from_frame(frame, symbol, market)


def _missing_before(valid: np.ndarray) -> np.ndarray:
    """m[i] = number of missing bars among positions [0, i)."""
    return np.concatenate([[0], np.cumsum(~valid)])


def context_ok(bars: Bars, lookback: int = 1500) -> np.ndarray:
    """True at ``t`` when the bars ``t - lookback .. t`` all exist."""
    n = bars.n
    miss = _missing_before(bars.valid)
    t = np.arange(n)
    lo = np.maximum(t - lookback, 0)
    return ((miss[t + 1] - miss[lo]) == 0) & (t >= lookback)


def forward_returns(bars: Bars, horizons=HORIZONS) -> dict[int, np.ndarray]:
    """``f[h][t] = (O[t+1+h] / O[t+1] - 1) * 1e4``; NaN unless bars ``t .. t+1+h`` all exist."""
    n, o = bars.n, bars.o
    miss = _missing_before(bars.valid)
    out = {}
    for h in horizons:
        f = np.full(n, np.nan)
        m = n - 1 - h
        if m > 0:
            t = np.arange(m)
            ok = (miss[t + h + 2] - miss[t]) == 0
            r = (o[h + 1:] / o[1:n - h] - 1.0) * 1e4
            f[:m] = np.where(ok, r, np.nan)
        out[h] = f
    return out


def forward_from_price(bars: Bars, idx: np.ndarray, price: np.ndarray, horizons=HORIZONS) -> dict[int, np.ndarray]:
    """Returns from a given fill price obtained DURING bar ``idx`` (a resting order) to the open of ``idx+1+h``.

    Horizon 0 is allowed and gives the mark-to-market at the next open (adverse selection).
    """
    n, o = bars.n, bars.o
    miss = _missing_before(bars.valid)
    out = {}
    for h in horizons:
        j = idx + 1 + h
        ok = j < n
        jj = np.where(ok, j, 0)
        clean = ok & ((miss[np.minimum(jj + 1, n)] - miss[idx]) == 0)
        out[h] = np.where(clean, (o[jj] / price - 1.0) * 1e4, np.nan)
    return out


# --------------------------------------------------------------------------- #
# Baseline: unconditional forward return matched on year, hour of day and volatility state
# --------------------------------------------------------------------------- #

def realised_vol(bars: Bars, window: int = 60) -> np.ndarray:
    """Root of the sum of squared 1-minute close-to-close log returns over the last ``window`` bars (known at close t)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.diff(np.log(bars.c), prepend=np.nan)
    return np.sqrt(pd.Series(r * r).rolling(window, min_periods=window).sum().to_numpy())


def baseline_cells(bars: Bars, n_vol: int = 5, window: int = 60) -> tuple[np.ndarray, int]:
    """Cell id per bar: (calendar year, hour of day, volatility quintile within the year). -1 where undefined.

    The quintile edges are computed inside each calendar year. This is a control
    variable for a benchmark, not a trading signal, so it need not be causal.
    """
    rv = realised_vol(bars, window)
    year = bars.index.year.to_numpy()
    hour = ((bars.minute % MINUTES_PER_DAY) // 60).astype(np.int64)
    years = np.unique(year)
    q = np.full(bars.n, -1, dtype=np.int64)
    for y in years:
        sel = (year == y) & np.isfinite(rv)
        if sel.sum() < n_vol * 100:
            continue
        edges = np.quantile(rv[sel], np.linspace(0, 1, n_vol + 1)[1:-1])
        q[sel] = np.searchsorted(edges, rv[sel], side="right")
    year_idx = np.searchsorted(years, year)
    cell = (year_idx * 24 + hour) * n_vol + q
    cell[q < 0] = -1
    return cell, int(len(years) * 24 * n_vol)


def matched_baseline(fwd: dict[int, np.ndarray], cell: np.ndarray, n_cells: int) -> dict[int, np.ndarray]:
    """Mean forward return of all bars in the same cell (the unconditional return an entry at a random bar of that cell earns)."""
    out = {}
    for h, f in fwd.items():
        ok = np.isfinite(f) & (cell >= 0)
        total = np.bincount(cell[ok], weights=f[ok], minlength=n_cells)
        count = np.bincount(cell[ok], minlength=n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / count, np.nan)
        b = np.full(len(f), np.nan)
        b[cell >= 0] = mean[cell[cell >= 0]]
        out[h] = b
    return out


# --------------------------------------------------------------------------- #
# Per-day sufficient statistics
# --------------------------------------------------------------------------- #

def aggregate_daily(day: np.ndarray, direction: np.ndarray, gross: dict[int, np.ndarray],
                    base: dict[int, np.ndarray], horizons=HORIZONS) -> pd.DataFrame:
    """Per (day, direction) sums. ``gross`` and ``base`` are already signed, one value per event.

    For each horizon h: n (events with a defined return), sg (sum gross), sb (sum
    baseline), sw (number with gross > 0), sgg (sum gross^2), see (sum (gross - base)^2).
    """
    if len(day) == 0:
        return pd.DataFrame(columns=["day", "dir"] + [f"{f}{h}" for h in horizons for f in STAT_FIELDS])
    d0 = int(day.min())
    key = (day - d0) * 2 + (direction > 0)
    size = int(key.max()) + 1
    cols: dict[str, np.ndarray] = {}
    for h in horizons:
        g, b = gross[h], base[h]
        ok = np.isfinite(g) & np.isfinite(b)
        k, gg, bb = key[ok], g[ok], b[ok]
        e = gg - bb
        cols[f"n{h}"] = np.bincount(k, minlength=size).astype(np.int64)
        cols[f"sg{h}"] = np.bincount(k, weights=gg, minlength=size)
        cols[f"sb{h}"] = np.bincount(k, weights=bb, minlength=size)
        cols[f"sw{h}"] = np.bincount(k, weights=(gg > 0).astype(float), minlength=size)
        cols[f"sgg{h}"] = np.bincount(k, weights=gg * gg, minlength=size)
        cols[f"see{h}"] = np.bincount(k, weights=e * e, minlength=size)
    frame = pd.DataFrame(cols)
    frame.insert(0, "dir", np.where(np.arange(size) % 2 == 1, 1, -1).astype(np.int8))
    frame.insert(0, "day", d0 + np.arange(size) // 2)
    keep = frame[[f"n{h}" for h in horizons]].sum(axis=1) > 0
    return frame[keep].reset_index(drop=True)


def event_daily(bars: Bars, idx: np.ndarray, direction: np.ndarray, fwd: dict[int, np.ndarray],
                base: dict[int, np.ndarray], ok: np.ndarray, entry_px: np.ndarray | None = None,
                horizons=HORIZONS) -> pd.DataFrame:
    """Daily sufficient statistics of one event set on one symbol.

    ``idx``: bar positions of the events; ``direction``: +1 / -1 hypothesis sign;
    ``ok``: per-bar mask of acceptable context (no gap in the look-back window).
    ``entry_px``: fill price of a resting order filled during bar ``idx``
    (otherwise the fill is the next bar's open).
    """
    keep = ok[idx]
    idx, direction = idx[keep], direction[keep].astype(float)
    if entry_px is not None:
        entry_px = entry_px[keep]
        raw = forward_from_price(bars, idx, entry_px, horizons)
        gross = {h: direction * raw[h] for h in horizons}
    else:
        gross = {h: direction * fwd[h][idx] for h in horizons}
    signed_base = {h: direction * base[h][idx] for h in horizons}
    return aggregate_daily(bars.day[idx], direction, gross, signed_base, horizons)


# --------------------------------------------------------------------------- #
# Day-clustered inference
# --------------------------------------------------------------------------- #

def cluster_mean_t(total: np.ndarray, count: np.ndarray, cluster_id: np.ndarray, lag: int = 0) -> tuple[float, float, float]:
    """Mean of event values and its cluster-robust t-statistic.

    ``total[g]`` = sum of the values of the events in cluster g, ``count[g]`` =
    their number, ``cluster_id[g]`` = integer day of the cluster. With ``lag``
    = 1 the covariance between ADJACENT days is added (needed when the forward
    returns of events on consecutive days overlap); the adjustment is never
    allowed to shrink the variance below the plain day-clustered one, so the
    statistic can only become more conservative.

    Returns (mean, standard error, t).
    """
    total = np.asarray(total, dtype=float)
    count = np.asarray(count, dtype=float)
    keep = count > 0
    total, count, cluster_id = total[keep], count[keep], np.asarray(cluster_id)[keep]
    groups, n = len(count), count.sum()
    if groups < 2 or n == 0:
        return (float(total.sum() / n) if n else float("nan")), float("nan"), float("nan")
    mean = total.sum() / n
    resid = total - count * mean
    meat = float((resid ** 2).sum())
    if lag >= 1:
        order = np.argsort(cluster_id)
        r, ids = resid[order], cluster_id[order]
        extra = 0.0
        for step in range(1, lag + 1):
            # pair every cluster with the cluster exactly `step` days later
            pos = np.searchsorted(ids, ids + step)
            hit = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == ids + step)
            extra += 2.0 * float((r[hit] * r[pos[hit]]).sum())
        meat = max(meat, meat + extra)
    var = meat / n ** 2 * groups / (groups - 1.0)
    se = float(np.sqrt(var))
    return float(mean), se, float(mean / se) if se > 0 else float("nan")


def hac_lag(horizon: int) -> int:
    """Adjacent-day covariance is added when returns can overlap across midnight to a material extent."""
    return 1 if horizon >= 240 else 0


def pooled_stats(daily: pd.DataFrame, horizons=HORIZONS, cost_bp: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per horizon for a set of daily rows (any mix of symbols and directions).

    Columns: events, days, gross (mean bp in hypothesis direction), base
    (matched unconditional return, same sign convention), excess = gross - base,
    t_excess and t_gross (day-clustered), t_naive (i.i.d. events, shown to expose
    the inflation), hit (share of events with gross > 0). When ``cost_bp``
    (symbol -> round-trip cost) and a ``symbol`` column are present: cost and net.
    """
    rows = []
    for h in horizons:
        names = [f"{f}{h}" for f in STAT_FIELDS]
        if names[0] not in daily.columns:  # horizon not computed for these rows (validation: selected pairs only)
            rows.append({"h": h, "events": 0, "days": 0})
            continue
        by_day = daily.groupby("day", sort=True)[names].sum()
        by_day = by_day[by_day[f"n{h}"] > 0]
        n = float(by_day[f"n{h}"].sum())
        row = {"h": h, "events": int(n), "days": int(len(by_day))}
        if n == 0:
            rows.append(row)
            continue
        ids = by_day.index.to_numpy()
        cnt = by_day[f"n{h}"].to_numpy(dtype=float)
        sg, sb = by_day[f"sg{h}"].to_numpy(), by_day[f"sb{h}"].to_numpy()
        lag = hac_lag(h)
        gross, se_g, t_g = cluster_mean_t(sg, cnt, ids, lag)
        excess, se_e, t_e = cluster_mean_t(sg - sb, cnt, ids, lag)
        var_naive = by_day[f"see{h}"].sum() / n - excess ** 2
        row.update(gross=gross, base=float(sb.sum() / n), excess=excess, se_excess=se_e, t_excess=t_e, t_gross=t_g,
                   t_naive=float(excess / np.sqrt(var_naive / n)) if var_naive > 0 else float("nan"),
                   hit=float(by_day[f"sw{h}"].sum() / n))
        if cost_bp is not None and "symbol" in daily.columns:
            per_symbol = daily.groupby("symbol", observed=True)[f"n{h}"].sum()
            cost = float(sum(per_symbol[s] * cost_bp[s] for s in per_symbol.index) / n)
            row.update(cost=cost, net=gross - cost)
        rows.append(row)
    return pd.DataFrame(rows).set_index("h")


def block_bootstrap_p(total: np.ndarray, count: np.ndarray, cluster_id: np.ndarray, block: int = 1,
                      n_boot: int = 4000, seed: int = 0) -> dict:
    """Moving-block bootstrap over the calendar (blocks of ``block`` consecutive days) for the event-weighted mean.

    Returns the bootstrap standard error, a 95% interval and the two-sided
    p-value of the mean being zero (from the centred bootstrap distribution).
    """
    cluster_id = np.asarray(cluster_id)
    first, last = int(cluster_id.min()), int(cluster_id.max())
    span = last - first + 1
    s = np.zeros(span)
    c = np.zeros(span)
    np.add.at(s, cluster_id - first, total)
    np.add.at(c, cluster_id - first, count)
    mean = s.sum() / c.sum()
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(span / block))
    starts = rng.integers(0, span - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    pick = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :span]
    boot_s, boot_c = s[pick].sum(axis=1), c[pick].sum(axis=1)
    boot = boot_s[boot_c > 0] / boot_c[boot_c > 0]
    centred = boot - boot.mean()
    return {
        "mean": float(mean),
        "se": float(boot.std(ddof=1)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "p_two_sided": float(np.mean(np.abs(centred) >= abs(mean))),
    }
