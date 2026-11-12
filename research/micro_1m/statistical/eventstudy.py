"""Event-study toolkit for 1-minute bars.

Timing convention (everything else follows from it)
---------------------------------------------------
Bars are indexed by their UTC *open* time. An event is flagged at the CLOSE of
bar ``t`` and may use bars ``<= t`` only. The earliest fill is the OPEN of bar
``t + 1``. The forward return over ``h`` minutes is

    fwd_h[t] = open[t + 1 + h] / open[t + 1] - 1

so nothing that happens inside bar ``t`` itself, nor the step from close[t] to
open[t + 1], is ever credited to the event.

Gaps
----
Bars are laid out on a complete 1-minute grid. A bar is an *outage* bar when it
is missing from the index or when it sits inside a run of at least
``OUTAGE_RUN`` consecutive zero-volume bars (Binance fills halts with flat,
zero-volume bars on the perpetual feed). A forward return is kept only if every
bar from the entry bar to the exit bar is a non-outage bar and both the entry
and the exit bar actually traded (volume > 0). Features are computed from a
*causal* view of the same information: a bar is blanked once the zero-volume
run that it belongs to has already lasted ``OUTAGE_RUN`` bars, so an event flag
never depends on bars after it.

Inference
---------
Events cluster in time and overlapping horizons share returns, so observations
are aggregated to (calendar day) sums and the standard error of the mean is
cluster-robust by day, with a Bartlett correction across adjacent days for the
horizons whose windows cross midnight. Every statistic can be recomputed from
the per-day sums that :func:`aggregate` returns, which is also what makes the
by-year and by-symbol breakdowns cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 15, 60, 240, 1440)
OUTAGE_RUN = 5
#: Bartlett lags (in days) added to the day-clustered variance, by horizon in minutes.
HAC_LAGS = {1: 0, 5: 0, 15: 0, 60: 1, 240: 1, 1440: 3}
#: cut-offs of (trailing 60-min RMS return) / (trailing 1-day RMS return); about the terciles on design data.
VOL_STATE_CUTS = (0.7, 1.0)
TRAIL = 1440  # one day of 1-minute bars: window of every trailing standardisation
TRAIL_MIN = 1296  # 90% of it must be present

_NS_PER_MIN = 60 * 10**9
_NS_PER_DAY = 1440 * _NS_PER_MIN


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #


@dataclass
class Grid:
    """One symbol's bars on a complete 1-minute grid (NaN where unusable)."""

    market: str
    symbol: str
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    qv: np.ndarray  # quote volume
    trades: np.ndarray
    tbqv: np.ndarray  # taker-buy quote volume
    raw_open: np.ndarray  # open with NaN only where the bar is missing (used for fills)
    raw_qv: np.ndarray
    raw_high: np.ndarray
    raw_low: np.ndarray
    raw_vwap: np.ndarray  # quote volume / base volume of the bar; NaN when it did not trade
    exists: np.ndarray  # bar present in the source index
    usable: np.ndarray  # causal: present and not yet recognisable as a halt
    outage: np.ndarray  # missing, or anywhere inside a zero-volume run of >= OUTAGE_RUN bars (forward validity only)
    day: np.ndarray  # days since 1970-01-01 (UTC) of the bar's open time
    minute_of_day: np.ndarray

    def __len__(self) -> int:
        return len(self.index)


def run_lengths(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For runs of True: (length of the run up to and including t, total length of the run)."""
    n = len(flag)
    position = np.arange(n)
    last_false = np.maximum.accumulate(np.where(~flag, position, -1))
    so_far = np.where(flag, position - last_false, 0)
    next_false = np.minimum.accumulate(np.where(~flag, position, n)[::-1])[::-1]
    total = np.where(flag, next_false - last_false - 1, 0)
    return so_far, total


def to_grid(frame: pd.DataFrame, market: str = "", symbol: str = "", outage_run: int = OUTAGE_RUN) -> Grid:
    """Lay bars out on a complete 1-minute grid and mark gaps and halts."""
    index = frame.index
    if index.tz is None:
        raise ValueError("bar index must be tz-aware (UTC)")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("bar index must be strictly increasing")
    stamps = index.as_unit("ns").asi8
    if (stamps % _NS_PER_MIN).any():
        raise ValueError("bar open times must fall on whole minutes")
    position = (stamps - stamps[0]) // _NS_PER_MIN
    n = int(position[-1]) + 1
    full = pd.date_range(index[0], periods=n, freq="1min")
    full_ns = stamps[0] + np.arange(n, dtype=np.int64) * _NS_PER_MIN

    def column(name: str) -> np.ndarray:
        out = np.full(n, np.nan)
        out[position] = frame[name].to_numpy(dtype=float)
        return out

    exists = np.zeros(n, dtype=bool)
    exists[position] = True
    raw_qv = column("quote_volume")
    zero = exists & (raw_qv <= 0)
    so_far, total = run_lengths(zero)
    stale = zero & (so_far >= outage_run)  # knowable at the bar's own close
    outage = ~exists | (zero & (total >= outage_run))  # uses the whole run: forward validity only
    usable = exists & ~stale

    fields = {}
    for key, name in (
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("qv", "quote_volume"),
        ("trades", "trades"),
        ("tbqv", "taker_buy_quote_volume"),
    ):
        values = raw_qv.copy() if name == "quote_volume" else column(name)
        if key == "open":
            raw_open = values.copy()
        elif key == "high":
            raw_high = values.copy()
        elif key == "low":
            raw_low = values.copy()
        values[stale] = np.nan
        fields[key] = values
    base_volume = column("volume") if "volume" in frame.columns else np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_vwap = np.where(base_volume > 0, raw_qv / base_volume, np.nan)
    return Grid(
        market=market,
        symbol=symbol,
        index=full,
        raw_open=raw_open,
        raw_qv=raw_qv,
        raw_high=raw_high,
        raw_low=raw_low,
        raw_vwap=raw_vwap,
        exists=exists,
        usable=usable,
        outage=outage,
        day=(full_ns // _NS_PER_DAY).astype(np.int64),
        minute_of_day=((full_ns % _NS_PER_DAY) // _NS_PER_MIN).astype(np.int64),
        **fields,
    )


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #


def forward_returns(grid: Grid, horizons=HORIZONS, entry_lag: int = 1, entry_price: str = "open", exit_price: str = "open") -> dict[int, np.ndarray]:
    """fwd_h[t] = open[t+1+h] / open[t+1] - 1; NaN unless the whole holding window is clean.

    Clean means: bars t+1 .. t+1+h all exist and none is an outage bar, and the
    entry bar (t+1) and exit bar (t+1+h) both traded.

    The defaults are the convention of the study. For robustness checks only:
    ``entry_lag`` = k enters at bar t+k (k >= 1) and leaves h bars later;
    ``entry_price`` / ``exit_price`` in {"open", "vwap", "high", "low"} take the fill
    from that field of the entry / exit bar (vwap: the bar's volume-weighted price;
    high / low: the worst price a buyer / seller could have paid inside the bar).
    """
    if entry_lag < 1:
        raise ValueError("entry_lag must be >= 1: a fill cannot precede the close of the event bar")
    fields = {"open": grid.raw_open, "vwap": grid.raw_vwap, "high": grid.raw_high, "low": grid.raw_low}
    n = len(grid)
    good = ~grid.outage
    count = np.concatenate([[0], np.cumsum(good)])  # good bars in [0, i)
    tradable = good & (grid.raw_qv > 0)
    price_in, price_out = fields[entry_price], fields[exit_price]
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        values = np.full(n, np.nan)
        m = n - entry_lag - h
        if m > 0:
            entry = np.arange(entry_lag, m + entry_lag)
            exit_ = entry + h
            first = entry - entry_lag + 1  # bar t+1: every bar from the first possible fill to the exit must be clean
            clean = (count[exit_ + 1] - count[first]) == (h + entry_lag)
            valid = clean & tradable[entry] & tradable[exit_]
            with np.errstate(invalid="ignore", divide="ignore"):
                ret = price_out[exit_] / price_in[entry] - 1.0
            values[:m] = np.where(valid, ret, np.nan)
        out[int(h)] = values
    return out


# --------------------------------------------------------------------------- #
# Trailing (causal) features
# --------------------------------------------------------------------------- #


def trailing_features(grid: Grid) -> dict[str, np.ndarray]:
    """Features known at the close of each bar. Every value at t uses bars <= t only."""
    close = pd.Series(grid.close)
    open_ = pd.Series(grid.open)
    high = pd.Series(grid.high)
    low = pd.Series(grid.low)
    qv = pd.Series(grid.qv)
    trades = pd.Series(grid.trades)
    tbqv = pd.Series(grid.tbqv)
    out: dict[str, np.ndarray] = {}

    r1 = close / close.shift(1) - 1.0
    sigma1 = r1.rolling(TRAIL, min_periods=TRAIL_MIN).std()  # 1-day std of 1-minute returns, ending at t
    out["r1"] = r1.to_numpy()
    for w in (1, 5, 15, 60):
        ret = close / close.shift(w) - 1.0
        out[f"ret{w}"] = ret.to_numpy()
        if w <= 15:
            # standardise by volatility measured strictly before the move's own window
            scale = sigma1.shift(w) * np.sqrt(w)
            out[f"z{w}"] = (ret / scale.where(scale > 0)).to_numpy()

    # volume and trade-count spikes: bar t against the median of the 60 bars before it
    med_v = qv.rolling(60, min_periods=54).median().shift(1)
    med_t = trades.rolling(60, min_periods=54).median().shift(1)
    out["vratio"] = (qv / med_v.where(med_v > 0)).to_numpy()
    out["tratio"] = (trades / med_t.where(med_t > 0)).to_numpy()

    # bar direction and the wick on the side the bar travelled towards, as a share of its range
    body = (close - open_).to_numpy()
    span = (high - low).to_numpy()
    direction = np.sign(body)
    with np.errstate(invalid="ignore", divide="ignore"):
        up_wick = (high.to_numpy() - close.to_numpy()) / span
        down_wick = (close.to_numpy() - low.to_numpy()) / span
    wick = np.where(direction > 0, up_wick, np.where(direction < 0, down_wick, np.nan))
    wick[~(span > 0)] = np.nan
    out["bar_dir"] = np.where(np.isfinite(body), direction, np.nan)
    out["wick"] = wick

    # taker-buy share of quote volume over the last w minutes, and its trailing z-score
    for w in (1, 5, 15):
        vol_w = qv.rolling(w, min_periods=w).sum()
        buy_w = tbqv.rolling(w, min_periods=w).sum()
        share = buy_w / vol_w.where(vol_w > 0)
        mean = share.rolling(TRAIL, min_periods=TRAIL_MIN).mean().shift(w)
        sd = share.rolling(TRAIL, min_periods=TRAIL_MIN).std().shift(w)
        out[f"ofi{w}"] = share.to_numpy()
        out[f"ofiz{w}"] = ((share - mean) / sd.where(sd > 0)).to_numpy()

    # volatility state for the matched baseline
    rms60 = np.sqrt((r1**2).rolling(60, min_periods=54).mean())
    rms_day = np.sqrt((r1**2).rolling(TRAIL, min_periods=TRAIL_MIN).mean())
    ratio = (rms60 / rms_day.where(rms_day > 0)).to_numpy()
    state = np.full(len(grid), -1, dtype=np.int64)
    finite = np.isfinite(ratio)
    state[finite] = np.digitize(ratio[finite], VOL_STATE_CUTS)
    out["vol_state"] = state
    return out


def first_in_window(mask: np.ndarray, window: int) -> np.ndarray:
    """Causal thinning: keep a flagged bar only if no kept bar lies in the previous ``window - 1`` bars."""
    if window <= 1:
        return mask.copy()
    flagged = np.flatnonzero(mask)
    keep = np.zeros(len(flagged), dtype=bool)
    last = -(10**12)
    for i, t in enumerate(flagged):
        if t - last >= window:
            keep[i] = True
            last = t
    out = np.zeros(len(mask), dtype=bool)
    out[flagged[keep]] = True
    return out


# --------------------------------------------------------------------------- #
# Baseline: unconditional forward return of comparable bars
# --------------------------------------------------------------------------- #


def baseline_cells(grid: Grid, vol_state: np.ndarray, by_hour: bool = True) -> tuple[np.ndarray, int]:
    """Cell id per bar: calendar year x (hour of day) x volatility state."""
    year = grid.index.year.to_numpy()
    year_id = year - year.min()
    state = vol_state + 1  # -1 (unknown) becomes its own cell
    if by_hour:
        hour = grid.minute_of_day // 60
        cell = (year_id * 24 + hour) * 4 + state
        return cell.astype(np.int64), int((year_id.max() + 1) * 24 * 4)
    cell = year_id * 4 + state
    return cell.astype(np.int64), int((year_id.max() + 1) * 4)


def baseline_returns(fwd: dict[int, np.ndarray], cell: np.ndarray, n_cells: int) -> dict[int, np.ndarray]:
    """Mean forward return of all bars in the same cell (the event bars included).

    This is a benchmark for inference, not a tradable quantity: it uses the
    whole segment on purpose.
    """
    out = {}
    for h, values in fwd.items():
        ok = np.isfinite(values)
        count = np.bincount(cell[ok], minlength=n_cells)
        total = np.bincount(cell[ok], weights=values[ok], minlength=n_cells)
        mean = np.divide(total, count, out=np.full(n_cells, np.nan), where=count > 0)
        out[h] = mean[cell]
    return out


# --------------------------------------------------------------------------- #
# Aggregation to day sums, and statistics from day sums
# --------------------------------------------------------------------------- #

AGG_FIELDS = ("n", "s", "b", "q", "w")  # events, sum x, sum baseline, sum x^2, wins


def aggregate(
    mask: np.ndarray,
    direction,
    fwd: dict[int, np.ndarray],
    base: dict[int, np.ndarray],
    day_index: np.ndarray,
    n_days: int,
    horizons=HORIZONS,
) -> np.ndarray:
    """Per-day sums of the directional forward returns of the flagged bars.

    Returns an array of shape (len(AGG_FIELDS), len(horizons), n_days). ``direction``
    is +1 / -1 per bar (or a scalar): the side of the hypothesis. ``day_index`` maps
    each bar to 0 .. n_days-1.
    """
    out = np.zeros((len(AGG_FIELDS), len(horizons), n_days))
    flagged = np.flatnonzero(mask)
    if len(flagged) == 0:
        return out
    side = direction[flagged].astype(float) if isinstance(direction, np.ndarray) else float(direction)
    days = day_index[flagged]
    for j, h in enumerate(horizons):
        x = fwd[h][flagged] * side
        ok = np.isfinite(x)
        if not ok.any():
            continue
        xs, ds = x[ok], days[ok]
        bs = (base[h][flagged] * side)[ok]
        out[0, j] = np.bincount(ds, minlength=n_days)
        out[1, j] = np.bincount(ds, weights=xs, minlength=n_days)
        out[2, j] = np.bincount(ds, weights=bs, minlength=n_days)
        out[3, j] = np.bincount(ds, weights=xs * xs, minlength=n_days)
        out[4, j] = np.bincount(ds, weights=(xs > 0).astype(float), minlength=n_days)
    return out


def cluster_mean(n_day: np.ndarray, s_day: np.ndarray, hac_lags: int = 0) -> dict:
    """Mean of event values and its day-clustered standard error.

    ``n_day[d]`` events with value sum ``s_day[d]`` on consecutive calendar days.
    Variance: sum of squared cluster scores e_d = s_d - mean * n_d (CR1 small-sample
    factor G/(G-1)), plus Bartlett-weighted cross products of adjacent days when
    ``hac_lags`` > 0 (for horizons whose windows run into the next days).
    """
    n_day = np.asarray(n_day, dtype=float)
    s_day = np.asarray(s_day, dtype=float)
    total = n_day.sum()
    groups = int((n_day > 0).sum())
    if total == 0 or groups < 2:
        return {"n": int(total), "days": groups, "mean": float("nan"), "se": float("nan"), "t": float("nan")}
    mean = s_day.sum() / total
    score = s_day - mean * n_day
    variance = float(score @ score)
    if hac_lags > 0:
        extra = 0.0
        for lag in range(1, hac_lags + 1):
            weight = 1.0 - lag / (hac_lags + 1.0)
            extra += 2.0 * weight * float(score[lag:] @ score[:-lag])
        if variance + extra > 0:
            variance += extra
    variance *= groups / (groups - 1.0)
    se = np.sqrt(variance) / total
    return {"n": int(total), "days": groups, "mean": float(mean), "se": float(se), "t": float(mean / se) if se > 0 else float("nan")}


def naive_t(n: float, s: float, q: float) -> float:
    """t-statistic that treats every event as independent (shown only to expose the inflation)."""
    if n < 2:
        return float("nan")
    mean = s / n
    var = (q - n * mean * mean) / (n - 1.0)
    return float(mean / np.sqrt(var / n)) if var > 0 else float("nan")


def day_block_bootstrap(n_day: np.ndarray, s_day: np.ndarray, block: int = 5, n_boot: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap of the ratio mean over days (blocks of consecutive calendar days)."""
    n_day = np.asarray(n_day, dtype=float)
    s_day = np.asarray(s_day, dtype=float)
    days = len(n_day)
    if n_day.sum() == 0 or days < 2 * block:
        return {"mean": float("nan"), "se": float("nan"), "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(days / block))
    starts = rng.integers(0, days - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    picks = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :days]
    num = s_day[picks].sum(axis=1)
    den = n_day[picks].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(den > 0, num / den, np.nan)
    means = means[np.isfinite(means)]
    mean = s_day.sum() / n_day.sum()
    centred = means - means.mean()
    p = float(np.mean(np.abs(centred) >= abs(mean)))
    return {"mean": float(mean), "se": float(means.std(ddof=1)), "p_two_sided": p, "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975))}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #

LOW_SLIPPAGE = {"BTCUSDT": 1.0, "ETHUSDT": 1.0}
COST_MODES = ("perp_taker", "perp_maker", "spot_taker")


def round_trip_cost_bp(symbol: str, mode: str) -> float:
    """Cost of entering and leaving once, in basis points of notional.

    perp_taker: 2 x (5 bp fee + slippage); slippage 1 bp for BTC/ETH, 2 bp otherwise.
    perp_maker: 2 x 2 bp fee and no slippage. Optimistic bound: a resting order
                only fills when price trades through it (adverse selection) and
                sometimes not at all.
    spot_taker: 2 x (10 bp fee + the same slippage).
    """
    slippage = LOW_SLIPPAGE.get(symbol, 2.0)
    if mode == "perp_taker":
        return 2.0 * (5.0 + slippage)
    if mode == "perp_maker":
        return 2.0 * 2.0
    if mode == "spot_taker":
        return 2.0 * (10.0 + slippage)
    raise KeyError(mode)


# --------------------------------------------------------------------------- #
# One-call event study for a single symbol (used by the tests and the runners)
# --------------------------------------------------------------------------- #


def summarise(agg: np.ndarray, horizons=HORIZONS, cost_bp: float | None = None) -> pd.DataFrame:
    """Table (one row per horizon) from an array returned by :func:`aggregate` (or a sum of them)."""
    rows = []
    for j, h in enumerate(horizons):
        n_day, s_day, b_day, q_day, w_day = (agg[k, j] for k in range(5))
        gross = cluster_mean(n_day, s_day, HAC_LAGS.get(h, 0))
        excess = cluster_mean(n_day, s_day - b_day, HAC_LAGS.get(h, 0))
        total = n_day.sum()
        row = {
            "h": h,
            "events": int(total),
            "days": gross["days"],
            "gross_bp": gross["mean"] * 1e4,
            "base_bp": (b_day.sum() / total * 1e4) if total else float("nan"),
            "excess_bp": excess["mean"] * 1e4,
            "t_day": excess["t"],
            "t_gross": gross["t"],
            "t_naive": naive_t(total, s_day.sum(), q_day.sum()),
            "hit": (w_day.sum() / total) if total else float("nan"),
        }
        if cost_bp is not None:
            row["net_bp"] = row["gross_bp"] - cost_bp
        rows.append(row)
    return pd.DataFrame(rows).set_index("h")
