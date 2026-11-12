"""Exact, parameterised, backward-looking definitions of 1-minute price-action structures.

These are the patterns that discretionary "smart money / ICT" traders name on
1-minute charts, turned into rules a computer can apply without judgement.
Every detector returns events that are KNOWN AT THE CLOSE of the bar they are
stamped with; nothing after that bar is used (proved by perturbation in
test_eventstudy.py). The sign of an event is the direction of the trade the
pattern is said to call for (+1 long, -1 short).

Families
--------
sweep        liquidity sweep and rejection ("stop run"): trade through the prior
             N-minute extreme by >= x bp, close back inside within k bars.
             Hypothesis: reversal. (A negative result is the continuation reading.)
sweepmat     the same, but the swept level must be at least 10 minutes old.
hold         the complement: trade through by >= x bp and CLOSE beyond the level
             for k consecutive bars. Hypothesis: continuation.
bos          break of structure: first close beyond the most recent confirmed
             swing high / low (swing = left / right bar count, known only after
             its right-hand bars). "with" = same direction as the previous break
             (BOS proper), "against" = opposite (change of character, CHoCH).
             Hypothesis: continuation in the direction of the break.
fvg          fair value gap (three-bar imbalance) of at least m x the average
             1-minute range. Hypothesis: continuation.
fvgfill      first return into an open gap within W minutes, entered at the next
             open provided the bar did not close through the far side.
             Hypothesis: bounce in the direction of the gap.
fvglimit     the same entry as a resting limit order at the gap edge (filled only
             if price trades strictly through it; never conditions on the close).
squeeze      range compression then breakout: the prior M-minute range is in the
             lowest q of its trailing day, then a close outside it; optional
             volume filter. Hypothesis: continuation.
pdsweep      sweep and rejection of the previous UTC day's high / low.
pdhold       acceptance beyond the previous day's high / low.
ob           order block: last opposite-colour candle before a structure break;
             first return into it within W minutes. Hypothesis: bounce.
oblimit      the same as a resting limit order at the block edge.
disp         displacement candle: body >= m x average body, closing in the outer
             quarter of its range. Hypothesis: continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from eventstudy import Bars, MINUTES_PER_DAY


@dataclass
class EventSet:
    def_id: str
    family: str
    params: dict
    hypothesis: str
    idx: np.ndarray  # bar position; the event is known at the close of this bar
    direction: np.ndarray  # +1 long, -1 short
    entry_px: np.ndarray | None = None  # fill price of a resting order filled during bar idx; None = next open
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Small causal helpers
# --------------------------------------------------------------------------- #

def shift(a: np.ndarray, j: int, fill) -> np.ndarray:
    """Value at t is a[t - j] (j >= 0)."""
    if j == 0:
        return a
    out = np.empty_like(a)
    out[:j] = fill
    out[j:] = a[:-j]
    return out


def prev_max(x: np.ndarray, n: int) -> np.ndarray:
    """max(x[t-n .. t-1]); NaN unless all n values exist."""
    return pd.Series(x).rolling(n, min_periods=n).max().shift(1).to_numpy()


def prev_min(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).min().shift(1).to_numpy()


def prev_mean(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().shift(1).to_numpy()


def from_signal(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(sig)
    return idx, sig[idx].astype(np.int8)


# --------------------------------------------------------------------------- #
# 1. Liquidity sweep / rejection, and its complement (acceptance)
# --------------------------------------------------------------------------- #

def _breach(h, l, c, lev_hi, lev_lo, x_bp):
    """Bar s trades beyond the level by >= x bp, coming from inside (previous close not beyond it)."""
    c1 = shift(c, 1, np.nan)
    up = (h >= lev_hi * (1.0 + x_bp * 1e-4)) & (c1 <= lev_hi)
    dn = (l <= lev_lo * (1.0 - x_bp * 1e-4)) & (c1 >= lev_lo)
    return up, dn


def sweep_reject(h, l, c, lev_hi, lev_lo, x_bp: float, k: int, ok_hi=None, ok_lo=None) -> np.ndarray:
    """+1 after a sweep of the low that closes back above it, -1 after a sweep of the high that closes back below it.

    The breach happens in bar s; the event is the FIRST close back inside the
    level, provided it comes in bar s .. s + k - 1 (k = 1: the breach bar itself).
    A bar that qualifies in both directions is dropped.
    """
    up, dn = _breach(h, l, c, lev_hi, lev_lo, x_bp)
    if ok_hi is not None:
        up &= ok_hi
    if ok_lo is not None:
        dn &= ok_lo
    bear = np.zeros(len(c), dtype=bool)
    bull = np.zeros(len(c), dtype=bool)
    for j in range(k):
        lv_hi, lv_lo = shift(lev_hi, j, np.nan), shift(lev_lo, j, np.nan)
        b = shift(up, j, False) & (c < lv_hi)
        u = shift(dn, j, False) & (c > lv_lo)
        for i in range(1, j + 1):
            ci = shift(c, i, np.nan)
            b &= ci >= lv_hi
            u &= ci <= lv_lo
        bear |= b
        bull |= u
    return bull.astype(np.int8) - bear.astype(np.int8)


def break_hold(h, l, c, lev_hi, lev_lo, x_bp: float, k: int) -> np.ndarray:
    """+1 when bar s breaches the high by >= x bp and bars s .. s+k-1 ALL close above it (event at s+k-1); -1 mirror."""
    up, dn = _breach(h, l, c, lev_hi, lev_lo, x_bp)
    j = k - 1
    lv_hi, lv_lo = shift(lev_hi, j, np.nan), shift(lev_lo, j, np.nan)
    long_, short = shift(up, j, False), shift(dn, j, False)
    for i in range(0, j + 1):
        ci = shift(c, i, np.nan)
        long_ = long_ & (ci > lv_hi)
        short = short & (ci < lv_lo)
    return long_.astype(np.int8) - short.astype(np.int8)


def previous_day_levels(minute: np.ndarray, h: np.ndarray, l: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """High and low of the previous UTC day (NaN unless that day has all 1440 bars)."""
    k = (minute // MINUTES_PER_DAY) - (minute[0] // MINUTES_PER_DAY)
    hi = pd.Series(h).groupby(k).max().to_numpy()
    lo = pd.Series(l).groupby(k).min().to_numpy()
    full = pd.Series(valid).groupby(k).sum().to_numpy() == MINUTES_PER_DAY
    prev_hi = np.concatenate([[np.nan], np.where(full, hi, np.nan)[:-1]])
    prev_lo = np.concatenate([[np.nan], np.where(full, lo, np.nan)[:-1]])
    return prev_hi[k], prev_lo[k]


# --------------------------------------------------------------------------- #
# 2. Swings and breaks of structure
# --------------------------------------------------------------------------- #

def confirmed_swings(h: np.ndarray, l: np.ndarray, left: int, right: int):
    """Most recent CONFIRMED swing high / low as known at the close of each bar.

    Bar s is a swing high when h[s] > max(h[s-left .. s-1]) and h[s] >= max(h[s+1 .. s+right]).
    It is confirmed (usable) at bar s + right. Returns (level_hi, conf_hi, level_lo, conf_lo)
    where conf_* is the bar at which the level in force was confirmed (-1 if none yet).
    """
    n = len(h)
    hs, ls = pd.Series(h), pd.Series(l)
    win_max = hs.rolling(right + 1, min_periods=right + 1).max().to_numpy()  # max h[t-right .. t]
    win_min = ls.rolling(right + 1, min_periods=right + 1).min().to_numpy()
    cand_hi, cand_lo = shift(h, right, np.nan), shift(l, right, np.nan)  # h[s], l[s] seen from t = s + right
    left_hi, left_lo = shift(prev_max(h, left), right, np.nan), shift(prev_min(l, left), right, np.nan)
    is_hi = (cand_hi >= win_max) & (cand_hi > left_hi)
    is_lo = (cand_lo <= win_min) & (cand_lo < left_lo)
    pos = np.arange(n)
    conf_hi = np.maximum.accumulate(np.where(is_hi, pos, -1))
    conf_lo = np.maximum.accumulate(np.where(is_lo, pos, -1))
    level_hi = np.where(conf_hi >= 0, cand_hi[np.maximum(conf_hi, 0)], np.nan)
    level_lo = np.where(conf_lo >= 0, cand_lo[np.maximum(conf_lo, 0)], np.nan)
    return level_hi, conf_hi, level_lo, conf_lo


def structure_breaks(h: np.ndarray, l: np.ndarray, c: np.ndarray, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """First close beyond the most recent confirmed swing.

    Returns (brk, kind): brk = +1 (close above the swing high in force, first
    such close since that swing was confirmed), -1 (mirror), 0 otherwise; a bar
    that breaks both is dropped. kind = +1 when the break has the same direction
    as the previous break (BOS proper), -1 when it has the opposite direction
    (change of character), 0 when there is no previous break.
    """
    level_hi, conf_hi, level_lo, conf_lo = confirmed_swings(h, l, left, right)
    above = (conf_hi >= 0) & (c > level_hi)
    below = (conf_lo >= 0) & (c < level_lo)
    cum_a, cum_b = np.cumsum(above), np.cumsum(below)
    first_a = above & ((cum_a - cum_a[np.maximum(conf_hi, 0)]) == 1)
    first_b = below & ((cum_b - cum_b[np.maximum(conf_lo, 0)]) == 1)
    brk = first_a.astype(np.int8) - first_b.astype(np.int8)
    # direction of the most recent break strictly before t
    pos = np.arange(len(c))
    last = np.maximum.accumulate(np.where(brk != 0, pos, -1))
    last_dir = np.where(last >= 0, brk[np.maximum(last, 0)], 0)
    prev_dir = shift(last_dir, 1, 0)
    kind = (brk * prev_dir).astype(np.int8)
    return brk, kind


# --------------------------------------------------------------------------- #
# 3. Fair value gaps and returns into a zone
# --------------------------------------------------------------------------- #

def fair_value_gaps(h: np.ndarray, l: np.ndarray, m: float, ref_n: int = 60):
    """Three-bar imbalance at bar t: l[t] > h[t-2] (bullish) or h[t] < l[t-2] (bearish).

    The gap must be at least m x the mean 1-minute range of the ref_n bars that
    END three bars earlier (so the pattern does not inflate its own yardstick).
    Returns (sig, near, far): near = the edge price first met when price comes
    back (top of a bullish gap), far = the other edge.
    """
    ref = shift(pd.Series(h - l).rolling(ref_n, min_periods=ref_n).mean().to_numpy(), 3, np.nan)
    h2, l2 = shift(h, 2, np.nan), shift(l, 2, np.nan)
    up = (l > h2) & ((l - h2) >= m * ref) & (ref > 0)
    dn = (h < l2) & ((l2 - h) >= m * ref) & (ref > 0)
    sig = up.astype(np.int8) - dn.astype(np.int8)
    near = np.where(up, l, np.where(dn, h, np.nan))
    far = np.where(up, h2, np.where(dn, l2, np.nan))
    return sig, near, far


def zone_first_touch(h: np.ndarray, l: np.ndarray, s: np.ndarray, near: np.ndarray, direction: np.ndarray,
                     wait: int, strict: bool) -> np.ndarray:
    """For each zone created at bar s: the first bar in s+1 .. s+wait that trades back to the near edge (-1 if none).

    Bullish zone: low <= near (strict: low < near, the condition under which a
    resting buy order at ``near`` is certainly filled). Bearish zone: mirror.
    """
    n = len(h)
    touch = np.full(len(s), -1, dtype=np.int64)
    pending = np.arange(len(s))
    for j in range(1, wait + 1):
        if len(pending) == 0:
            break
        t = s[pending] + j
        inside = t < n
        pending, t = pending[inside], t[inside]
        if strict:
            hit = np.where(direction[pending] > 0, l[t] < near[pending], h[t] > near[pending])
        else:
            hit = np.where(direction[pending] > 0, l[t] <= near[pending], h[t] >= near[pending])
        touch[pending[hit]] = t[hit]
        pending = pending[~hit]
    return touch


def zone_retest_taker(h, l, c, s, near, far, direction, wait: int) -> np.ndarray:
    """Signal at the first touch of each zone, kept only if that bar did not CLOSE beyond the far edge.

    Several zones touched in the same bar count once; opposite zones cancel.
    """
    touch = zone_first_touch(h, l, s, near, direction, wait, strict=False)
    got = touch >= 0
    t, d, fr = touch[got], direction[got], far[got]
    alive = np.where(d > 0, c[t] > fr, c[t] < fr)
    score = np.zeros(len(c), dtype=np.int64)
    np.add.at(score, t[alive], d[alive].astype(np.int64))
    return np.sign(score).astype(np.int8)


def zone_retest_maker(h, l, s, near, direction, wait: int):
    """Resting limit order at the near edge of each zone: (fill bar, direction, fill price) for the zones that fill."""
    touch = zone_first_touch(h, l, s, near, direction, wait, strict=True)
    got = touch >= 0
    order = np.argsort(touch[got], kind="stable")
    return touch[got][order], direction[got][order].astype(np.int8), near[got][order]


def order_blocks(o, h, l, c, brk: np.ndarray, max_age: int = 30):
    """Zone = range of the last opposite-colour candle before a structure break at bar b.

    Bullish break: the last candle with close < open among bars b-max_age .. b-1;
    near = its high, far = its low; the break bar must have closed above the zone.
    Returns (b, near, far, direction).
    """
    pos = np.arange(len(c))
    last_down = np.maximum.accumulate(np.where(c < o, pos, -1))
    last_up = np.maximum.accumulate(np.where(c > o, pos, -1))
    b = np.flatnonzero(brk)
    b = b[b >= 1]
    d = brk[b].astype(np.int8)
    u = np.where(d > 0, last_down[b - 1], last_up[b - 1])
    ok = (u >= 0) & ((b - u) <= max_age)
    b, d, u = b[ok], d[ok], u[ok]
    near = np.where(d > 0, h[u], l[u])
    far = np.where(d > 0, l[u], h[u])
    clear = np.where(d > 0, c[b] > near, c[b] < near)
    return b[clear], near[clear], far[clear], d[clear]


# --------------------------------------------------------------------------- #
# 4. Compression then breakout; 5. displacement
# --------------------------------------------------------------------------- #

def compression(h, l, m_bars: int):
    """Box of the bars t-M .. t-1 (high, low) and its relative width (high - low) / low."""
    hi, lo = prev_max(h, m_bars), prev_min(l, m_bars)
    width = (hi - lo) / lo
    return hi, lo, width


def compression_breakout(h, l, c, v, m_bars: int, q: float, vol_mult: float | None,
                         rank_window: int = MINUTES_PER_DAY, _cache: dict | None = None) -> np.ndarray:
    """+1 when the M-minute box is tight and bar t closes above it, -1 when it closes below.

    Tight = the box width (high - low) / low is at or below the q-quantile of the
    box widths of the trailing ``rank_window`` bars. Volume filter: v[t] >=
    vol_mult x mean volume of the box bars.
    """
    key = ("box", m_bars)
    if _cache is not None and key in _cache:
        hi, lo, width = _cache[key]
    else:
        hi, lo, width = compression(h, l, m_bars)
        if _cache is not None:
            _cache[key] = (hi, lo, width)
    qkey = ("thr", m_bars, q)
    if _cache is not None and qkey in _cache:
        thr = _cache[qkey]
    else:
        thr = pd.Series(width).rolling(rank_window, min_periods=rank_window).quantile(q).to_numpy()
        if _cache is not None:
            _cache[qkey] = thr
    tight = width <= thr
    up = tight & (c > hi)
    dn = tight & (c < lo)
    if vol_mult is not None:
        big = v >= vol_mult * prev_mean(v, m_bars)
        up &= big
        dn &= big
    return up.astype(np.int8) - dn.astype(np.int8)


def displacement(o, h, l, c, m: float, ref_n: int = 60, close_frac: float = 0.75) -> np.ndarray:
    """+1 for a bar whose body (close - open) is >= m x the mean absolute body of the previous ref_n bars and that closes in the top quarter of its range; -1 mirror."""
    body = c - o
    ref = prev_mean(np.abs(body), ref_n)
    rng = h - l
    up = (body >= m * ref) & (ref > 0) & ((c - l) >= close_frac * rng)
    dn = (-body >= m * ref) & (ref > 0) & ((h - c) >= close_frac * rng)
    return up.astype(np.int8) - dn.astype(np.int8)


# --------------------------------------------------------------------------- #
# The grid: every definition that is examined (70 definitions)
# --------------------------------------------------------------------------- #

SWEEP_N = (30, 60, 240)
SWEEP_X = (2, 10)
SWEEP_K = (1, 3)
MATURE_AGE = 10
SWING_SIZES = (3, 5, 10)
FVG_M = (0.5, 1.0, 2.0)
FILL_WAIT = (60, 240)
SQUEEZE_M = (30, 120)
SQUEEZE_Q = (0.10, 0.25)
DISP_M = (3.0, 6.0)
OB_SWING = 5


def _fmt(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def family_sweeps(bars: Bars) -> Iterator[EventSet]:
    h, l, c = bars.h, bars.l, bars.c
    for n in SWEEP_N:
        lev_hi, lev_lo = prev_max(h, n), prev_min(l, n)
        for x in SWEEP_X:
            for k in SWEEP_K:
                idx, d = from_signal(sweep_reject(h, l, c, lev_hi, lev_lo, x, k))
                yield EventSet(f"sweep_N{n}_x{x}_k{k}", "sweep", {"N": n, "x_bp": x, "k": k},
                               "reversal after a sweep of the prior N-minute extreme", idx, d)
                idx, d = from_signal(break_hold(h, l, c, lev_hi, lev_lo, x, k))
                yield EventSet(f"hold_N{n}_x{x}_k{k}", "hold", {"N": n, "x_bp": x, "k": k},
                               "continuation after k closes beyond the prior N-minute extreme", idx, d)
        if n in (60, 240):
            # the swept level must be at least MATURE_AGE minutes old
            ok_hi = prev_max(h, MATURE_AGE) < lev_hi
            ok_lo = prev_min(l, MATURE_AGE) > lev_lo
            for k in SWEEP_K:
                idx, d = from_signal(sweep_reject(h, l, c, lev_hi, lev_lo, 2, k, ok_hi, ok_lo))
                yield EventSet(f"sweepmat_N{n}_x2_k{k}", "sweepmat", {"N": n, "x_bp": 2, "k": k, "age": MATURE_AGE},
                               "reversal after a sweep of an N-minute extreme that is at least 10 minutes old", idx, d)


def family_prevday(bars: Bars) -> Iterator[EventSet]:
    h, l, c = bars.h, bars.l, bars.c
    lev_hi, lev_lo = previous_day_levels(bars.minute, h, l, bars.valid)
    for x in SWEEP_X:
        for k in SWEEP_K:
            idx, d = from_signal(sweep_reject(h, l, c, lev_hi, lev_lo, x, k))
            yield EventSet(f"pdsweep_x{x}_k{k}", "pdsweep", {"x_bp": x, "k": k},
                           "reversal after a sweep of the previous day's high / low", idx, d)
            idx, d = from_signal(break_hold(h, l, c, lev_hi, lev_lo, x, k))
            yield EventSet(f"pdhold_x{x}_k{k}", "pdhold", {"x_bp": x, "k": k},
                           "continuation after k closes beyond the previous day's high / low", idx, d)


def family_structure(bars: Bars) -> Iterator[EventSet]:
    o, h, l, c = bars.o, bars.h, bars.l, bars.c
    for size in SWING_SIZES:
        brk, kind = structure_breaks(h, l, c, size, size)
        idx, d = from_signal(brk)
        yield EventSet(f"bos_s{size}_all", "bos", {"left": size, "right": size, "kind": "all"},
                       "continuation after the first close beyond the latest confirmed swing", idx, d)
        sel = kind[idx] > 0
        yield EventSet(f"bos_s{size}_with", "bos", {"left": size, "right": size, "kind": "with"},
                       "continuation after a break in the direction of the previous break (BOS)", idx[sel], d[sel])
        sel = kind[idx] < 0
        yield EventSet(f"bos_s{size}_against", "bos", {"left": size, "right": size, "kind": "against"},
                       "continuation after a break against the previous break (change of character)", idx[sel], d[sel])
        if size == OB_SWING:
            b, near, far, dz = order_blocks(o, h, l, c, brk)
            for wait in FILL_WAIT:
                idx2, d2 = from_signal(zone_retest_taker(h, l, c, b, near, far, dz, wait))
                yield EventSet(f"ob_W{wait}", "ob", {"swing": size, "wait": wait},
                               "bounce on the first return into the order block", idx2, d2)
            t, d3, px = zone_retest_maker(h, l, b, near, dz, 60)
            yield EventSet("oblimit_W60", "oblimit", {"swing": size, "wait": 60},
                           "bounce; resting limit order at the order-block edge", t, d3, px)


def family_fvg(bars: Bars) -> Iterator[EventSet]:
    h, l, c = bars.h, bars.l, bars.c
    for m in FVG_M:
        sig, near, far = fair_value_gaps(h, l, m)
        s, d = from_signal(sig)
        yield EventSet(f"fvg_m{_fmt(m)}", "fvg", {"m": m}, "continuation after a fair value gap", s, d)
        for wait in FILL_WAIT:
            idx, d2 = from_signal(zone_retest_taker(h, l, c, s, near[s], far[s], d, wait))
            yield EventSet(f"fvgfill_m{_fmt(m)}_W{wait}", "fvgfill", {"m": m, "wait": wait},
                           "bounce on the first return into the gap", idx, d2)
        t, d3, px = zone_retest_maker(h, l, s, near[s], d, 60)
        yield EventSet(f"fvglimit_m{_fmt(m)}_W60", "fvglimit", {"m": m, "wait": 60},
                       "bounce; resting limit order at the gap edge", t, d3, px)


def family_squeeze(bars: Bars) -> Iterator[EventSet]:
    h, l, c, v = bars.h, bars.l, bars.c, bars.v
    cache: dict = {}
    for m_bars in SQUEEZE_M:
        for q in SQUEEZE_Q:
            for vol_mult in (None, 2.0):
                sig = compression_breakout(h, l, c, v, m_bars, q, vol_mult, _cache=cache)
                idx, d = from_signal(sig)
                yield EventSet(f"squeeze_M{m_bars}_q{_fmt(q)}_v{0 if vol_mult is None else 2}", "squeeze",
                               {"M": m_bars, "q": q, "vol_mult": vol_mult},
                               "continuation after a close outside a compressed range", idx, d)


def family_displacement(bars: Bars) -> Iterator[EventSet]:
    for m in DISP_M:
        idx, d = from_signal(displacement(bars.o, bars.h, bars.l, bars.c, m))
        yield EventSet(f"disp_m{_fmt(m)}", "disp", {"m": m}, "continuation after a displacement candle", idx, d)


FAMILIES: dict[str, Callable[[Bars], Iterator[EventSet]]] = {
    "sweeps": family_sweeps,
    "prevday": family_prevday,
    "structure": family_structure,
    "fvg": family_fvg,
    "squeeze": family_squeeze,
    "displacement": family_displacement,
}

# which generator produces which def_id prefix
PREFIX_TO_GENERATOR = {
    "sweep": "sweeps", "sweepmat": "sweeps", "hold": "sweeps",
    "pdsweep": "prevday", "pdhold": "prevday",
    "bos": "structure", "ob": "structure", "oblimit": "structure",
    "fvg": "fvg", "fvgfill": "fvg", "fvglimit": "fvg",
    "squeeze": "squeeze", "disp": "displacement",
}


def detect_all(bars: Bars) -> Iterator[EventSet]:
    for generator in FAMILIES.values():
        yield from generator(bars)


def detect(bars: Bars, def_ids) -> dict[str, EventSet]:
    """Only the requested definitions (runs just the generators that are needed)."""
    wanted = set(def_ids)
    needed = {PREFIX_TO_GENERATOR[d.split("_")[0]] for d in wanted}
    out = {}
    for name in needed:
        for events in FAMILIES[name](bars):
            if events.def_id in wanted:
                out[events.def_id] = events
    missing = wanted - set(out)
    if missing:
        raise KeyError(f"unknown definitions: {sorted(missing)}")
    return out


def signal_array(events: EventSet, n: int) -> np.ndarray:
    """Dense +1 / -1 / 0 array of a next-open definition."""
    sig = np.zeros(n, dtype=np.int8)
    sig[events.idx] = events.direction
    return sig
