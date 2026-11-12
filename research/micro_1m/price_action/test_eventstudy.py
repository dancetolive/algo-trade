"""Synthetic-data proofs for the event-study module and the structure detectors.

Run:  /home/user/algotrade/.venv/bin/python -I -B test_eventstudy.py
(or with pytest: python -I -B -m pytest -p no:cacheprovider test_eventstudy.py)

What is proved
--------------
1. No look-ahead: for every one of the 70 definitions, the events stamped at or
   before bar t are identical when all bars after t are replaced by different
   data, and when the series is truncated at t.
2. Each vectorised detector equals a slow, literal re-implementation of its
   written definition (loops, no shifts), including ties from tick rounding.
3. Forward returns run from the open of t+1 to the open of t+1+h and are NaN if
   any bar in between is missing.
4. A planted effect of known size is recovered through the whole chain
   (detector -> forward returns -> baseline -> day-clustered statistics).
5. The day-clustered t-statistic is calibrated under the null when events
   cluster and horizons overlap; the naive one is badly inflated.
6. The matched baseline removes an hour-of-day drift.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eventstudy as es  # noqa: E402
import structures as st  # noqa: E402

START_MINUTE = 26_300_000 - (26_300_000 % 1440)  # a UTC midnight


# --------------------------------------------------------------------------- #
# Synthetic bars
# --------------------------------------------------------------------------- #

def synth(n: int, seed: int, tick: float | None = 0.5, base: float = 10_000.0, start_minute: int = START_MINUTE) -> es.Bars:
    """Random walk with stochastic volatility built from 12 sub-steps per minute; optional tick rounding."""
    rng = np.random.default_rng(seed)
    log_vol = np.zeros(n)
    shocks = rng.normal(size=n) * 0.12
    for i in range(1, n):
        log_vol[i] = 0.985 * log_vol[i - 1] + shocks[i]
    sigma = 3.0 * np.exp(log_vol)
    steps = rng.normal(size=(n, 12)) * sigma[:, None]
    path = base + np.cumsum(steps.ravel()).reshape(n, 12)
    if tick:
        path = np.round(path / tick) * tick
    o, c = path[:, 0].copy(), path[:, -1].copy()
    h, l = path.max(axis=1), path.min(axis=1)
    v = rng.lognormal(0.0, 0.6, size=n) * (1.0 + (h - l) / 20.0)
    return es.Bars("SYN", "test", start_minute + np.arange(n, dtype=np.int64), o, h, l, c, v, np.ones(n, dtype=bool))


def splice(bars: es.Bars, cut: int, seed: int) -> es.Bars:
    """Keep bars 0..cut, replace everything after by a different path (prices AND volumes change)."""
    n = bars.n
    other = synth(n, seed, tick=0.5, base=float(bars.c[cut]) * 1.003)
    out = {}
    for name in ("o", "h", "l", "c", "v"):
        a = getattr(bars, name).copy()
        a[cut + 1:] = getattr(other, name)[cut + 1:]
        out[name] = a
    return es.Bars(bars.symbol, bars.market, bars.minute, out["o"], out["h"], out["l"], out["c"], out["v"], bars.valid.copy())


def events_until(bars: es.Bars, cut: int) -> dict:
    found = {}
    for ev in st.detect_all(bars):
        keep = ev.idx <= cut
        px = None if ev.entry_px is None else ev.entry_px[keep]
        found[ev.def_id] = (ev.idx[keep], ev.direction[keep], px)
    return found


# --------------------------------------------------------------------------- #
# 1. No look-ahead
# --------------------------------------------------------------------------- #

def test_no_lookahead_all_definitions():
    bars = synth(9_000, seed=1)
    full = {ev.def_id: ev for ev in st.detect_all(bars)}
    assert len(full) == 70, len(full)
    empty = [k for k, ev in full.items() if len(ev.idx) < 5]
    assert not empty, f"definitions that never fire on the synthetic data (test would be vacuous): {empty}"
    for cut in (4_321, 6_000, 7_777, 8_640):
        reference = {k: (ev.idx[ev.idx <= cut], ev.direction[ev.idx <= cut],
                         None if ev.entry_px is None else ev.entry_px[ev.idx <= cut]) for k, ev in full.items()}
        for label, other in (("altered future", splice(bars, cut, seed=100 + cut)), ("truncated", bars.cut(cut + 1))):
            got = events_until(other, cut)
            for key, (idx, direction, px) in reference.items():
                g_idx, g_dir, g_px = got[key]
                assert np.array_equal(idx, g_idx), f"{key}: event bars change when the future is {label} (cut {cut})"
                assert np.array_equal(direction, g_dir), f"{key}: directions change ({label}, cut {cut})"
                if px is not None:
                    assert np.array_equal(px, g_px), f"{key}: fill prices change ({label}, cut {cut})"


def test_lookahead_check_catches_a_cheat():
    """The harness itself must fail for a detector that peeks one bar ahead."""
    bars = synth(3_000, seed=2)
    cut = 2_000

    def cheat(b):
        return np.flatnonzero(np.append(b.c[1:] > b.c[:-1], False))  # "buy before an up bar"

    assert not np.array_equal(cheat(bars)[cheat(bars) <= cut], cheat(splice(bars, cut, 5))[cheat(splice(bars, cut, 5)) <= cut])


# --------------------------------------------------------------------------- #
# 2. Literal reference implementations
# --------------------------------------------------------------------------- #

def ref_prev(x, n, fn):
    out = np.full(len(x), np.nan)
    for t in range(n, len(x)):
        out[t] = fn(x[t - n:t])
    return out


def ref_sweep_reject(h, l, c, lev_hi, lev_lo, x, k, age=None):
    n = len(c)
    sig = np.zeros(n, dtype=np.int8)
    for t in range(1, n):
        bull = bear = False
        for s in range(max(t - k + 1, 1), t + 1):
            if np.isfinite(lev_hi[s]) and h[s] >= lev_hi[s] * (1.0 + x * 1e-4) and c[s - 1] <= lev_hi[s]:
                mature = age is None or (s - age >= 0 and max(h[s - age:s]) < lev_hi[s])
                if mature and all(c[u] >= lev_hi[s] for u in range(s, t)) and c[t] < lev_hi[s]:
                    bear = True
            if np.isfinite(lev_lo[s]) and l[s] <= lev_lo[s] * (1.0 - x * 1e-4) and c[s - 1] >= lev_lo[s]:
                mature = age is None or (s - age >= 0 and min(l[s - age:s]) > lev_lo[s])
                if mature and all(c[u] <= lev_lo[s] for u in range(s, t)) and c[t] > lev_lo[s]:
                    bull = True
        sig[t] = int(bull) - int(bear)
    return sig


def ref_break_hold(h, l, c, lev_hi, lev_lo, x, k):
    n = len(c)
    sig = np.zeros(n, dtype=np.int8)
    for t in range(k, n):
        s = t - k + 1
        up = np.isfinite(lev_hi[s]) and h[s] >= lev_hi[s] * (1.0 + x * 1e-4) and c[s - 1] <= lev_hi[s] \
            and all(c[u] > lev_hi[s] for u in range(s, t + 1))
        dn = np.isfinite(lev_lo[s]) and l[s] <= lev_lo[s] * (1.0 - x * 1e-4) and c[s - 1] >= lev_lo[s] \
            and all(c[u] < lev_lo[s] for u in range(s, t + 1))
        sig[t] = int(up) - int(dn)
    return sig


def ref_structure_breaks(h, l, c, left, right):
    """State machine, the way the rule is described in words."""
    n = len(c)
    brk = np.zeros(n, dtype=np.int8)
    kind = np.zeros(n, dtype=np.int8)
    lev_hi = lev_lo = None
    hi_broken = lo_broken = True
    prev = 0
    for t in range(n):
        s = t - right
        if s - left >= 0:
            if h[s] > max(h[s - left:s]) and h[s] >= max(h[s + 1:t + 1]):
                lev_hi, hi_broken = h[s], False
            if l[s] < min(l[s - left:s]) and l[s] <= min(l[s + 1:t + 1]):
                lev_lo, lo_broken = l[s], False
        above = lev_hi is not None and c[t] > lev_hi
        below = lev_lo is not None and c[t] < lev_lo
        first_above = above and not hi_broken
        first_below = below and not lo_broken
        hi_broken = hi_broken or above
        lo_broken = lo_broken or below
        d = int(first_above) - int(first_below)
        if d != 0:
            brk[t] = d
            kind[t] = d * prev
            prev = d
    return brk, kind


def ref_fvg(h, l, m, ref_n=60):
    n = len(h)
    sig = np.zeros(n, dtype=np.int8)
    near = np.full(n, np.nan)
    far = np.full(n, np.nan)
    for t in range(ref_n + 2, n):
        ref = float(np.mean(h[t - 2 - ref_n:t - 2] - l[t - 2 - ref_n:t - 2]))  # bars t-62 .. t-3
        if ref <= 0:
            continue
        if l[t] > h[t - 2] and l[t] - h[t - 2] >= m * ref:
            sig[t], near[t], far[t] = 1, l[t], h[t - 2]
        elif h[t] < l[t - 2] and l[t - 2] - h[t] >= m * ref:
            sig[t], near[t], far[t] = -1, h[t], l[t - 2]
    return sig, near, far


def ref_zone_taker(h, l, c, s, near, far, d, wait):
    n = len(c)
    score = np.zeros(n, dtype=np.int64)
    for i in range(len(s)):
        for t in range(s[i] + 1, min(s[i] + wait, n - 1) + 1):
            if (d[i] > 0 and l[t] <= near[i]) or (d[i] < 0 and h[t] >= near[i]):
                if (d[i] > 0 and c[t] > far[i]) or (d[i] < 0 and c[t] < far[i]):
                    score[t] += d[i]
                break
    return np.sign(score).astype(np.int8)


def ref_zone_maker(h, l, s, near, d, wait):
    n = len(h)
    fills = []
    for i in range(len(s)):
        for t in range(s[i] + 1, min(s[i] + wait, n - 1) + 1):
            if (d[i] > 0 and l[t] < near[i]) or (d[i] < 0 and h[t] > near[i]):
                fills.append((t, i, int(d[i]), float(near[i])))
                break
    fills.sort(key=lambda r: (r[0], r[1]))
    return (np.array([f[0] for f in fills], dtype=np.int64), np.array([f[2] for f in fills], dtype=np.int8),
            np.array([f[3] for f in fills]))


def ref_order_blocks(o, h, l, c, brk, max_age=30):
    rows = []
    for b in np.flatnonzero(brk):
        d = int(brk[b])
        u = None
        for j in range(b - 1, max(b - max_age, 0) - 1, -1):
            if (d > 0 and c[j] < o[j]) or (d < 0 and c[j] > o[j]):
                u = j
                break
        if u is None:
            continue
        near, far = (h[u], l[u]) if d > 0 else (l[u], h[u])
        if (d > 0 and c[b] > near) or (d < 0 and c[b] < near):
            rows.append((b, near, far, d))
    return (np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]), np.array([r[3] for r in rows], dtype=np.int8))


def ref_squeeze(h, l, c, v, m_bars, q, vol_mult, rank_window):
    n = len(c)
    hi, lo = ref_prev(h, m_bars, max), ref_prev(l, m_bars, min)
    width = (hi - lo) / lo
    sig = np.zeros(n, dtype=np.int8)
    for t in range(m_bars + rank_window - 1, n):
        thr = np.quantile(width[t - rank_window + 1:t + 1], q)
        if not width[t] <= thr:
            continue
        if vol_mult is not None and not v[t] >= vol_mult * np.mean(v[t - m_bars:t]):
            continue
        sig[t] = int(c[t] > hi[t]) - int(c[t] < lo[t])
    return sig


def ref_displacement(o, h, l, c, m, ref_n=60, close_frac=0.75):
    n = len(c)
    sig = np.zeros(n, dtype=np.int8)
    for t in range(ref_n, n):
        ref = float(np.mean(np.abs(c[t - ref_n:t] - o[t - ref_n:t])))
        if ref <= 0:
            continue
        body, rng = c[t] - o[t], h[t] - l[t]
        if body >= m * ref and c[t] - l[t] >= close_frac * rng:
            sig[t] = 1
        elif -body >= m * ref and h[t] - c[t] >= close_frac * rng:
            sig[t] = -1
    return sig


def ref_prev_day(minute, h, l):
    day = minute // 1440
    hi = np.full(len(h), np.nan)
    lo = np.full(len(h), np.nan)
    for d in np.unique(day):
        prev = day == d - 1
        if prev.sum() == 1440:
            hi[day == d], lo[day == d] = h[prev].max(), l[prev].min()
    return hi, lo


def test_sweeps_and_holds_match_reference():
    bars = synth(4_500, seed=3)
    h, l, c = bars.h, bars.l, bars.c
    total = 0
    for n in (30, 240):
        lev_hi, lev_lo = st.prev_max(h, n), st.prev_min(l, n)
        assert np.array_equal(lev_hi, ref_prev(h, n, max), equal_nan=True)
        assert np.array_equal(lev_lo, ref_prev(l, n, min), equal_nan=True)
        for x in (2, 10):
            for k in (1, 3):
                got = st.sweep_reject(h, l, c, lev_hi, lev_lo, x, k)
                assert np.array_equal(got, ref_sweep_reject(h, l, c, lev_hi, lev_lo, x, k)), (n, x, k)
                total += int(np.abs(got).sum())
                got = st.break_hold(h, l, c, lev_hi, lev_lo, x, k)
                assert np.array_equal(got, ref_break_hold(h, l, c, lev_hi, lev_lo, x, k)), (n, x, k)
                total += int(np.abs(got).sum())
        ok_hi, ok_lo = st.prev_max(h, st.MATURE_AGE) < lev_hi, st.prev_min(l, st.MATURE_AGE) > lev_lo
        for k in (1, 3):
            got = st.sweep_reject(h, l, c, lev_hi, lev_lo, 2, k, ok_hi, ok_lo)
            assert np.array_equal(got, ref_sweep_reject(h, l, c, lev_hi, lev_lo, 2, k, age=st.MATURE_AGE)), (n, k)
            total += int(np.abs(got).sum())
    assert total > 500


def test_previous_day_levels_match_reference():
    bars = synth(1440 * 5 + 300, seed=4, start_minute=START_MINUTE + 700)  # starts mid-day: first day incomplete
    hi, lo = st.previous_day_levels(bars.minute, bars.h, bars.l, bars.valid)
    r_hi, r_lo = ref_prev_day(bars.minute, bars.h, bars.l)
    assert np.array_equal(hi, r_hi, equal_nan=True) and np.array_equal(lo, r_lo, equal_nan=True)
    assert np.isnan(hi[: 1440 - 700 + 1440]).all()  # the incomplete first day gives no level for day two
    got = st.sweep_reject(bars.h, bars.l, bars.c, hi, lo, 2, 3)
    assert np.array_equal(got, ref_sweep_reject(bars.h, bars.l, bars.c, hi, lo, 2, 3))
    got = st.break_hold(bars.h, bars.l, bars.c, hi, lo, 2, 3)
    assert np.array_equal(got, ref_break_hold(bars.h, bars.l, bars.c, hi, lo, 2, 3))


def test_structure_breaks_match_reference():
    bars = synth(5_000, seed=5)  # tick-rounded: equal highs and lows occur
    assert (np.diff(bars.h) == 0).sum() > 20
    for size in (3, 5, 10):
        brk, kind = st.structure_breaks(bars.h, bars.l, bars.c, size, size)
        r_brk, r_kind = ref_structure_breaks(bars.h, bars.l, bars.c, size, size)
        assert np.array_equal(brk, r_brk), size
        assert np.array_equal(kind, r_kind), size
        assert (brk != 0).sum() > 50 and (kind > 0).sum() > 10 and (kind < 0).sum() > 10
    brk, _ = st.structure_breaks(bars.h, bars.l, bars.c, 2, 4)  # asymmetric left / right
    assert np.array_equal(brk, ref_structure_breaks(bars.h, bars.l, bars.c, 2, 4)[0])


def test_fvg_zones_and_order_blocks_match_reference():
    cont = synth(4_000, seed=6, tick=None)  # continuous prices: no ties in the size threshold
    for m in (0.5, 1.0, 2.0):
        sig, near, far = st.fair_value_gaps(cont.h, cont.l, m)
        r_sig, r_near, r_far = ref_fvg(cont.h, cont.l, m)
        assert np.array_equal(sig, r_sig), m
        assert np.allclose(near, r_near, equal_nan=True) and np.allclose(far, r_far, equal_nan=True)
    sig, near, far = st.fair_value_gaps(cont.h, cont.l, 0.5)
    s, d = st.from_signal(sig)
    assert len(s) > 40
    for bars in (cont, synth(4_000, seed=7)):
        sig, near, far = st.fair_value_gaps(bars.h, bars.l, 0.5)
        s, d = st.from_signal(sig)
        for wait in (5, 60):
            got = st.zone_retest_taker(bars.h, bars.l, bars.c, s, near[s], far[s], d, wait)
            assert np.array_equal(got, ref_zone_taker(bars.h, bars.l, bars.c, s, near[s], far[s], d, wait)), wait
            t, dd, px = st.zone_retest_maker(bars.h, bars.l, s, near[s], d, wait)
            r_t, r_d, r_px = ref_zone_maker(bars.h, bars.l, s, near[s], d, wait)
            assert np.array_equal(t, r_t) and np.array_equal(dd, r_d) and np.array_equal(px, r_px)
            assert len(t) > 10
            # a resting buy is filled strictly below its price, a resting sell strictly above
            assert np.all(np.where(dd > 0, bars.l[t] < px, bars.h[t] > px))
        brk, _ = st.structure_breaks(bars.h, bars.l, bars.c, 5, 5)
        b, nr, fr, dz = st.order_blocks(bars.o, bars.h, bars.l, bars.c, brk)
        r_b, r_nr, r_fr, r_dz = ref_order_blocks(bars.o, bars.h, bars.l, bars.c, brk)
        assert np.array_equal(b, r_b) and np.array_equal(nr, r_nr) and np.array_equal(fr, r_fr) and np.array_equal(dz, r_dz)
        assert len(b) > 20


def test_squeeze_and_displacement_match_reference():
    bars = synth(3_000, seed=8, tick=None)
    fired = 0
    for m_bars in (30, 120):
        for q in (0.10, 0.25):
            for vol_mult in (None, 2.0):
                got = st.compression_breakout(bars.h, bars.l, bars.c, bars.v, m_bars, q, vol_mult, rank_window=300)
                ref = ref_squeeze(bars.h, bars.l, bars.c, bars.v, m_bars, q, vol_mult, 300)
                assert np.array_equal(got, ref), (m_bars, q, vol_mult)
                fired += int(np.abs(got).sum())
    assert fired > 30
    for m in (3.0, 6.0):
        got = st.displacement(bars.o, bars.h, bars.l, bars.c, m)
        assert np.array_equal(got, ref_displacement(bars.o, bars.h, bars.l, bars.c, m)), m
    assert np.abs(st.displacement(bars.o, bars.h, bars.l, bars.c, 3.0)).sum() > 10


# --------------------------------------------------------------------------- #
# 3. Forward returns and gaps
# --------------------------------------------------------------------------- #

def test_forward_returns_alignment_and_gaps():
    n = 40
    minute = START_MINUTE + np.arange(n)
    o = 100.0 + np.arange(n, dtype=float)  # open of bar i is 100 + i
    frame = pd.DataFrame({"open": o, "high": o + 1, "low": o - 1, "close": o + 0.5, "volume": 1.0},
                         index=pd.to_datetime(minute * 60, unit="s", utc=True))
    frame = frame.drop(frame.index[[20, 21]])  # the exchange delivered no bars for minutes 20 and 21
    bars = es.bars_from_frame(frame, "SYN", "test")
    assert bars.n == n and not bars.valid[20] and not bars.valid[21] and bars.valid.sum() == n - 2
    fwd = es.forward_returns(bars, horizons=(1, 5))
    # event at t: from open[t+1] to open[t+1+h]
    assert np.isclose(fwd[1][3], (105.0 / 104.0 - 1.0) * 1e4)
    assert np.isclose(fwd[5][3], (109.0 / 104.0 - 1.0) * 1e4)
    # a return may not start in, end in or span the gap
    assert np.isfinite(fwd[1][17]) and np.isnan(fwd[1][18])  # 17 -> opens 18,19 fine; 18 -> opens 19,20: 20 missing
    assert np.isnan(fwd[5][14]) and np.isfinite(fwd[5][13])  # 14 -> open 20 missing; 13 -> opens 14 .. 19
    assert np.isnan(fwd[1][19]) and np.isnan(fwd[1][20]) and np.isnan(fwd[1][21]) and np.isfinite(fwd[1][22])
    assert np.isnan(fwd[5][16])  # 16 -> opens 17 and 22 both exist, but the window spans the gap: must be dropped
    assert np.isnan(fwd[1][n - 2]) and np.isnan(fwd[5][n - 6]) and np.isfinite(fwd[5][n - 7])
    ok = es.context_ok(bars, lookback=5)
    assert not ok[:5].any() and ok[19] and not ok[20:27].any() and ok[27]
    # resting-order variant: fill price given, same gap rule, horizon 0 allowed
    got = es.forward_from_price(bars, np.array([3, 16, 22]), np.array([103.5, 116.0, 122.0]), horizons=(0, 5))
    assert np.isclose(got[0][0], (104.0 / 103.5 - 1.0) * 1e4) and np.isclose(got[5][0], (109.0 / 103.5 - 1.0) * 1e4)
    assert np.isnan(got[5][1]) and np.isfinite(got[5][2])


def test_costs():
    assert es.round_trip_cost_bp("BTCUSDT", "perp_taker") == 12.0
    assert es.round_trip_cost_bp("SOLUSDT", "perp_taker") == 14.0
    assert es.round_trip_cost_bp("ETHUSDT", "perp_maker") == 4.0
    assert es.round_trip_cost_bp("BTCUSDT", "spot_taker") == 22.0
    assert es.round_trip_cost_bp("DOGEUSDT", "spot_taker") == 24.0


# --------------------------------------------------------------------------- #
# 4. Planted effect, recovered through the whole chain
# --------------------------------------------------------------------------- #

def planted_path(n: int, delta_bp: float, seed: int, hold: int = 5):
    """Bar-by-bar generator with a causal planted effect.

    Rare exogenous jumps (10 sigma, random sign) are the only bars that qualify as displacement candles
    with m = 6 (the threshold is about 4.8 sigma, which Gaussian noise does not reach). Each displacement
    candle detected at bar t - by a literal sequential loop inside the generator - is followed by a drift
    of delta_bp per bar, in the direction of the candle, over the bars t+1 .. t+hold. Because the jumps
    are exogenous and have random signs, overlapping windows add zero on average and the expected signed
    forward return is exactly delta_bp * min(h, hold).
    """
    rng = np.random.default_rng(seed)
    sigma = 5e-4  # 5 bp per bar
    eps = rng.normal(size=n)
    jump = rng.random(n) < 0.004
    jump_sign = rng.choice([-1.0, 1.0], size=n)
    wick_up, wick_dn = np.abs(rng.normal(size=n)) * sigma * 0.3, np.abs(rng.normal(size=n)) * sigma * 0.3
    o, h, l, c = (np.empty(n) for _ in range(4))
    drift = np.zeros(n + hold + 1)
    events, price = [], 10_000.0
    bodies = []
    for t in range(n):
        move = drift[t] + sigma * eps[t] + (jump_sign[t] * 10 * sigma if jump[t] else 0.0)
        o[t] = price
        c[t] = price * np.exp(move)
        h[t] = max(o[t], c[t]) * (1 + wick_up[t])
        l[t] = min(o[t], c[t]) * (1 - wick_dn[t])
        price = c[t]
        body = c[t] - o[t]
        if t >= 60:
            ref = float(np.mean(np.abs(bodies[-60:])))
            rng_ = h[t] - l[t]
            d = 0
            if body >= 6.0 * ref and c[t] - l[t] >= 0.75 * rng_:
                d = 1
            elif -body >= 6.0 * ref and h[t] - c[t] >= 0.75 * rng_:
                d = -1
            if d:
                events.append((t, d))
                drift[t + 1:t + 1 + hold] += d * delta_bp * 1e-4
        bodies.append(body)
    bars = es.Bars("SYN", "test", START_MINUTE + np.arange(n, dtype=np.int64), o, h, l, c, np.ones(n), np.ones(n, dtype=bool))
    return bars, np.array([e[0] for e in events]), np.array([e[1] for e in events], dtype=np.int8), jump


def test_planted_effect_is_recovered():
    delta = 2.0
    bars, ev_idx, ev_dir, jump = planted_path(400_000, delta, seed=11)
    events = st.detect(bars, ["disp_m6"])["disp_m6"]
    # the vectorised detector finds exactly the events the sequential generator acted on
    assert np.array_equal(events.idx, ev_idx) and np.array_equal(events.direction, ev_dir)
    assert len(ev_idx) > 1200 and jump[ev_idx].mean() > 0.99  # the events are the exogenous jumps
    fwd = es.forward_returns(bars)
    cell, n_cells = es.baseline_cells(bars)
    base = es.matched_baseline(fwd, cell, n_cells)
    ok = es.context_ok(bars)
    daily = es.event_daily(bars, events.idx, events.direction, fwd, base, ok)
    daily["symbol"] = "SOLUSDT"
    table = es.pooled_stats(daily, cost_bp={"SOLUSDT": es.round_trip_cost_bp("SOLUSDT", "perp_taker")})
    expected = {1: delta, 5: 5 * delta, 15: 5 * delta, 60: 5 * delta}
    for h, want in expected.items():
        row = table.loc[h]
        assert abs(row["excess"] - want) < 3.5 * row["se_excess"], (h, row["excess"], want, row["se_excess"])
        assert abs(row["gross"] - want) < 3.5 * row["se_excess"], (h, row["gross"], want)
    assert table.loc[1, "se_excess"] < 0.2 and table.loc[5, "se_excess"] < 0.4  # so the tolerances above are tight
    assert table.loc[5, "t_excess"] > 20  # a 10 bp effect on > 1000 events is unmistakable
    assert np.isclose(table.loc[5, "net"], table.loc[5, "gross"] - 14.0)
    # Alignment: stamping the event one bar early would swallow the 50 bp displacement candle itself,
    # one bar late would lose one fifth of the planted drift.
    early = np.nanmean(events.direction * fwd[5][events.idx - 1])
    late = np.nanmean(events.direction * fwd[5][events.idx + 1])
    assert early > 45 and abs(late - 4 * delta) < 1.5, (early, late)
    # and with no planted effect nothing is found
    bars0, _, _, _ = planted_path(400_000, 0.0, seed=12)
    ev0 = st.detect(bars0, ["disp_m6"])["disp_m6"]
    fwd0 = es.forward_returns(bars0)
    cell0, k0 = es.baseline_cells(bars0)
    daily0 = es.event_daily(bars0, ev0.idx, ev0.direction, fwd0, es.matched_baseline(fwd0, cell0, k0), es.context_ok(bars0))
    table0 = es.pooled_stats(daily0)
    assert (table0["t_excess"].abs() < 3.5).all(), table0["t_excess"].to_dict()


def test_resting_order_returns_show_adverse_selection():
    """A limit buy that is filled when price trades through it is, on a driftless walk, under water at the next open."""
    bars = synth(30_000, seed=13, tick=None)
    sig, near, far = st.fair_value_gaps(bars.h, bars.l, 0.5)
    s, d = st.from_signal(sig)
    t, dd, px = st.zone_retest_maker(bars.h, bars.l, s, near[s], d, 60)
    raw = es.forward_from_price(bars, t, px, horizons=(0,))
    signed = dd * raw[0]
    assert len(t) > 200 and np.nanmean(signed) < 0


# --------------------------------------------------------------------------- #
# 5. Calibration of the clustered t-statistic
# --------------------------------------------------------------------------- #

def test_cluster_t_is_calibrated_and_naive_is_not():
    rng = np.random.default_rng(21)
    t_cluster, t_naive, t_lag0, t_lag1 = [], [], [], []
    for _ in range(600):
        groups = 250
        size = rng.integers(1, 60, size=groups)
        shock = rng.normal(size=groups)  # common to all events of the day
        total = np.array([size[g] * shock[g] + rng.normal(size=size[g]).sum() for g in range(groups)])
        _, _, t = es.cluster_mean_t(total, size, np.arange(groups))
        t_cluster.append(t)
        n = size.sum()
        # naive: events treated as independent with their pooled variance (2 = day shock + own noise)
        t_naive.append(total.sum() / n / np.sqrt(2.0 / n))
        # adjacent days correlated (overlapping forward returns): MA(1) day shocks
        z = rng.normal(size=groups + 1)
        ma = z[1:] + z[:-1]
        total2 = size * ma
        t_lag0.append(es.cluster_mean_t(total2, size, np.arange(groups), lag=0)[2])
        t_lag1.append(es.cluster_mean_t(total2, size, np.arange(groups), lag=1)[2])
    assert 0.9 < np.std(t_cluster) < 1.12, np.std(t_cluster)
    assert np.std(t_naive) > 3.0
    assert np.std(t_lag0) > 1.25 and np.std(t_lag1) < 1.15, (np.std(t_lag0), np.std(t_lag1))
    assert np.mean(np.abs(t_cluster) > 3) < 0.02


def test_end_to_end_null_with_bursts_and_overlap():
    """Random walk, events in bursts, 60-minute horizon: clustered t ~ N(0,1), naive t far too large."""
    rng = np.random.default_rng(22)
    days, n = 40, 40 * 1440
    minute = START_MINUTE + np.arange(n, dtype=np.int64)
    t_c, t_n = [], []
    for _ in range(150):
        o = 10_000.0 * np.exp(np.cumsum(rng.normal(size=n) * 5e-4))
        bars = es.Bars("SYN", "test", minute, o, o, o, o, np.ones(n), np.ones(n, dtype=bool))
        fwd = es.forward_returns(bars, horizons=(60,))
        starts = rng.integers(0, n - 200, size=days * 6)
        idx = np.unique((starts[:, None] + np.arange(15)[None, :]).ravel())  # bursts of 15 consecutive bars
        base = {60: np.zeros(n)}
        daily = es.event_daily(bars, idx, np.ones(len(idx), dtype=np.int8), fwd, base, np.ones(n, dtype=bool), horizons=(60,))
        row = es.pooled_stats(daily, horizons=(60,)).loc[60]
        t_c.append(row["t_excess"])
        t_n.append(row["t_naive"])
    assert 0.8 < np.std(t_c) < 1.3, np.std(t_c)
    assert np.std(t_n) > 2.5, np.std(t_n)


# --------------------------------------------------------------------------- #
# 6. Baseline
# --------------------------------------------------------------------------- #

def test_matched_baseline_removes_hour_of_day_drift():
    rng = np.random.default_rng(31)
    n = 200 * 1440
    minute = START_MINUTE + np.arange(n, dtype=np.int64)
    hour = (minute % 1440) // 60
    drift = np.where(hour == 3, 0.5e-4, np.where(hour == 15, -0.5e-4, 0.0))  # +/- 0.5 bp per minute
    steps = drift + rng.normal(size=n) * 1e-4
    c = 10_000.0 * np.exp(np.cumsum(steps))
    o = np.concatenate([[10_000.0], c[:-1]])
    bars = es.Bars("SYN", "test", minute, o, np.maximum(o, c), np.minimum(o, c), c, np.ones(n), np.ones(n, dtype=bool))
    fwd = es.forward_returns(bars, horizons=(5,))
    cell, n_cells = es.baseline_cells(bars)
    base = es.matched_baseline(fwd, cell, n_cells)
    ok = es.context_ok(bars)
    # long events at random minutes 0..49 of hour 3: raw gross is 2.5 bp (five drift bars) although the
    # events carry no information. The hour-3 cell average is 2.375 bp (the last five minutes of the hour
    # see fewer drift bars), so the excess is 0.125 bp instead of 2.5 bp.
    idx = np.flatnonzero((hour == 3) & ((minute % 60) < 50) & (rng.random(n) < 0.2) & ok)
    daily = es.event_daily(bars, idx, np.ones(len(idx), dtype=np.int8), fwd, base, ok, horizons=(5,))
    row = es.pooled_stats(daily, horizons=(5,)).loc[5]
    assert row["events"] > 1500
    assert abs(row["gross"] - 2.5) < 0.2, row["gross"]
    assert abs(row["base"] - 2.375) < 0.2, row["base"]
    assert abs(row["excess"] - 0.125) < 0.2 and row["t_gross"] > 15, (row["excess"], row["t_gross"])
    # short events carry the baseline with the opposite sign
    daily_s = es.event_daily(bars, idx, -np.ones(len(idx), dtype=np.int8), fwd, base, ok, horizons=(5,))
    row_s = es.pooled_stats(daily_s, horizons=(5,)).loc[5]
    assert np.isclose(row_s["gross"], -row["gross"]) and np.isclose(row_s["base"], -row["base"])
    # events spread over all hours and directions: baseline ~ 0
    idx_all = np.flatnonzero((rng.random(n) < 0.01) & ok)
    d_all = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(idx_all))
    row_a = es.pooled_stats(es.event_daily(bars, idx_all, d_all, fwd, base, ok, horizons=(5,)), horizons=(5,)).loc[5]
    assert abs(row_a["excess"]) < 0.15 and abs(row_a["t_excess"]) < 3.5


def test_bootstrap_agrees_with_cluster_se():
    rng = np.random.default_rng(41)
    groups = 400
    size = rng.integers(1, 30, size=groups)
    total = size * rng.normal(size=groups) + 0.2 * size
    mean, se, _ = es.cluster_mean_t(total, size, np.arange(groups))
    boot = es.block_bootstrap_p(total, size, np.arange(groups), block=1, n_boot=3000, seed=1)
    assert np.isclose(boot["mean"], mean) and 0.8 < boot["se"] / se < 1.25
    assert boot["p_two_sided"] < 0.05


if __name__ == "__main__":
    import time

    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except AssertionError as exc:
            failed += 1
            import traceback

            print(f"FAIL  {name}: {exc!r}")
            traceback.print_exc(limit=2)
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
