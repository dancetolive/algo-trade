"""Synthetic-data proofs for the event-study module.

Run:  /home/user/algotrade/.venv/bin/python -I -m pytest research/micro_1m/statistical/test_eventstudy.py -q

What is proven here, before any real data is touched:
* forward returns start at the NEXT bar's open and never credit the event bar or the close->open step;
* a forward return whose holding window touches a gap or a halt is dropped;
* no feature and no event flag at bar t changes when bars after t are altered, deleted or halted;
* a planted effect of known size is recovered at the right horizons, and nothing is found without one;
* the day-clustered t-statistic has roughly the nominal false-positive rate where the naive one is wildly inflated;
* the matched baseline removes a pure hour-of-day drift;
* cost arithmetic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import definitions as defs  # noqa: E402
import eventstudy as es  # noqa: E402

COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]


def synth(n, seed=0, sigma=5e-4, start="2021-03-01", drift=None, jump=None, vol_mult=None, heavy=False, vol_sigma=0.5):
    """Random-walk 1-minute bars built from 4 sub-steps per minute.

    drift[t]: extra log return earned inside bar t (open -> close).
    jump[t]:  extra log return between close[t-1] and open[t] (not capturable by an order sent at close[t-1]... or later).
    """
    rng = np.random.default_rng(seed)
    sub = 4
    if heavy:
        steps = rng.standard_t(3, size=(n, sub)) * sigma / np.sqrt(sub) / np.sqrt(3.0)
    else:
        steps = rng.normal(0.0, sigma / np.sqrt(sub), size=(n, sub))
    if drift is not None:
        steps = steps + np.asarray(drift)[:, None] / sub
    jump = np.zeros(n) if jump is None else np.asarray(jump, dtype=float)
    bar_total = steps.sum(axis=1)
    log_open = np.log(100.0) + np.concatenate([[0.0], np.cumsum(bar_total)[:-1]]) + np.cumsum(jump)
    inside = log_open[:, None] + np.cumsum(steps, axis=1)
    open_ = np.exp(log_open)
    close = np.exp(inside[:, -1])
    high = np.maximum(open_, np.exp(inside).max(axis=1))
    low = np.minimum(open_, np.exp(inside).min(axis=1))
    qv = np.exp(rng.normal(10.0, vol_sigma, n))
    if vol_mult is not None:
        qv = qv * np.asarray(vol_mult)
    trades = np.maximum(1, np.round(qv / 150.0)).astype(np.int64)
    share = np.clip(rng.normal(0.5, 0.1, n), 0.02, 0.98)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": qv / close,
            "quote_volume": qv,
            "trades": trades,
            "taker_buy_volume": qv * share / close,
            "taker_buy_quote_volume": qv * share,
        },
        index=pd.date_range(start, periods=n, freq="1min", tz="UTC", name="open_time"),
    )
    return frame[COLUMNS]


def halt(frame, start, length):
    """Turn bars into an exchange halt as Binance prints it: flat, zero volume."""
    frame = frame.copy()
    rows = frame.index[start : start + length]
    price = frame["close"].iloc[start - 1]
    frame.loc[rows, ["open", "high", "low", "close"]] = price
    frame.loc[rows, ["volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]] = 0.0
    frame.loc[rows, "trades"] = 0
    return frame


# --------------------------------------------------------------------------- #
# forward returns and gaps
# --------------------------------------------------------------------------- #


def test_forward_return_starts_at_next_open():
    frame = synth(200, seed=1)
    grid = es.to_grid(frame)
    fwd = es.forward_returns(grid, horizons=(1, 5))
    o = frame["open"].to_numpy()
    t = 50
    assert fwd[1][t] == pytest.approx(o[t + 2] / o[t + 1] - 1.0)
    assert fwd[5][t] == pytest.approx(o[t + 6] / o[t + 1] - 1.0)
    # the last bars have no complete window
    assert np.isnan(fwd[1][-2:]).all() and np.isnan(fwd[5][-6:]).all()
    assert np.isfinite(fwd[1][:-2]).all() and np.isfinite(fwd[5][:-6]).all()


def test_alternative_fills_are_later_or_worse_never_earlier():
    frame = synth(300, seed=21)
    grid = es.to_grid(frame)
    o, hi, lo = (frame[c].to_numpy() for c in ("open", "high", "low"))
    vwap = (frame["quote_volume"] / frame["volume"]).to_numpy()
    t = 100
    assert es.forward_returns(grid, (5,), entry_lag=3)[5][t] == pytest.approx(o[t + 8] / o[t + 3] - 1.0)
    assert es.forward_returns(grid, (5,), entry_price="vwap", exit_price="vwap")[5][t] == pytest.approx(vwap[t + 6] / vwap[t + 1] - 1.0)
    worst_buy = es.forward_returns(grid, (5,), entry_price="high")[5]
    plain = es.forward_returns(grid, (5,))[5]
    ok = np.isfinite(plain)
    assert worst_buy[t] == pytest.approx(o[t + 6] / hi[t + 1] - 1.0)
    assert (worst_buy[ok] <= plain[ok] + 1e-15).all()  # buying at the bar's high is never better than at its open
    worst_sell = es.forward_returns(grid, (5,), entry_price="low")[5]
    assert (worst_sell[ok] >= plain[ok] - 1e-15).all()  # for a short, entering at the low is never better
    with pytest.raises(ValueError):
        es.forward_returns(grid, (5,), entry_lag=0)


def test_move_inside_event_bar_and_close_to_open_step_are_not_credited():
    n = 400
    drift = np.zeros(n)
    jump = np.zeros(n)
    drift[100] = 0.05  # a 5% move inside the event bar itself
    jump[101] = 0.03  # a 3% step from close[100] to open[101]
    quiet = synth(n, seed=2, sigma=1e-6)
    loud = synth(n, seed=2, sigma=1e-6, drift=drift, jump=jump)
    for h in (1, 5, 15):
        a = es.forward_returns(es.to_grid(quiet), (h,))[h][100]
        b = es.forward_returns(es.to_grid(loud), (h,))[h][100]
        assert abs(a) < 1e-4 and abs(b) < 1e-4  # neither the bar's own 5% nor the 3% gap shows up
    # ... while an event one bar earlier would (correctly) have earned the 5% bar: entry at open[100]
    earlier = es.forward_returns(es.to_grid(loud), (1,))[1][99]
    assert earlier == pytest.approx(np.exp(0.05 + 0.03) - 1.0, rel=1e-3)


def test_missing_bars_invalidate_every_window_that_touches_them():
    frame = synth(600, seed=3)
    cut = frame.drop(frame.index[300:320])  # 20 missing bars: grid positions 300..319
    grid = es.to_grid(cut)
    assert len(grid) == 600 and (~grid.exists).sum() == 20 and grid.outage[300:320].all()
    for h in (1, 5, 60):
        fwd = es.forward_returns(grid, (h,))[h]
        # window of t is bars t+1 .. t+1+h; it touches the gap iff t+1 <= 319 and t+1+h >= 300
        touches = np.array([(t + 1 <= 319) and (t + 1 + h >= 300) for t in range(600)])
        complete = np.arange(600) + 1 + h <= 599
        assert np.isnan(fwd[touches]).all()
        assert np.isfinite(fwd[~touches & complete]).all()
    # features never bridge the gap either: the first return after the gap is undefined
    feats = es.trailing_features(grid)
    assert np.isnan(feats["r1"][300:321]).all() and np.isfinite(feats["r1"][321])


def test_halts_printed_as_zero_volume_bars_are_gaps_but_single_quiet_minutes_are_not():
    frame = halt(synth(600, seed=4), 200, 30)  # a 30-minute halt
    frame = halt(frame, 400, 1)  # one quiet minute (no trade), not a halt
    grid = es.to_grid(frame)
    assert grid.outage[200:230].all() and not grid.outage[400]
    assert grid.usable[200:204].all() and not grid.usable[204:230].any()  # recognisable from the 5th bar on
    fwd = es.forward_returns(grid, (1, 15))
    # the window of t is bars t+1 .. t+16: it touches the halt (bars 200..229) for t = 184 .. 228
    assert np.isnan(fwd[15][184:229]).all()
    assert np.isfinite(fwd[15][183]) and np.isfinite(fwd[15][229])
    # the quiet minute may sit inside a window, but nobody can be filled at its open
    assert np.isnan(fwd[1][399]) and np.isnan(fwd[1][398]) and np.isfinite(fwd[15][390])
    assert np.isnan(fwd[15][399])  # entry bar did not trade
    assert np.isnan(fwd[15][384])  # exit bar did not trade


# --------------------------------------------------------------------------- #
# no look-ahead
# --------------------------------------------------------------------------- #


def _rough(n, seed):
    """Fat-tailed bars with gaps, halts, quiet minutes and a few violent moves, so that every code path is exercised."""
    rng = np.random.default_rng(seed + 1000)
    jump = np.zeros(n)
    drift = np.zeros(n)
    for t in rng.integers(1500, n - 5, size=12):
        drift[t] = rng.choice([-1, 1]) * 0.03
    frame = synth(n, seed=seed, heavy=True, vol_sigma=1.2, drift=drift, jump=jump)
    for start in rng.integers(1500, n - 50, size=3):
        frame = halt(frame, int(start), int(rng.integers(5, 25)))
    for start in rng.integers(1500, n - 50, size=10):
        frame = halt(frame, int(start), 1)
    drop = np.concatenate([np.arange(s, s + rng.integers(1, 8)) for s in rng.integers(1500, n - 50, size=3)])
    return frame.drop(frame.index[drop])


def _funding(index, seed):
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(index[0].floor("8h"), index[-1], freq="8h")
    return pd.Series(rng.choice([-5e-4, -1e-4, 1e-4, 4e-4, 0.0], size=len(stamps)), index=stamps)


def _flags(frame, funding):
    grid = es.to_grid(frame)
    feats = es.trailing_features(grid)
    out = {}
    for d in defs.generate(grid, feats, funding):
        side = d.direction if isinstance(d.direction, np.ndarray) else np.full(len(grid), float(d.direction))
        out[d.name] = np.where(d.mask, side, 0.0)  # 0 = no event, +1 / -1 = event and its side
    return grid, feats, out


def test_no_feature_or_event_flag_depends_on_later_bars():
    n = 7000
    base = _rough(n, seed=10)
    funding = _funding(base.index, seed=1)
    grid, feats, flags = _flags(base, funding)
    fired = [name for name, values in flags.items() if (values != 0).any()]
    assert len(fired) >= 0.9 * len(flags), "the synthetic data must exercise nearly every definition"
    assert sum(int((v != 0).sum()) for v in flags.values()) > 20000

    other = _rough(n, seed=77)  # a completely different future, with its own gaps and halts
    other_funding = _funding(other.index, seed=2)
    for cut_time in (base.index[2000], base.index[3333], base.index[5000], base.index[-40]):
        past = base[base.index <= cut_time]
        future = other[other.index > cut_time]
        mixed = pd.concat([past, future])
        mixed_funding = pd.concat([funding[funding.index <= cut_time], other_funding[other_funding.index > cut_time]])
        grid2, feats2, flags2 = _flags(mixed, mixed_funding)
        last = int(grid.index.searchsorted(cut_time, side="right"))  # number of grid bars <= cut
        assert grid.index[last - 1] == cut_time
        for key in feats:
            a, b = feats[key][:last].astype(float), feats2[key][:last].astype(float)
            assert np.array_equal(a, b, equal_nan=True), f"feature {key} changed before the cut"
        for name in flags:
            assert np.array_equal(flags[name][:last], flags2[name][:last]), f"definition {name} changed before the cut"
        assert np.array_equal(grid.usable[:last], grid2.usable[:last])

        # truncation: dropping the future altogether gives the same past as well
        grid3, feats3, flags3 = _flags(past, funding[funding.index <= cut_time])
        assert len(grid3) == last
        for name in flags:
            assert np.array_equal(flags[name][:last], flags3[name]), f"definition {name} changed under truncation"


def test_a_deliberately_leaky_definition_is_caught_by_the_same_check():
    """The check above is only worth something if it can fail: a centred window must trip it."""
    base = _rough(4000, seed=10)
    other = _rough(4000, seed=77)
    cut_time = base.index[2500]
    mixed = pd.concat([base[base.index <= cut_time], other[other.index > cut_time]])

    def leaky(frame):
        grid = es.to_grid(frame)
        centred = pd.Series(grid.qv).rolling(11, center=True, min_periods=1).median().to_numpy()
        return grid, grid.qv > 3 * centred

    grid, a = leaky(base)
    _, b = leaky(mixed)
    last = int(grid.index.searchsorted(cut_time, side="right"))
    assert not np.array_equal(a[:last], b[:last])


def test_first_in_window_is_causal_thinning():
    mask = np.zeros(30, dtype=bool)
    mask[[3, 4, 5, 9, 10, 20]] = True
    kept = es.first_in_window(mask, 5)
    assert np.flatnonzero(kept).tolist() == [3, 9, 20]  # 4, 5 fall within 5 bars of 3; 10 within 5 of 9
    assert np.array_equal(es.first_in_window(mask, 1), mask)
    # causal: truncating the input does not change what was kept
    assert np.array_equal(es.first_in_window(mask[:11], 5), kept[:11])


# --------------------------------------------------------------------------- #
# planted effect
# --------------------------------------------------------------------------- #


def _study(frame, name, horizons=es.HORIZONS):
    grid = es.to_grid(frame)
    feats = es.trailing_features(grid)
    fwd = es.forward_returns(grid, horizons)
    cell, n_cells = es.baseline_cells(grid, feats["vol_state"])
    base = es.baseline_returns(fwd, cell, n_cells)
    day_index = grid.day - grid.day.min()
    n_days = int(day_index.max()) + 1
    (d,) = list(defs.generate(grid, feats, only={name}))
    agg = es.aggregate(d.mask, d.direction, fwd, base, day_index, n_days, horizons)
    return es.summarise(agg, horizons), d, grid


def test_planted_effect_is_recovered_at_the_right_horizons_and_absent_otherwise():
    n, h0, effect = 400_000, 15, 20e-4  # 20 bp earned over the 15 minutes after the event, in the direction of the bar
    rng = np.random.default_rng(5)
    events = np.arange(2000, n - 3000, 150) + rng.integers(0, 20, size=len(np.arange(2000, n - 3000, 150)))
    vol_mult = np.ones(n)
    vol_mult[events] = 100.0  # the planted structure: a volume spike
    plain = synth(n, seed=6, vol_mult=vol_mult)
    bar_side = np.sign((plain["close"] - plain["open"]).to_numpy())[events]
    drift = np.zeros(n)
    for e, side in zip(events, bar_side):
        drift[e + 1 : e + 1 + h0] += side * effect / h0  # bars t+1 .. t+h0, i.e. open[t+1] -> open[t+1+h0]
    planted = synth(n, seed=6, vol_mult=vol_mult, drift=drift)

    table, d, _ = _study(planted, "volspike10_all_both")
    assert abs(int(d.mask.sum()) - len(events)) <= 3  # the detector finds the planted bars and (almost) nothing else
    for h, expected in ((1, effect / h0), (5, effect * 5 / h0), (15, effect), (60, effect), (240, effect), (1440, effect)):
        row = table.loc[h]
        se_bp = abs(row["excess_bp"] / row["t_day"])
        assert abs(row["gross_bp"] - expected * 1e4) < 4 * se_bp + 0.05, (h, row["gross_bp"], expected * 1e4, se_bp)
    assert table.loc[15, "t_day"] > 25 and table.loc[5, "t_day"] > 10
    assert abs(table.loc[15, "base_bp"]) < 1.0  # the baseline of a signed event is ~0
    assert table.loc[15, "hit"] > 0.75

    null_table, _, _ = _study(plain, "volspike10_all_both")
    assert (null_table["t_day"].abs() < 3.5).all()
    assert abs(null_table.loc[15, "gross_bp"]) < 2.0

    # a continuation effect, tested as a reversal hypothesis, comes out with the opposite sign
    exh, d_exh, _ = _study(planted, "volspike10_exh_both")
    assert d_exh.mask.sum() > 100 and exh.loc[15, "gross_bp"] < -15


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


def test_cluster_mean_matches_hand_computation():
    n_day = np.array([2.0, 0.0, 3.0, 1.0])
    s_day = np.array([4.0, 0.0, -3.0, 2.0])
    out = es.cluster_mean(n_day, s_day)
    mean = 3.0 / 6.0
    score = s_day - mean * n_day
    se = np.sqrt((score**2).sum() * 3 / 2) / 6.0
    assert out["n"] == 6 and out["days"] == 3
    assert out["mean"] == pytest.approx(mean) and out["se"] == pytest.approx(se) and out["t"] == pytest.approx(mean / se)
    # with one event per day and no HAC it is the ordinary t-test
    x = np.random.default_rng(0).normal(0.3, 1.0, 50)
    out = es.cluster_mean(np.ones(50), x)
    assert out["t"] == pytest.approx(x.mean() / (x.std(ddof=1) / np.sqrt(50)))


def _null_rejections(h, days, reps, hac, seed, burst=30, one_sided=False):
    """Share of |t| > 1.96 under the null for clustered, overlapping events: (clustered, naive).

    one_sided=True makes every event a buy, so that windows running into the next
    day make adjacent day sums positively correlated (the hard case for h = 1440).
    """
    rng = np.random.default_rng(seed)
    n = days * 1440
    day_index = np.arange(n) // 1440
    clustered = naive = 0
    for _ in range(reps):
        r = rng.normal(0.0, 5e-4, n + h + 2)
        cum = np.concatenate([[0.0], np.cumsum(r)])
        fwd = {h: cum[np.arange(n) + 1 + h] - cum[np.arange(n) + 1]}  # log return over bars t+1 .. t+h
        mask = np.zeros(n, dtype=bool)
        side = np.zeros(n)
        for day in range(days):
            for _ in range(rng.integers(0, 4)):
                start = day * 1440 + rng.integers(0, 1440 - burst)
                mask[start : start + burst] = True
                side[start : start + burst] = 1.0 if one_sided else rng.choice([-1.0, 1.0])
        agg = es.aggregate(mask, side, fwd, {h: np.zeros(n)}, day_index, days, (h,))
        stat = es.cluster_mean(agg[0, 0], agg[1, 0], hac)
        clustered += abs(stat["t"]) > 1.96
        naive += abs(es.naive_t(agg[0, 0].sum(), agg[1, 0].sum(), agg[3, 0].sum())) > 1.96
    return clustered / reps, naive / reps


def test_day_clustered_t_is_calibrated_where_the_naive_t_is_not():
    clustered, naive = _null_rejections(h=60, days=120, reps=300, hac=es.HAC_LAGS[60], seed=11)
    assert naive > 0.5  # bursts of 30 overlapping 60-minute returns: the naive t rejects most of the time
    assert 0.01 <= clustered <= 0.10  # nominal 5%


def _se_calibration(h, days, reps, hacs, seed):
    """(sd of the estimate across simulations) / (average reported se) when EVERY bar is a buy event.

    This is the worst case for windows that run into the following days: 1.0 means
    the reported standard error is right, above 1 means it is too small. Returns
    one ratio per entry of ``hacs`` (number of Bartlett lags), on the same simulations.
    """
    rng = np.random.default_rng(seed)
    n = days * 1440
    day_index = np.arange(n) // 1440
    mask = np.ones(n, dtype=bool)
    means, ses = [], {hac: [] for hac in hacs}
    for _ in range(reps):
        r = rng.normal(0.0, 5e-4, n + h + 2)
        cum = np.concatenate([[0.0], np.cumsum(r)])
        fwd = {h: cum[np.arange(n) + 1 + h] - cum[np.arange(n) + 1]}
        agg = es.aggregate(mask, 1.0, fwd, {h: np.zeros(n)}, day_index, days, (h,))
        for hac in hacs:
            stat = es.cluster_mean(agg[0, 0], agg[1, 0], hac)
            ses[hac].append(stat["se"])
        means.append(stat["mean"])
    return {hac: float(np.std(means, ddof=1) / np.mean(ses[hac])) for hac in hacs}


def test_bartlett_lags_repair_the_standard_error_at_the_one_day_horizon():
    # theory for this worst case: day clustering alone reports sqrt(2/3) of the true error (ratio 1.22),
    # three Bartlett lags report sqrt(2/3 + 2 * 0.75 / 6) = 0.957 of it (ratio 1.04)
    ratio = _se_calibration(h=1440, days=240, reps=500, hacs=(0, es.HAC_LAGS[1440]), seed=12)
    assert ratio[0] > 1.12
    assert 0.92 < ratio[es.HAC_LAGS[1440]] < 1.15
    short = _se_calibration(h=60, days=120, reps=300, hacs=(es.HAC_LAGS[60],), seed=13)
    assert 0.88 < short[es.HAC_LAGS[60]] < 1.12


def test_block_bootstrap_agrees_with_cluster_se_on_a_simple_case():
    rng = np.random.default_rng(3)
    n_day = rng.integers(0, 6, 400).astype(float)
    s_day = rng.normal(0.0, 1.0, 400) * np.sqrt(n_day)
    a = es.cluster_mean(n_day, s_day)
    b = es.day_block_bootstrap(n_day, s_day, block=5, n_boot=1500, seed=1)
    assert b["mean"] == pytest.approx(a["mean"])
    assert b["se"] == pytest.approx(a["se"], rel=0.2)


# --------------------------------------------------------------------------- #
# baseline and costs
# --------------------------------------------------------------------------- #


def test_matched_baseline_removes_a_pure_hour_of_day_drift():
    n = 200_000
    index = pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC")
    drift = np.where(index.hour == 14, 1e-4, 0.0)  # +1 bp per minute between 14:00 and 15:00, every day
    frame = synth(n, seed=8, sigma=2e-4, drift=drift)
    grid = es.to_grid(frame)
    feats = es.trailing_features(grid)
    fwd = es.forward_returns(grid, (15,))
    day_index = grid.day - grid.day.min()
    n_days = int(day_index.max()) + 1
    # an "event" with no information beyond the clock: minute 10 of hour 14
    mask = grid.usable & (grid.minute_of_day == 14 * 60 + 10)
    cell, n_cells = es.baseline_cells(grid, feats["vol_state"], by_hour=True)
    matched = es.summarise(es.aggregate(mask, 1.0, fwd, es.baseline_returns(fwd, cell, n_cells), day_index, n_days, (15,)), (15,))
    assert matched.loc[15, "gross_bp"] > 12 and abs(matched.loc[15, "t_gross"]) > 10  # looks like a large effect
    assert abs(matched.loc[15, "excess_bp"]) < 2.5 and abs(matched.loc[15, "t_day"]) < 3  # and is not one
    cell0, n0 = es.baseline_cells(grid, feats["vol_state"], by_hour=False)
    unmatched = es.summarise(es.aggregate(mask, 1.0, fwd, es.baseline_returns(fwd, cell0, n0), day_index, n_days, (15,)), (15,))
    assert unmatched.loc[15, "excess_bp"] > 10  # the hour effect itself is visible against the non-hourly baseline


def test_round_trip_costs():
    assert es.round_trip_cost_bp("BTCUSDT", "perp_taker") == 12.0
    assert es.round_trip_cost_bp("ETHUSDT", "perp_taker") == 12.0
    assert es.round_trip_cost_bp("SOLUSDT", "perp_taker") == 14.0
    assert es.round_trip_cost_bp("DOGEUSDT", "perp_maker") == 4.0
    assert es.round_trip_cost_bp("BTCUSDT", "spot_taker") == 22.0
    assert es.round_trip_cost_bp("XRPUSDT", "spot_taker") == 24.0
    table = es.summarise(es.aggregate(np.array([True, True]), 1.0, {1: np.array([0.002, 0.004])}, {1: np.zeros(2)}, np.array([0, 1]), 2, (1,)), (1,), cost_bp=12.0)
    assert table.loc[1, "gross_bp"] == pytest.approx(30.0) and table.loc[1, "net_bp"] == pytest.approx(18.0)


def test_previous_funding_uses_only_rates_already_stamped():
    frame = synth(1500, seed=9, start="2021-03-01 00:00")
    grid = es.to_grid(frame)
    stamps = pd.to_datetime(["2021-03-01 00:00", "2021-03-01 08:00", "2021-03-01 16:00"], utc=True)
    rate = defs.previous_funding(grid, pd.Series([1e-4, -2e-4, 3e-4], index=stamps))
    assert rate[0] == 1e-4  # stamped at this bar's open: already paid
    assert rate[479] == 1e-4  # bar 07:59, closes at 08:00: the 08:00 rate is NOT yet used
    assert rate[480] == -2e-4 and rate[959] == -2e-4 and rate[960] == 3e-4
