import numpy as np
import pandas as pd
import pytest

from algotrade import signals, universe
from algotrade.validate import check_no_lookahead


def random_panel(n=400, m=6, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    names = [f"C{i}USDT" for i in range(m)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (n, m)), axis=0)), index=index, columns=names)
    close.iloc[:120, 1] = np.nan  # listed later
    close.iloc[300:, 2] = np.nan  # delisted
    volume = pd.DataFrame(rng.lognormal(10, 1, (n, m)), index=index, columns=names).where(close.notna())
    return {"close": close, "open": close.shift(1).fillna(close), "quote_volume": volume}


def test_symbol_classification():
    names = ["BTCUSDT", "BTCUPUSDT", "BTCDOWNUSDT", "JUPUSDT", "USDCUSDT", "PAXGUSDT", "WBTCUSDT",
             "1000PEPEUSDT", "PEPEUSDT", "ETHBULLUSDT", "ETHUSDT", "BEARUSDT", "SUPERUSDT"]
    table = universe.classify(names)
    assert table.loc["BTCUPUSDT", "kind"] == "leveraged"
    assert table.loc["BTCDOWNUSDT", "kind"] == "leveraged"
    assert table.loc["ETHBULLUSDT", "kind"] == "leveraged"
    assert table.loc["BEARUSDT", "kind"] == "leveraged"
    assert table.loc["JUPUSDT", "kind"] == "coin"  # ends in "UP" but "J" is not a coin
    assert table.loc["SUPERUSDT", "kind"] == "coin"
    assert table.loc["USDCUSDT", "kind"] == "stable"
    assert table.loc["PAXGUSDT", "kind"] == "gold"
    assert table.loc["WBTCUSDT", "kind"] == "pegged"
    assert universe.base_asset("1000PEPEUSDT") == "PEPE"
    assert universe.base_asset("1000000MOGUSDT") == "MOG"
    assert universe.coin_symbols(names) == ["1000PEPEUSDT", "BTCUSDT", "ETHUSDT", "JUPUSDT", "PEPEUSDT", "SUPERUSDT"]
    with pytest.raises(ValueError):
        universe.base_asset("BTCEUR")


def test_top_liquid_respects_history_listing_and_size():
    panel = random_panel()
    member = universe.top_liquid(panel["quote_volume"], top_n=3, lookback=30, min_history=60)
    assert not member.iloc[:59].any().any()
    assert not member["C1USDT"].iloc[: 120 + 59].any()  # needs its own 60 bars after listing
    assert not member["C2USDT"].iloc[300:].any()  # gone once delisted
    counts = member.sum(axis=1).iloc[200:]
    assert counts.min() >= 3 and counts.max() <= 4  # between top_n and exit_rank


def test_top_liquid_does_not_look_ahead():
    panel = random_panel()
    volume = panel["quote_volume"]
    full = universe.top_liquid(volume, top_n=3)
    cut = 250
    part = universe.top_liquid(volume.iloc[:cut], top_n=3)
    pd.testing.assert_frame_equal(part, full.iloc[:cut])


def test_hysteresis_keeps_a_member_that_slips_just_outside():
    index = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    volume = pd.DataFrame({"A": 100.0, "B": 90.0, "C": 80.0, "D": 70.0, "E": 1.0}, index=index)
    volume.iloc[5:, volume.columns.get_loc("D")] = 85.0  # D overtakes C from bar 5
    member = universe.top_liquid(volume, top_n=3, lookback=1, min_history=1, exit_rank=4)
    assert member.iloc[4].tolist() == [True, True, True, False, False]
    assert member.iloc[5].tolist() == [True, True, True, True, False]  # C stays (rank 4), D enters


def strategy(panel):
    close = panel["close"]
    member = universe.top_liquid(panel["quote_volume"], top_n=4, min_history=40)
    vol = signals.ewm_volatility(close, halflife=20, periods=365.25)
    trend = signals.ema_trend(close, 10, 40).clip(lower=0) * signals.breakout_state(close, 20).abs()
    momentum = signals.cross_sectional_rank(signals.past_return(close, 30, skip=1), member)
    weights = signals.volatility_target(trend, vol, target_vol=0.2, max_weight=0.3)
    return weights.where(member & (momentum > 0.5), 0.0).fillna(0.0)


def test_library_signals_pass_the_lookahead_check():
    assert check_no_lookahead(strategy, random_panel()) == []


def test_lookahead_check_catches_leaks():
    panel = random_panel()

    def peeks_one_bar(p):
        return (p["close"].shift(-1) > p["close"]).astype(float) * 0.1

    def full_sample_scaling(p):
        close = p["close"]
        return ((close - close.mean()) / close.std()).clip(0, 1).fillna(0.0) * 0.1

    assert check_no_lookahead(peeks_one_bar, panel)
    assert check_no_lookahead(full_sample_scaling, panel)


def test_breakout_state_machine():
    index = pd.date_range("2024-01-01", periods=9, freq="1D", tz="UTC")
    close = pd.DataFrame({"A": [10, 10, 10, 11, 12, 11.5, 9.5, 9.0, 12.5]}, index=index, dtype=float)
    state = signals.breakout_state(close, entry=3, exit_=2)["A"].tolist()
    #        warm-up        breakout   hold  exit below 2-bar low -> also 3-bar low -> short ... new high -> long
    assert state == [0, 0, 0, 1, 1, 1, -1, -1, 1]


def test_volatility_target_scales_and_clips():
    index = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    signal = pd.DataFrame({"A": [1.0, -1.0], "B": [1.0, 1.0]}, index=index)
    vol = pd.DataFrame({"A": [0.4, 0.4], "B": [0.05, 0.0]}, index=index)
    weights = signals.volatility_target(signal, vol, target_vol=0.2, max_weight=1.0)
    assert weights["A"].tolist() == [0.5, -0.5]
    assert weights["B"].iloc[0] == 1.0  # 0.2 / 0.05 = 4, clipped
    assert np.isnan(weights["B"].iloc[1])  # no volatility estimate -> no position
