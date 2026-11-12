import numpy as np
import pandas as pd
import pytest

from algotrade.backtest import Costs, SimConfig, align_funding, simulate

FREE = Costs(fee_bps=0.0, slippage_bps=0.0)


def frame(values: dict, start="2024-01-01", freq="1D") -> pd.DataFrame:
    n = len(next(iter(values.values())))
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(values, index=index, dtype=float)


def once(index, weights: dict) -> pd.DataFrame:
    """A single decision at the first bar; later rows are 'no decision'."""
    out = pd.DataFrame(np.nan, index=index, columns=list(weights))
    out.iloc[0] = list(weights.values())
    return out


def test_buy_and_hold_fills_at_next_open():
    o = frame({"A": [100, 101, 103, 102]})
    c = frame({"A": [101, 103, 102, 110]})
    w = pd.DataFrame(1.0, index=o.index, columns=["A"])
    res = simulate(o, c, w, SimConfig(initial_equity=1000, costs=FREE))
    # decided at the close of bar 0, filled at the open of bar 1 (101)
    assert res.equity.iloc[0] == pytest.approx(1000.0)
    assert res.equity.iloc[1] == pytest.approx(1000 * 103 / 101)
    assert res.equity.iloc[-1] == pytest.approx(1000 * 110 / 101)
    assert res.turnover.iloc[1] == pytest.approx(1.0)
    assert res.turnover.iloc[2:].sum() == pytest.approx(0.0, abs=1e-12)


def test_decision_cannot_earn_its_own_bar():
    # +50% happens inside bar 2. A decision taken at the close of bar 2 must miss it.
    o = frame({"A": [100, 100, 100, 150, 150]})
    c = frame({"A": [100, 100, 150, 150, 150]})
    w = pd.DataFrame(np.nan, index=o.index, columns=["A"])
    w.iloc[2] = 1.0
    res = simulate(o, c, w, SimConfig(costs=FREE))
    assert res.equity.iloc[2] == pytest.approx(1000.0)
    assert res.equity.iloc[-1] == pytest.approx(1000.0)

    # ...whereas a decision at the close of bar 1 catches it.
    w = pd.DataFrame(np.nan, index=o.index, columns=["A"])
    w.iloc[1] = 1.0
    res = simulate(o, c, w, SimConfig(costs=FREE))
    assert res.equity.iloc[2] == pytest.approx(1500.0)


def test_costs_are_charged_on_traded_notional():
    o = frame({"A": [100, 101, 103]})
    c = frame({"A": [101, 103, 103]})
    res = simulate(o, c, once(o.index, {"A": 1.0}), SimConfig(costs=Costs(fee_bps=10, slippage_bps=0)))
    units = 1000 / 101
    assert res.costs.iloc[1] == pytest.approx(1.0)
    assert res.equity.iloc[1] == pytest.approx(1000 - 1.0 + units * (103 - 101))
    assert res.costs.sum() == pytest.approx(1.0)


def test_no_decision_rows_let_positions_drift():
    o = frame({"A": [10, 10, 10, 10], "B": [20, 20, 20, 20]})
    c = frame({"A": [10, 12, 15, 15], "B": [20, 20, 10, 10]})
    res = simulate(o, c, once(o.index, {"A": 0.5, "B": 0.5}), SimConfig(costs=FREE))
    # gap moves (close -> next open) are part of the P&L too
    assert res.equity.iloc[1] == pytest.approx(500 * 1.2 + 500)
    assert res.equity.iloc[2] == pytest.approx(50 * 15 + 25 * 10)
    assert res.turnover.iloc[2:].sum() == 0.0
    assert res.weights.iloc[2]["A"] == pytest.approx(750 / 1000)


def test_constant_mix_matches_closed_form():
    rng = np.random.default_rng(1)
    n = 60
    opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(n, 2)), axis=0))
    closes = opens * np.exp(rng.normal(0, 0.01, size=(n, 2)))
    o, c = frame({"A": opens[:, 0], "B": opens[:, 1]}), frame({"A": closes[:, 0], "B": closes[:, 1]})
    w = pd.DataFrame(0.5, index=o.index, columns=["A", "B"])
    res = simulate(o, c, w, SimConfig(costs=FREE))
    growth = 1.0
    for t in range(1, n - 1):
        growth *= 1 + 0.5 * (opens[t + 1] / opens[t] - 1).sum()
    growth *= 1 + 0.5 * (closes[-1] / opens[-1] - 1).sum()
    assert res.equity.iloc[-1] == pytest.approx(1000 * growth, rel=1e-10)


def test_short_profits_when_price_falls_and_receives_positive_funding():
    o = frame({"A": [100, 100, 90]})
    c = frame({"A": [100, 90, 90]})
    f = frame({"A": [0.0, 0.001, 0.0]})
    cfg = SimConfig(costs=FREE, allow_short=True)
    res = simulate(o, c, once(o.index, {"A": -1.0}), cfg, funding=f, high=np.maximum(o, c), low=np.minimum(o, c))
    assert res.equity.iloc[1] == pytest.approx(1000 + 100 + 10 * 90 * 0.001)
    assert res.funding.iloc[1] == pytest.approx(-0.9)


def test_long_pays_positive_funding():
    o = frame({"A": [100, 100, 100]})
    c = frame({"A": [100, 100, 100]})
    f = frame({"A": [0.0, 0.001, 0.002]})
    res = simulate(o, c, once(o.index, {"A": 1.0}), SimConfig(costs=FREE), funding=f)
    assert res.equity.iloc[1] == pytest.approx(1000 - 1.0)
    assert res.equity.iloc[2] == pytest.approx(1000 - 1.0 - 2.0)
    assert res.funding.sum() == pytest.approx(3.0)


def test_funding_is_not_charged_before_the_position_exists():
    o = frame({"A": [100, 100, 100]})
    c = frame({"A": [100, 100, 100]})
    f = frame({"A": [0.01, 0.0, 0.0]})  # bar 0: nothing is held yet
    res = simulate(o, c, once(o.index, {"A": 1.0}), SimConfig(costs=FREE), funding=f)
    assert res.equity.iloc[-1] == pytest.approx(1000.0)


def test_align_funding_assigns_payments_to_the_bar_that_carried_the_position():
    index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    stamps = pd.to_datetime(
        [
            "2024-01-01 00:00",  # before any bar was carried: dropped
            "2024-01-02 00:00",  # boundary: belongs to Jan 1
            "2024-01-02 08:00",
            "2024-01-02 16:00",
            "2024-01-03 00:00",  # boundary: belongs to Jan 2
            "2024-01-04 00:00",  # end of the last bar: belongs to Jan 3
            "2024-01-04 08:00",  # after the last bar: dropped
        ],
        utc=True,
    )
    funding = pd.DataFrame({"A": [9.0, 1.0, 2.0, 4.0, 8.0, 16.0, 99.0]}, index=stamps)
    aligned = align_funding(funding, index)
    assert aligned["A"].tolist() == [1.0, 14.0, 16.0]


def test_delisted_position_is_closed_at_last_close_with_penalty():
    o = frame({"A": [100, 100, 100, np.nan, np.nan]})
    c = frame({"A": [100, 100, 80, np.nan, np.nan]})
    cfg = SimConfig(costs=FREE, delist_slippage_bps=100)
    res = simulate(o, c, once(o.index, {"A": 1.0}), cfg)
    assert res.equity.iloc[2] == pytest.approx(800.0)
    assert res.equity.iloc[3] == pytest.approx(800.0 - 8.0)
    assert res.equity.iloc[4] == pytest.approx(792.0)
    assert res.weights.iloc[3]["A"] == 0.0


def test_unlisted_asset_is_not_bought():
    o = frame({"A": [np.nan, np.nan, 100, 100]})
    c = frame({"A": [np.nan, np.nan, 100, 110]})
    w = pd.DataFrame(1.0, index=o.index, columns=["A"])
    res = simulate(o, c, w, SimConfig(costs=FREE))
    assert res.equity.iloc[2] == pytest.approx(1000.0)  # filled at the open of bar 2
    assert res.weights.iloc[2]["A"] == pytest.approx(1.0)
    assert res.equity.iloc[3] == pytest.approx(1100.0)


def test_gross_exposure_is_capped():
    o = frame({"A": [10, 10, 10], "B": [10, 10, 10]})
    res = simulate(o, o, once(o.index, {"A": 1.5, "B": 1.5}), SimConfig(costs=FREE, max_gross=1.5), high=o, low=o)
    assert res.weights.iloc[1].tolist() == pytest.approx([0.75, 0.75])
    assert res.gross.iloc[1] == pytest.approx(1.5)


def test_rebalance_band_skips_small_changes_but_always_allows_exits():
    o = frame({"A": [10] * 6})
    w = pd.DataFrame({"A": [0.50, 0.52, 0.60, 0.0, 0.0, 0.0]}, index=o.index)
    res = simulate(o, o, w, SimConfig(costs=FREE, rebalance_band=0.05))
    assert res.turnover.iloc[1] == pytest.approx(0.50)
    assert res.turnover.iloc[2] == 0.0  # 0.50 -> 0.52 is inside the band
    assert res.turnover.iloc[3] == pytest.approx(0.10)  # 0.50 -> 0.60
    assert res.turnover.iloc[4] == pytest.approx(0.60)  # full exit
    assert res.weights.iloc[4]["A"] == 0.0

    w = pd.DataFrame({"A": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=o.index)
    res = simulate(o, o, w, SimConfig(costs=FREE, rebalance_band=0.05))
    assert res.turnover.sum() == 0.0  # the entry itself was inside the band


def test_min_trade_notional_skips_dust():
    o = frame({"A": [10] * 4})
    w = pd.DataFrame({"A": [0.5, 0.505, 0.0, 0.0]}, index=o.index)
    res = simulate(o, o, w, SimConfig(costs=FREE, min_trade_notional=10.0))
    assert res.turnover.iloc[1] == pytest.approx(0.5)
    assert res.turnover.iloc[2] == 0.0  # 5 of notional < 10
    assert res.turnover.iloc[3] == pytest.approx(0.5)  # exit is always allowed


def test_leveraged_position_is_liquidated_on_adverse_intrabar_move():
    o = frame({"A": [100, 100, 100]})
    c = frame({"A": [100, 100, 100]})
    h = frame({"A": [100, 101, 100]})
    low = frame({"A": [100, 75, 100]})
    cfg = SimConfig(costs=FREE, max_gross=5.0, maintenance_margin=0.05)
    res = simulate(o, c, once(o.index, {"A": 5.0}), cfg, high=h, low=low)
    assert res.liquidations == 1
    assert res.ruined
    assert res.equity.iloc[1] == 0.0
    assert res.equity.iloc[2] == 0.0
    assert res.returns.iloc[2] == 0.0


def test_unlevered_long_is_never_liquidated():
    o = frame({"A": [100, 100, 100]})
    c = frame({"A": [100, 60, 60]})
    h = frame({"A": [100, 100, 60]})
    low = frame({"A": [100, 50, 60]})
    cfg = SimConfig(costs=FREE, maintenance_margin=0.05)
    res = simulate(o, c, once(o.index, {"A": 1.0}), cfg, high=h, low=low)
    assert res.liquidations == 0
    assert res.equity.iloc[1] == pytest.approx(600.0)


def test_invalid_inputs_are_rejected():
    o = frame({"A": [1, 1, 1]})
    with pytest.raises(ValueError):
        simulate(o, o, once(o.index, {"A": -1.0}), SimConfig())
    with pytest.raises(ValueError):
        simulate(o, o, once(o.index, {"A": 1.0}), SimConfig(exec_delay=0))
    with pytest.raises(KeyError):
        simulate(o, o, once(o.index, {"Z": 1.0}), SimConfig())
    with pytest.raises(ValueError, match="high and low"):
        simulate(o, o, once(o.index, {"A": -1.0}), SimConfig(allow_short=True))
    with pytest.raises(ValueError, match="high and low"):
        simulate(o, o, once(o.index, {"A": 1.0}), SimConfig(max_gross=2.0))
    holes = o.copy()
    holes.iloc[1] = np.nan
    with pytest.raises(ValueError, match="high / low are missing"):
        simulate(o, o, once(o.index, {"A": -1.0}), SimConfig(allow_short=True), high=holes, low=o)
    stray = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2030-01-01"], utc=True))
    with pytest.raises(ValueError):
        simulate(o, o, stray, SimConfig())


def test_returns_compound_to_equity():
    rng = np.random.default_rng(3)
    n = 200
    opens = 50 * np.exp(np.cumsum(rng.normal(0, 0.03, size=(n, 3)), axis=0))
    closes = opens * np.exp(rng.normal(0, 0.02, size=(n, 3)))
    names = ["A", "B", "C"]
    o = frame(dict(zip(names, opens.T)))
    c = frame(dict(zip(names, closes.T)))
    w = pd.DataFrame(rng.uniform(-0.5, 0.5, size=(n, 3)), index=o.index, columns=names)
    cfg = SimConfig(allow_short=True, max_gross=1.5, costs=Costs(5, 2))
    res = simulate(o, c, w, cfg, high=np.maximum(o, c) * 1.001, low=np.minimum(o, c) * 0.999)
    assert res.equity.iloc[-1] == pytest.approx(1000 * np.prod(1 + res.returns.to_numpy()))
    assert (res.gross <= 1.5 * 1.25).all()  # drift can exceed the cap a little, never wildly
    assert res.costs.sum() > 0


def test_halted_market_keeps_its_position_and_books_the_move_when_it_reopens():
    # no bars on days 2 and 3, then the market reopens 30% higher
    o = frame({"A": [100, 100, np.nan, np.nan, 130, 130]})
    c = frame({"A": [100, 100, np.nan, np.nan, 130, 143]})
    res = simulate(o, c, once(o.index, {"A": 1.0}), SimConfig(costs=FREE))
    assert res.equity.tolist() == pytest.approx([1000, 1000, 1000, 1000, 1300, 1430])
    assert res.turnover.iloc[2:].sum() == 0.0  # nothing was sold or bought back
    assert res.costs.sum() == 0.0
    assert res.weights.iloc[2]["A"] == pytest.approx(1.0)  # carried at the last close


def test_long_absence_is_a_delisting_and_a_relisting_jump_is_not_booked():
    days = 30
    gap = [np.nan] * days
    o = frame({"A": [100, 100, 100] + gap + [500, 500]})
    c = frame({"A": [100, 100, 80] + gap + [500, 500]})
    res = simulate(o, c, once(o.index, {"A": 1.0}), SimConfig(costs=FREE, delist_slippage_bps=100, max_halt_days=7))
    assert res.equity.iloc[2] == pytest.approx(800.0)
    assert res.equity.iloc[3] == pytest.approx(792.0)  # settled at the last close, 1% penalty
    assert res.equity.iloc[-1] == pytest.approx(792.0)  # the ticker's return at 500 is somebody else's contract
    assert res.weights.iloc[-1]["A"] == 0.0


def test_halted_position_uses_up_gross_exposure():
    o = frame({"A": [10, 10, np.nan, np.nan, 10, 10], "B": [10, 10, 10, 10, 10, 10]})
    w = pd.DataFrame(np.nan, index=o.index, columns=["A", "B"])
    w.iloc[0] = [0.6, 0.0]
    w.iloc[1] = [0.6, 1.0]  # filled at bar 2, where A is halted: only 0.4 of room is left for B
    res = simulate(o, o, w, SimConfig(costs=FREE))
    assert res.weights.iloc[2]["A"] == pytest.approx(0.6)
    assert res.weights.iloc[2]["B"] == pytest.approx(0.4)
    assert res.gross.iloc[2] == pytest.approx(1.0)


def test_funding_frame_must_cover_the_traded_columns_and_be_aligned():
    o = frame({"A": [100, 100, 100], "B": [50, 50, 50]})
    weights = once(o.index, {"A": 0.5, "B": 0.5})
    with pytest.raises(KeyError, match="funding"):
        simulate(o, o, weights, SimConfig(costs=FREE), funding=frame({"A": [0.0, 0.0, 0.0]}))
    raw = pd.DataFrame({"A": [0.001], "B": [0.001]}, index=pd.to_datetime(["2024-01-02 08:00"], utc=True))
    with pytest.raises(ValueError, match="align_funding"):
        simulate(o, o, weights, SimConfig(costs=FREE), funding=raw)
    aligned = align_funding(raw, o.index)
    res = simulate(o, o, weights, SimConfig(costs=FREE), funding=aligned)
    assert res.funding.iloc[1] == pytest.approx(1.0)  # 0.1% of 1,000 of notional
    assert res.funding_missing == 2  # bar 2 has no funding data for either symbol


def test_align_funding_marks_uncovered_bars_as_missing():
    index = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    stamps = pd.to_datetime(["2024-01-02 08:00", "2024-01-04 08:00"], utc=True)
    funding = pd.DataFrame({"A": [0.001, 0.002], "B": [np.nan, np.nan]}, index=stamps)
    aligned = align_funding(funding, index)
    assert np.isnan(aligned["A"].iloc[0]) and np.isnan(aligned["A"].iloc[4])
    assert aligned["A"].iloc[1:4].tolist() == [0.001, 0.0, 0.002]
    assert aligned["B"].isna().all()


def test_short_is_wiped_out_inside_the_bar_even_if_the_close_recovers():
    o = frame({"A": [100, 100, 100, 100]})
    c = frame({"A": [100, 75, 75, 75]})  # closes 25% lower: a 2x short would look like +50%
    h = frame({"A": [100, 152, 100, 100]})  # ...but it traded 52% higher first
    low = frame({"A": [100, 75, 75, 75]})
    cfg = SimConfig(costs=FREE, allow_short=True, max_gross=2.0)
    res = simulate(o, c, once(o.index, {"A": -2.0}), cfg, high=h, low=low)
    assert res.ruined and res.liquidations == 1
    assert res.equity.iloc[1:].tolist() == [0.0, 0.0, 0.0]


def test_account_that_gaps_below_zero_at_the_open_is_ruined():
    o = frame({"A": [100, 100, 140, 100]})
    c = frame({"A": [100, 100, 100, 100]})
    h = frame({"A": [100, 100, 140, 100]})
    low = frame({"A": [100, 100, 100, 100]})
    cfg = SimConfig(costs=FREE, allow_short=True, max_gross=3.0, liquidation_fee_bps=0)
    res = simulate(o, c, once(o.index, {"A": -3.0}), cfg, high=h, low=low)
    assert res.ruined
    assert res.equity.tolist() == [1000.0, 1000.0, 0.0, 0.0]
    assert res.returns.iloc[2] == -1.0 and res.returns.iloc[3] == 0.0


def test_holdings_round_down_to_whole_lots():
    o = frame({"A": [100_000.0] * 4})
    w = pd.DataFrame({"A": [0.05, 0.26, 0.26, 0.0]}, index=o.index)
    cfg = SimConfig(costs=FREE, lot_step={"A": 0.001})
    res = simulate(o, o, w, cfg)
    assert res.weights.iloc[1]["A"] == 0.0  # 50 of notional is less than one lot of 100
    assert res.weights.iloc[2]["A"] == pytest.approx(0.2)  # 260 rounds down to two lots
    assert res.turnover.iloc[3] == 0.0  # already at the nearest attainable holding
    assert res.equity.iloc[-1] == pytest.approx(1000.0)
