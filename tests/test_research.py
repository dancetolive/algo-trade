import json

import numpy as np
import pandas as pd
import pytest

from algotrade import research
from algotrade.data import store


def panel(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")  # spans design, validation and hold-out
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, (n, 2)), axis=0)), index=index, columns=["AUSDT", "BUSDT"]
    )
    return close.shift(1).fillna(close), close


def test_segments_partition_the_index():
    index = pd.date_range("2023-12-30", "2025-07-02", freq="1D", tz="UTC")
    masks = research.segments(index)
    assert (masks["design"].astype(int) + masks["validation"].astype(int) + masks["holdout"].astype(int) == 1).all()
    assert index[masks["design"]][-1] == pd.Timestamp("2023-12-31", tz="UTC")
    assert index[masks["validation"]][0] == store.DESIGN_END
    assert index[masks["validation"]][-1] == pd.Timestamp("2025-06-30", tz="UTC")
    assert index[masks["holdout"]][0] == store.HOLDOUT_START


def test_venues_translate_into_simulator_settings():
    spot = research.VENUES["spot"]
    config = spot.sim_config(initial_equity=500, rebalance_band=0.02)
    assert config.costs.rate == pytest.approx(0.0015)
    assert not config.allow_short and config.max_gross == 1.0
    assert config.initial_equity == 500 and config.rebalance_band == 0.02
    stressed = spot.stressed(2)
    assert stressed.fee_bps == 20 and stressed.slippage_bps == 10 and stressed.name == "spot_x2"
    assert research.VENUES["perp"].allow_short


def test_run_trial_scores_segments_and_logs(tmp_path):
    open_, close = panel()
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    record, result = research.run_trial(
        "demo", "half-half", {"w": 0.5}, weights, open_, close, research.VENUES["spot"], registry=tmp_path
    )
    assert set(record["metrics"]) == {"design", "validation", "holdout"}
    design = record["metrics"]["design"]
    assert design["bars"] == 365
    assert 0.9 < design["avg_gross"] <= 1.01
    assert design["turnover_per_year"] > 1.0  # the initial purchase plus daily rebalancing
    assert design["cost_per_year"] == pytest.approx(design["turnover_per_year"] * 0.0015, rel=0.05)
    assert record["trial_id"].startswith("0000-")
    assert result.equity.iloc[-1] > 0

    second, _ = research.run_trial(
        "demo", "all-in-a", {"w": 1.0}, weights * [2.0, 0.0], open_, close, research.VENUES["spot"], registry=tmp_path
    )
    assert second["trial_id"].startswith("0001-")
    ledger = (tmp_path / "demo" / "trials.jsonl").read_text().strip().splitlines()
    assert len(ledger) == 2 and json.loads(ledger[1])["name"] == "all-in-a"

    table = research.load_trials("demo", registry=tmp_path)
    assert list(table["name"]) == ["half-half", "all-in-a"]
    assert {"design_sharpe", "validation_sharpe", "design_turnover_per_year"} <= set(table.columns)

    returns = research.trial_returns("demo", "design", registry=tmp_path)
    assert returns.shape == (365, 2)
    assert returns.index.max() < store.DESIGN_END


def test_unlogged_trial_leaves_no_trace(tmp_path):
    open_, close = panel()
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    record, _ = research.run_trial(
        "demo", "smoke", {}, weights, open_, close, research.VENUES["spot"], registry=tmp_path, log=False
    )
    assert "trial_id" not in record
    assert not (tmp_path / "demo").exists()
    assert research.load_trials("demo", registry=tmp_path).empty


def test_overfitting_report_counts_every_trial(tmp_path):
    open_, close = panel(seed=1)
    rng = np.random.default_rng(2)
    chosen = None
    for k in range(12):
        weights = pd.DataFrame(rng.uniform(0, 0.5, close.shape), index=close.index, columns=close.columns)
        record, _ = research.run_trial(
            "noise", f"random-{k}", {"k": k}, weights, open_, close, research.VENUES["spot"], registry=tmp_path
        )
        chosen = chosen or record["trial_id"]
    report = research.overfitting_report("noise", chosen, registry=tmp_path)
    assert report["trials"] == 12
    assert 0.0 <= report["pbo_pbo"] <= 1.0
    assert report["deflated_sharpe"] <= report["psr"]
    assert report["expected_max_sharpe_annualised"] > 0
    with pytest.raises(KeyError):
        research.overfitting_report("noise", "missing", registry=tmp_path)


def test_buy_and_hold_benchmark():
    open_, close = panel(n=50)
    result = research.buy_and_hold(open_, close, "AUSDT", research.VENUES["spot"], initial_equity=1000)
    entry = open_["AUSDT"].iloc[1]
    expected = 1000 * (1 - 0.0015) + (1000 / entry) * (close["AUSDT"].iloc[-1] - entry)
    assert result.equity.iloc[-1] == pytest.approx(expected)
    assert result.turnover.iloc[2:].sum() == 0.0


def test_perpetual_trials_require_funding_and_extremes(tmp_path):
    open_, close = panel(n=200)
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    perp = research.VENUES["perp"]
    with pytest.raises(ValueError, match="perpetual trials need funding"):
        research.run_trial("p", "x", {}, weights, open_, close, perp, registry=tmp_path)
    high, low = np.maximum(open_, close), np.minimum(open_, close)
    funding = pd.DataFrame(0.0001, index=close.index, columns=close.columns)
    funding.iloc[:50] = np.nan  # no funding data for the first 50 bars
    record, result = research.run_trial(
        "p", "x", {}, weights, open_, close, perp, funding=funding, high=high, low=low, registry=tmp_path
    )
    assert result.config.maintenance_margin == 0.025
    assert record["metrics"]["design"]["funding_missing_bars"] == result.funding_missing == 2 * 49
    assert record["metrics"]["design"]["funding_per_year"] > 0


def test_intraday_results_are_scored_and_logged_on_daily_returns(tmp_path):
    rng = np.random.default_rng(7)
    index = pd.date_range("2023-06-01", periods=24 * 250, freq="1h", tz="UTC")
    close = pd.DataFrame({"AUSDT": 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(index))))}, index=index)
    open_ = close.shift(1).fillna(close)
    weights = pd.DataFrame(1.0, index=index, columns=["AUSDT"])
    record, result = research.run_trial("h", "hold", {}, weights, open_, close, research.VENUES["spot"], registry=tmp_path)
    design = record["metrics"]["design"]
    assert design["bars"] == 214  # days, not hours (1 June to 31 December 2023)
    daily = research.daily_returns(result.returns)
    assert len(daily) == 250
    assert np.prod(1 + daily) == pytest.approx(np.prod(1 + result.returns))
    assert design["intraday_max_drawdown"] <= design["max_drawdown"] <= 0
    stored = research.trial_returns("h", None, registry=tmp_path)
    assert len(stored) == 250
