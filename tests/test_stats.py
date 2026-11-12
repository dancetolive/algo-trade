import numpy as np
import pandas as pd
import pytest

from algotrade import stats


def test_periods_per_year():
    daily = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    hourly = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    assert stats.periods_per_year(daily) == pytest.approx(365.25)
    assert stats.periods_per_year(hourly) == pytest.approx(365.25 * 24)


def test_sharpe_and_cagr_on_known_series():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert stats.sharpe(r, 252) == pytest.approx(expected)
    assert stats.cagr(np.full(10, 0.01), periods=10) == pytest.approx(1.01**10 - 1)
    assert np.isnan(stats.sharpe(np.zeros(5), 252))


def test_max_drawdown():
    assert stats.max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    assert stats.max_drawdown([0.1, 0.1]) == 0.0
    assert stats.max_drawdown([-0.2]) == pytest.approx(-0.2)  # counts the fall from the starting value


def test_summary_fields():
    index = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, 400), index=index)
    s = stats.summary(r)
    assert s["bars"] == 400
    assert s["years"] == pytest.approx(400 / 365.25)
    assert s["max_drawdown"] < 0
    assert s["calmar"] == pytest.approx(s["cagr"] / abs(s["max_drawdown"]))
    table = stats.yearly(r)
    assert list(table.index) == [2024, 2025]
    assert table["bars"].sum() == 400


def test_probabilistic_sharpe_behaviour():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.01, 5000)
    assert 0.05 < stats.probabilistic_sharpe(noise) < 0.95
    skilled = rng.normal(0.001, 0.01, 5000)
    assert stats.probabilistic_sharpe(skilled) > 0.99
    # the same edge is less certain on a shorter record
    assert stats.probabilistic_sharpe(skilled[:200]) < stats.probabilistic_sharpe(skilled)


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0006, 0.01, 2000)
    psr = stats.probabilistic_sharpe(r)
    variance = 1.0 / 2000  # variance of a per-bar Sharpe estimate under the null
    few = stats.deflated_sharpe(r, n_trials=2, trial_sr_variance=variance)
    many = stats.deflated_sharpe(r, n_trials=500, trial_sr_variance=variance)
    assert many < few < psr
    assert stats.expected_max_sharpe(1, variance) == 0.0
    assert stats.expected_max_sharpe(100, variance) > stats.expected_max_sharpe(10, variance) > 0


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(2)
    t, n = 500, 50
    best = [stats.trial_sharpes(pd.DataFrame(rng.normal(0, 0.01, (t, n)))).max() for _ in range(200)]
    predicted = stats.expected_max_sharpe(n, 1.0 / t)
    assert np.mean(best) == pytest.approx(predicted, rel=0.1)


def test_pbo_is_high_for_noise_and_low_for_a_real_edge():
    rng = np.random.default_rng(4)
    t, n = 2000, 20
    noise = pd.DataFrame(rng.normal(0, 0.01, (t, n)))
    result = stats.pbo_cscv(noise, n_blocks=10)
    assert result["splits"] == 252
    assert 0.3 < result["pbo"] < 0.7
    assert result["mean_oos_sharpe_of_best"] < result["mean_is_sharpe_of_best"]

    edge = noise.copy()
    edge[0] = edge[0] + 0.002
    assert stats.pbo_cscv(edge, n_blocks=10)["pbo"] < 0.05

    with pytest.raises(ValueError):
        stats.pbo_cscv(noise, n_blocks=7)


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(5)
    r = rng.normal(0.001, 0.01, 1500)
    ci = stats.bootstrap_sharpe_ci(r, periods=365.25, n_boot=500, seed=1)
    assert ci["low"] < ci["sharpe"] < ci["high"]
    assert ci["p_positive"] > 0.9
    again = stats.bootstrap_sharpe_ci(r, periods=365.25, n_boot=500, seed=1)
    assert again == ci


def ar1(rng, n, phi, scale=0.01):
    noise = rng.normal(0, 1, n + 200)
    x = np.empty(n + 200)
    x[0] = noise[0]
    for i in range(1, n + 200):
        x[i] = phi * x[i - 1] + noise[i]
    return scale * x[200:]


def test_serial_correlation_is_measured_and_deflates_the_sharpe():
    rng = np.random.default_rng(0)
    iid = rng.normal(0.0005, 0.01, 2000)
    assert stats.autocorr_inflation(iid) < 1.4
    smooth = 0.0004 + ar1(rng, 2000, 0.8, scale=0.002)  # a carry-like, strongly autocorrelated stream
    q = stats.autocorr_inflation(smooth)
    assert q > 4
    index = pd.date_range("2020-01-01", periods=2000, freq="1D", tz="UTC")
    s = stats.summary(pd.Series(smooth, index=index))
    assert s["sharpe_adj"] == pytest.approx(s["sharpe"] / np.sqrt(q))
    assert s["vol_adj"] == pytest.approx(s["vol"] * np.sqrt(q))
    # the same stream judged from monthly returns agrees with the adjusted figure, not the raw one
    monthly = pd.Series(smooth, index=index).groupby(np.arange(2000) // 30).apply(lambda c: np.prod(1 + c) - 1)
    monthly_sharpe = stats.sharpe(monthly, 365.25 / 30)
    assert abs(s["sharpe_adj"] - monthly_sharpe) < abs(s["sharpe"] - monthly_sharpe)
    # under the null, autocorrelated noise must not look significant more often than chance allows
    hits = [stats.probabilistic_sharpe(ar1(np.random.default_rng(k), 1500, 0.8)) > 0.95 for k in range(200)]
    assert np.mean(hits) < 0.12


def test_thin_records_are_not_certified():
    one_win = np.zeros(547)
    one_win[100] = 0.05
    assert np.isnan(stats.probabilistic_sharpe(one_win))
    assert np.isnan(stats.deflated_sharpe(one_win, 20, 1 / 547))
    with pytest.raises(ValueError, match="too thin"):
        stats.bootstrap_sharpe_ci(one_win, 365.25)
    few = np.zeros(547)
    few[:40] = 0.01  # forty winners and not a single loss: still not a record one can judge
    assert np.isnan(stats.probabilistic_sharpe(few))


def test_constant_series_have_no_sharpe():
    for n in (5, 10, 365, 1000):
        assert np.isnan(stats.sharpe(np.full(n, 0.001), 365.25))
    assert np.isnan(stats.probabilistic_sharpe(np.full(1000, 0.001)))
    rng = np.random.default_rng(4)
    noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 20)))
    base = stats.pbo_cscv(noise)["pbo"]
    with_cash = noise.copy()
    with_cash[20] = 0.0001  # a fixed-yield column must not hijack the selection
    assert stats.pbo_cscv(with_cash)["pbo"] == pytest.approx(base, abs=0.08)
    holed = noise.copy()
    holed.iloc[5, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        stats.pbo_cscv(holed)


def test_holes_do_not_shorten_the_years():
    days = pd.date_range("2020-01-01", periods=1461, freq="1D", tz="UTC")
    kept = days[:500].append(days[900:])  # a 400-day halt
    r = pd.Series(0.001, index=kept) + np.random.default_rng(1).normal(0, 0.01, len(kept))
    s = stats.summary(r)
    assert s["years"] == pytest.approx(4.0, abs=0.01)
    assert stats.periods_per_year(kept) == pytest.approx(365.25)
    assert stats.effective_periods(kept) == pytest.approx(len(kept) / 4.0, rel=0.01)


def test_bootstrap_scales_to_long_series_and_picks_longer_blocks_for_persistent_returns():
    rng = np.random.default_rng(3)
    long = rng.normal(0.00001, 0.001, 300_000)
    ci = stats.bootstrap_sharpe_ci(long, periods=525_960, n_boot=40, seed=2)
    assert ci["low"] < ci["high"]
    assert stats.automatic_block_length(rng.normal(0, 0.01, 1500)) < 6
    assert stats.automatic_block_length(ar1(rng, 1500, 0.8)) > 12
    persistent = stats.bootstrap_sharpe_ci(0.0005 + ar1(rng, 1500, 0.8), 365.25, n_boot=200, seed=1)
    assert persistent["mean_block"] > 12
