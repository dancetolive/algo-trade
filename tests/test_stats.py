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
