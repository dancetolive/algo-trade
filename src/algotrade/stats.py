"""Performance statistics and overfitting diagnostics.

References
----------
* Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR).
* Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" (DSR).
* Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting" (CSCV).
* Politis & Romano (1994), "The Stationary Bootstrap".
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bars per year inferred from the median bar spacing (crypto trades 24/7)."""
    if len(index) < 2:
        raise ValueError("need at least two bars")
    step = pd.Series(index[1:] - index[:-1]).median().total_seconds()
    return SECONDS_PER_YEAR / step


def _clean(returns) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    return values[np.isfinite(values)]


def sharpe(returns, periods: float) -> float:
    """Annualised Sharpe ratio of per-bar returns (zero risk-free rate)."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino(returns, periods: float) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else float("nan")


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(returns) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def cagr(returns, periods: float) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(r)) - 1.0


def summary(returns: pd.Series, periods: float | None = None) -> dict:
    """Headline statistics for a per-bar return series."""
    if periods is None:
        periods = periods_per_year(returns.index)
    r = _clean(returns)
    growth = float(np.prod(1.0 + r)) if len(r) else float("nan")
    annual = cagr(r, periods)
    mdd = max_drawdown(r)
    return {
        "bars": int(len(r)),
        "years": len(r) / periods,
        "total_return": growth - 1.0,
        "cagr": annual,
        "vol": float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else float("nan"),
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "max_drawdown": mdd,
        "calmar": float(annual / abs(mdd)) if mdd < 0 else float("nan"),
        "skew": float(sps.skew(r)) if len(r) > 2 else float("nan"),
        "kurtosis": float(sps.kurtosis(r, fisher=False)) if len(r) > 3 else float("nan"),
        "worst_bar": float(r.min()) if len(r) else float("nan"),
        "best_bar": float(r.max()) if len(r) else float("nan"),
        "share_positive": float((r > 0).mean()) if len(r) else float("nan"),
    }


def yearly(returns: pd.Series) -> pd.DataFrame:
    """Calendar-year return, volatility-free Sharpe and worst drawdown."""
    periods = periods_per_year(returns.index)
    rows = {}
    for year, chunk in returns.groupby(returns.index.year):
        rows[int(year)] = {
            "return": float(np.prod(1.0 + chunk.to_numpy()) - 1.0),
            "sharpe": sharpe(chunk, periods),
            "max_drawdown": max_drawdown(chunk),
            "bars": int(len(chunk)),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Is the Sharpe ratio statistically distinguishable from luck?
# --------------------------------------------------------------------------- #


def probabilistic_sharpe(returns, benchmark_sr: float = 0.0) -> float:
    """P(true per-bar Sharpe > benchmark_sr), allowing for skew and fat tails.

    ``benchmark_sr`` is in per-bar (not annualised) units.
    """
    r = _clean(returns)
    t = len(r)
    if t < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    skew = sps.skew(r)
    kurt = sps.kurtosis(r, fisher=False)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    z = (sr - benchmark_sr) * np.sqrt(t - 1.0) / np.sqrt(denom)
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: int, trial_sr_variance: float) -> float:
    """Expected best per-bar Sharpe among ``n_trials`` strategies with no skill."""
    if n_trials < 2:
        return 0.0
    a = sps.norm.ppf(1.0 - 1.0 / n_trials)
    b = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(trial_sr_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(returns, n_trials: int, trial_sr_variance: float) -> float:
    """PSR against the Sharpe that luck alone would produce after ``n_trials`` tries.

    ``trial_sr_variance`` is the variance of the per-bar Sharpe ratios across all
    the configurations that were tried (including the ones that were discarded).
    Values above ~0.95 suggest the result is not just the best of many draws.
    """
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, trial_sr_variance))


def trial_sharpes(returns_matrix: pd.DataFrame) -> np.ndarray:
    """Per-bar Sharpe of each column (one column per tried configuration)."""
    values = returns_matrix.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mean / sd, np.nan)


def pbo_cscv(returns_matrix: pd.DataFrame, n_blocks: int = 10) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric CV.

    ``returns_matrix`` holds per-bar returns, one column per configuration that
    was tried. The rows are cut into ``n_blocks`` contiguous blocks; for every
    way of choosing half of them as in-sample, the best in-sample configuration
    is located and its out-of-sample rank recorded. PBO is the share of splits in
    which that configuration lands in the bottom half out of sample.
    """
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks must be an even number >= 4")
    values = returns_matrix.dropna(how="any").to_numpy(dtype=float)
    t, n = values.shape
    if n < 2:
        raise ValueError("need at least two configurations")
    if t < n_blocks * 2:
        raise ValueError("not enough rows for the requested number of blocks")
    edges = np.linspace(0, t, n_blocks + 1).astype(int)
    count = np.diff(edges).astype(float)
    total = np.stack([values[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    square = np.stack([(values[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])

    def block_sharpe(selected: np.ndarray) -> np.ndarray:
        k = count[selected].sum()
        s = total[selected].sum(axis=0)
        ss = square[selected].sum(axis=0)
        mean = s / k
        var = (ss - k * mean**2) / (k - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, mean / np.sqrt(var), -np.inf)

    everything = np.arange(n_blocks)
    logits, is_best, oos_of_best = [], [], []
    for chosen in combinations(range(n_blocks), n_blocks // 2):
        inside = np.array(chosen)
        outside = np.setdiff1d(everything, inside)
        sr_in, sr_out = block_sharpe(inside), block_sharpe(outside)
        best = int(np.argmax(sr_in))
        rank = sps.rankdata(sr_out)[best]  # 1 = worst, n = best
        omega = rank / (n + 1.0)
        logits.append(np.log(omega / (1.0 - omega)))
        is_best.append(sr_in[best])
        oos_of_best.append(sr_out[best])
    logits = np.array(logits)
    return {
        "pbo": float(np.mean(logits <= 0)),
        "splits": int(len(logits)),
        "median_logit": float(np.median(logits)),
        "mean_is_sharpe_of_best": float(np.mean(is_best)),
        "mean_oos_sharpe_of_best": float(np.mean(oos_of_best)),
    }


def bootstrap_sharpe_ci(
    returns,
    periods: float,
    n_boot: int = 2000,
    mean_block: float | None = None,
    level: float = 0.90,
    seed: int = 0,
) -> dict:
    """Stationary-bootstrap confidence interval for the annualised Sharpe ratio."""
    r = _clean(returns)
    t = len(r)
    if t < 20:
        raise ValueError("too few observations to bootstrap")
    if mean_block is None:
        mean_block = max(5.0, t ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    restart = rng.random((n_boot, t)) < 1.0 / mean_block
    restart[:, 0] = True
    fresh = rng.integers(0, t, size=(n_boot, t))
    index = np.empty((n_boot, t), dtype=np.int64)
    index[:, 0] = fresh[:, 0]
    for k in range(1, t):
        index[:, k] = np.where(restart[:, k], fresh[:, k], (index[:, k - 1] + 1) % t)
    sample = r[index]
    sd = sample.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.where(sd > 0, sample.mean(axis=1) / sd, np.nan) * np.sqrt(periods)
    boot = boot[np.isfinite(boot)]
    tail = (1.0 - level) / 2.0
    return {
        "sharpe": sharpe(r, periods),
        "low": float(np.quantile(boot, tail)),
        "high": float(np.quantile(boot, 1.0 - tail)),
        "p_positive": float(np.mean(boot > 0)),
        "level": level,
    }
