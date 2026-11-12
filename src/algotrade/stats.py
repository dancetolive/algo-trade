"""Performance statistics and overfitting diagnostics.

References
----------
* Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR).
* Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" (DSR).
* Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting" (CSCV).
* Politis & Romano (1994), "The Stationary Bootstrap"; Politis & White (2004)
  and Patton, Politis & White (2009) for the automatic block length.

Two ways a Sharpe ratio flatters, both handled here
---------------------------------------------------
* Serial correlation. Annualising with sqrt(periods) assumes independent
  returns. Smooth return streams (funding carry is the standard example) have
  strongly autocorrelated returns: the textbook Sharpe can be several times too
  high. :func:`autocorr_inflation` measures the long-run variance ratio ``q``;
  ``summary`` reports ``sharpe_adj = sharpe / sqrt(q)`` next to the raw figure,
  and the significance measures use it.
* Sparse records. A strategy that trades on a handful of bars can show a huge
  Sharpe from one lucky bar. The significance measures return NaN unless there
  are enough active and enough losing bars to say anything.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MIN_ACTIVE = 30  # non-zero returns needed before significance is reported
MIN_LOSING = 5  # negative returns needed before significance is reported


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bars per year inferred from the median bar spacing (crypto trades 24/7)."""
    if len(index) < 2:
        raise ValueError("need at least two bars")
    step = pd.Series(index[1:] - index[:-1]).median().total_seconds()
    return SECONDS_PER_YEAR / step


def effective_periods(index: pd.DatetimeIndex) -> float:
    """Bars per calendar year actually present in ``index``.

    Unlike :func:`periods_per_year` this accounts for holes: a series with a
    400-day halt has fewer bars per year, so annualised figures are not inflated.
    """
    if len(index) < 2:
        raise ValueError("need at least two bars")
    step = pd.Series(index[1:] - index[:-1]).median()
    span = (index[-1] - index[0] + step).total_seconds()
    return len(index) * SECONDS_PER_YEAR / span


def _clean(returns) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    return values[np.isfinite(values)]


def _spread(r: np.ndarray) -> float:
    """Sample standard deviation, or 0.0 when the series is constant up to round-off."""
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    return sd if sd > 1e-12 * max(1.0, abs(float(r.mean()))) else 0.0


def sharpe(returns, periods: float) -> float:
    """Annualised Sharpe ratio of per-bar returns (zero risk-free rate, no autocorrelation adjustment)."""
    r = _clean(returns)
    sd = _spread(r)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino(returns, periods: float) -> float:
    r = _clean(returns)
    if _spread(r) == 0:
        return float("nan")
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else float("nan")


def autocorr_inflation(returns, lags: int | None = None) -> float:
    """Long-run variance ratio ``q >= 1`` of a return series (Bartlett weights).

    ``q`` is how much larger the variance of long-horizon returns is than
    independence would imply: the honest annualised volatility is
    ``vol * sqrt(q)`` and the honest Sharpe ratio ``sharpe / sqrt(q)``. ``lags``
    defaults to 30 (about a month of daily returns), shortened for short series.
    """
    r = _clean(returns)
    n = len(r)
    if n < 10 or _spread(r) == 0:
        return 1.0
    if lags is None:
        lags = 30
    lags = int(max(1, min(lags, n // 4)))
    x = r - r.mean()
    spectrum = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(spectrum * np.conj(spectrum))[: lags + 1] / n
    if acov[0] <= 0:
        return 1.0
    weights = 1.0 - np.arange(1, lags + 1) / (lags + 1.0)
    return float(max(1.0, 1.0 + 2.0 * np.sum(weights * acov[1:]) / acov[0]))


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
    """Headline statistics for a per-bar return series.

    When ``periods`` is not given it is taken from the calendar span of the
    index (bars per year actually present), so holes do not inflate annualised
    figures. ``sharpe`` is the textbook figure; ``sharpe_adj`` and ``vol_adj``
    correct it for serial correlation (see the module docstring) and are the
    ones to quote.
    """
    if periods is None:
        periods = effective_periods(returns.index)
    r = _clean(returns)
    growth = float(np.prod(1.0 + r)) if len(r) else float("nan")
    annual = cagr(r, periods)
    mdd = max_drawdown(r)
    raw = sharpe(r, periods)
    q = autocorr_inflation(r)
    sd = _spread(r)
    return {
        "bars": int(len(r)),
        "years": len(r) / periods,
        "total_return": growth - 1.0,
        "cagr": annual,
        "vol": float(sd * np.sqrt(periods)) if sd > 0 else float("nan"),
        "sharpe": raw,
        "autocorr_inflation": q,
        "sharpe_adj": float(raw / np.sqrt(q)) if np.isfinite(raw) else float("nan"),
        "vol_adj": float(sd * np.sqrt(periods * q)) if sd > 0 else float("nan"),
        "sortino": sortino(r, periods),
        "max_drawdown": mdd,
        "calmar": float(annual / abs(mdd)) if mdd < 0 else float("nan"),
        "skew": float(sps.skew(r)) if len(r) > 2 and sd > 0 else float("nan"),
        "kurtosis": float(sps.kurtosis(r, fisher=False)) if len(r) > 3 and sd > 0 else float("nan"),
        "worst_bar": float(r.min()) if len(r) else float("nan"),
        "best_bar": float(r.max()) if len(r) else float("nan"),
        "share_positive": float((r > 0).mean()) if len(r) else float("nan"),
        "active_bars": int(np.count_nonzero(r)),
        "losing_bars": int(np.sum(r < 0)),
    }


def yearly(returns: pd.Series) -> pd.DataFrame:
    """Calendar-year return, Sharpe and worst drawdown."""
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


def _enough_activity(r: np.ndarray, min_active: int, min_losing: int) -> bool:
    return np.count_nonzero(r) >= min_active and int(np.sum(r < 0)) >= min_losing


def probabilistic_sharpe(
    returns,
    benchmark_sr: float = 0.0,
    min_active: int = MIN_ACTIVE,
    min_losing: int = MIN_LOSING,
    lags: int | None = None,
) -> float:
    """P(true per-bar Sharpe > benchmark_sr), allowing for skew, fat tails and serial correlation.

    ``benchmark_sr`` is in per-bar (not annualised) units. Returns NaN when the
    record is too thin to judge: fewer than ``min_active`` non-zero returns or
    fewer than ``min_losing`` losing ones (a normal approximation certifies a
    single lucky bar as "97.7% significant").
    """
    r = _clean(returns)
    t = len(r)
    sd = _spread(r)
    if t < 3 or sd == 0 or not _enough_activity(r, min_active, min_losing):
        return float("nan")
    sr = r.mean() / sd
    skew = sps.skew(r)
    kurt = sps.kurtosis(r, fisher=False)
    denom = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) * autocorr_inflation(r, lags)
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


def deflated_sharpe(returns, n_trials: int, trial_sr_variance: float, **psr_kwargs) -> float:
    """PSR against the Sharpe that luck alone would produce after ``n_trials`` tries.

    ``trial_sr_variance`` is the variance of the per-bar Sharpe ratios across all
    the configurations that were tried (including the ones that were discarded).
    Values above ~0.95 suggest the result is not just the best of many draws.
    """
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, trial_sr_variance), **psr_kwargs)


def trial_sharpes(returns_matrix: pd.DataFrame) -> np.ndarray:
    """Per-bar Sharpe of each column (one column per tried configuration)."""
    values = returns_matrix.to_numpy(dtype=float)
    out = np.full(values.shape[1], np.nan)
    for k in range(values.shape[1]):
        r = _clean(values[:, k])
        sd = _spread(r)
        if sd > 0:
            out[k] = r.mean() / sd
    return out


def pbo_cscv(returns_matrix: pd.DataFrame, n_blocks: int = 10) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric CV.

    ``returns_matrix`` holds per-bar returns, one column per configuration that
    was tried, all on the same bars (NaN is refused: aligning trials is the
    caller's decision, see ``research.trial_returns``). The rows are cut into
    ``n_blocks`` contiguous blocks; for every way of choosing half of them as
    in-sample, the best in-sample configuration is located and its
    out-of-sample rank recorded. PBO is the share of splits in which that
    configuration lands in the bottom half out of sample.
    """
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks must be an even number >= 4")
    values = returns_matrix.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("returns matrix contains NaN: align the trials on a common set of bars first")
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
            # a constant column has a variance of pure round-off: it has no Sharpe ratio
            return np.where(var > 1e-12 * np.maximum(mean**2, 1e-30), mean / np.sqrt(np.abs(var)), -np.inf)

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


def automatic_block_length(returns) -> float:
    """Mean block length for the stationary bootstrap, chosen from the data.

    Politis & White (2004) with the correction of Patton, Politis & White (2009):
    about 1-2 bars for independent returns, tens of bars for persistent ones.
    """
    x = _clean(returns)
    n = len(x)
    if n < 20 or _spread(x) == 0:
        return 1.0
    x = x - x.mean()
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    acov = np.array([np.dot(x[k:], x[: n - k]) / n for k in range(mmax + 1)])
    rho = acov / acov[0]
    critical = 2.0 * np.sqrt(np.log10(n) / n)
    mhat = None
    for m in range(0, mmax - kn + 1):
        if np.all(np.abs(rho[m + 1 : m + kn + 1]) < critical):
            mhat = m
            break
    if mhat is None:
        big = np.flatnonzero(np.abs(rho[1:]) > critical)
        mhat = int(big.max() + 1) if len(big) else 1
    width = min(2 * max(mhat, 1), mmax)
    k = np.arange(1, width + 1)
    taper = np.where(k / width <= 0.5, 1.0, 2.0 * (1.0 - k / width))
    g = 2.0 * np.sum(taper * k * acov[1 : width + 1])
    long_run = acov[0] + 2.0 * np.sum(taper * acov[1 : width + 1])
    d = 2.0 * long_run**2
    if d <= 0 or g == 0:
        return 1.0
    block = (2.0 * g**2 / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(block, 1.0), np.ceil(min(3.0 * np.sqrt(n), n / 3.0))))


def bootstrap_sharpe_ci(
    returns,
    periods: float,
    n_boot: int = 2000,
    mean_block: float | None = None,
    level: float = 0.90,
    seed: int = 0,
    min_active: int = MIN_ACTIVE,
    min_losing: int = MIN_LOSING,
) -> dict:
    """Stationary-bootstrap confidence interval for the annualised Sharpe ratio.

    The mean block length is chosen from the data unless given (and never below
    5 bars). A resample with no variation counts as a Sharpe of 0; it is not
    dropped, which would condition on the lucky bars being drawn. For very
    persistent series even long blocks under-cover with a few years of data:
    read the interval as a lower bound on the uncertainty.
    """
    r = _clean(returns)
    t = len(r)
    if t < 20:
        raise ValueError("too few observations to bootstrap")
    if not _enough_activity(r, min_active, min_losing):
        raise ValueError(
            f"record too thin to bootstrap: {np.count_nonzero(r)} active and {int(np.sum(r < 0))} losing bars "
            f"(need {min_active} and {min_losing})"
        )
    if mean_block is None:
        mean_block = max(5.0, automatic_block_length(r))
    rng = np.random.default_rng(seed)
    position = np.arange(t)
    chunk = max(1, int(4_000_000 // t))
    pieces = []
    for start in range(0, n_boot, chunk):
        rows = min(chunk, n_boot - start)
        restart = rng.random((rows, t)) < 1.0 / mean_block
        restart[:, 0] = True
        fresh = rng.integers(0, t, size=(rows, t))
        origin = np.maximum.accumulate(np.where(restart, position, 0), axis=1)
        index = (np.take_along_axis(fresh, origin, axis=1) + position - origin) % t
        sample = r[index]
        sd = sample.std(axis=1, ddof=1)
        mean = sample.mean(axis=1)
        ok = sd > 1e-12 * np.maximum(1.0, np.abs(mean))
        pieces.append(np.where(ok, mean / np.where(ok, sd, 1.0), 0.0) * np.sqrt(periods))
    boot = np.concatenate(pieces)
    tail = (1.0 - level) / 2.0
    return {
        "sharpe": sharpe(r, periods),
        "low": float(np.quantile(boot, tail)),
        "high": float(np.quantile(boot, 1.0 - tail)),
        "p_positive": float(np.mean(boot > 0)),
        "level": level,
        "mean_block": float(mean_block),
    }
