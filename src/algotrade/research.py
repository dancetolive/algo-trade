"""One standard way to run, score and log a strategy trial.

Why a harness: the honest significance of a backtest depends on how many
configurations were tried, including the discarded ones. Every call to
:func:`run_trial` is appended to a registry so that the trial count, the
probability of backtest overfitting and the deflated Sharpe ratio can be
computed afterwards from what was actually done, not from what is remembered.

Segments
--------
* design      bars before 2024-01-01: free to explore; choose parameters here.
* validation  2024-01-01 .. 2025-06-30: look rarely; never tune on it.
* hold-out    from 2025-07-01: locked by the data loader, evaluated once at the end.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .backtest import Costs, SimConfig, SimResult, simulate
from .data import store


@dataclass(frozen=True)
class Venue:
    """Trading conditions of a class of venue. Costs are per side, in basis points."""

    name: str
    market: str  # "spot" or "um"
    fee_bps: float
    slippage_bps: float
    allow_short: bool
    max_gross: float
    min_trade_notional: float
    maintenance_margin: float | None = None

    def sim_config(self, **overrides) -> SimConfig:
        base = dict(
            costs=Costs(self.fee_bps, self.slippage_bps),
            allow_short=self.allow_short,
            max_gross=self.max_gross,
            min_trade_notional=self.min_trade_notional,
            maintenance_margin=self.maintenance_margin,
        )
        base.update(overrides)
        return SimConfig(**base)

    def stressed(self, factor: float = 2.0) -> "Venue":
        """Same venue with fees and slippage multiplied by ``factor``."""
        return Venue(
            name=f"{self.name}_x{factor:g}",
            market=self.market,
            fee_bps=self.fee_bps * factor,
            slippage_bps=self.slippage_bps * factor,
            allow_short=self.allow_short,
            max_gross=self.max_gross,
            min_trade_notional=self.min_trade_notional,
            maintenance_margin=self.maintenance_margin,
        )


VENUES = {
    # USDT-quoted spot at taker fees of about 0.10% (large offshore venues; the cheaper local ones).
    "spot": Venue("spot", "spot", fee_bps=10.0, slippage_bps=5.0, allow_short=False, max_gross=1.0, min_trade_notional=10.0),
    # Lira-quoted spot on local venues: taker fees of about 0.22-0.28% and wider spreads.
    "spot_dear": Venue("spot_dear", "spot", fee_bps=25.0, slippage_bps=10.0, allow_short=False, max_gross=1.0, min_trade_notional=10.0),
    # USDT-margined perpetuals at taker fees of about 0.05%; funding is charged separately.
    # The maintenance margin is deliberately above the exchange's lowest tiers (0.4-2.5%).
    "perp": Venue("perp", "um", fee_bps=5.0, slippage_bps=3.0, allow_short=True, max_gross=2.0, min_trade_notional=10.0, maintenance_margin=0.025),
}


def segments(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Boolean masks of the research segments over a bar index."""
    return {
        "design": np.asarray(index < store.DESIGN_END),
        "validation": np.asarray((index >= store.DESIGN_END) & (index < store.HOLDOUT_START)),
        "holdout": np.asarray(index >= store.HOLDOUT_START),
    }


def score(result: SimResult, min_bars: int = 30) -> dict[str, dict]:
    """Per-segment statistics of a simulation, including trading intensity."""
    index = result.returns.index
    periods = stats.periods_per_year(index)
    equity_before = result.equity.shift(1).fillna(result.config.initial_equity)
    out: dict[str, dict] = {}
    for name, mask in segments(index).items():
        if mask.sum() < min_bars:
            continue
        r = result.returns[mask]
        years = mask.sum() / periods
        row = stats.summary(r, periods)
        safe_equity = equity_before[mask].where(equity_before[mask] > 0)
        row.update(
            turnover_per_year=float(result.turnover[mask].sum() / years),
            cost_per_year=float((result.costs[mask] / safe_equity).sum() / years),
            funding_per_year=float((result.funding[mask] / safe_equity).sum() / years),
            avg_gross=float(result.gross[mask].mean()),
            avg_net=float(result.net[mask].mean()),
            time_in_market=float((result.gross[mask] > 0.01).mean()),
            psr=stats.probabilistic_sharpe(r),
            funding_missing_bars=int(result.funding_missing),
            start=str(index[mask][0]),
            end=str(index[mask][-1]),
        )
        out[name] = row
    return out


def registry_root() -> Path:
    override = os.environ.get("ALGOTRADE_TRIALS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "reports" / "trials"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_trial(
    track: str,
    name: str,
    params: dict,
    target_weights: pd.DataFrame,
    open_: pd.DataFrame,
    close: pd.DataFrame,
    venue: Venue,
    *,
    funding: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    initial_equity: float = 2_000.0,
    log: bool = True,
    registry: Path | None = None,
    **sim_overrides,
) -> tuple[dict, SimResult]:
    """Simulate one configuration, score it per segment and append it to the registry.

    ``params`` must describe everything needed to reproduce the configuration.
    Returns ``(record, result)``; ``record["metrics"]["design"]["sharpe"]`` etc.
    Pass ``log=False`` only for smoke tests, never to hide a configuration.
    """
    if venue.market == "um" and (funding is None or high is None or low is None):
        raise ValueError(
            "perpetual trials need funding (research.perp_funding) and high / low; "
            "without them funding costs and liquidations would silently be ignored"
        )
    config = venue.sim_config(initial_equity=initial_equity, **sim_overrides)
    result = simulate(open_, close, target_weights, config, funding=funding, high=high, low=low)
    record = {
        "track": track,
        "name": name,
        "params": _jsonable(params),
        "venue": asdict(venue),
        "sim": _jsonable({k: v for k, v in asdict(config).items() if k != "costs"}),
        "metrics": _jsonable(score(result)),
        "ruined": bool(result.ruined),
        "liquidations": int(result.liquidations),
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if log:
        folder = (registry or registry_root()) / track
        (folder / "returns").mkdir(parents=True, exist_ok=True)
        ledger = folder / "trials.jsonl"
        sequence = sum(1 for _ in ledger.open()) if ledger.exists() else 0
        digest = hashlib.sha1(json.dumps([name, record["params"], record["venue"], record["sim"]], sort_keys=True).encode()).hexdigest()[:10]
        record["trial_id"] = f"{sequence:04d}-{digest}"
        result.returns.to_frame("return").to_parquet(folder / "returns" / f"{record['trial_id']}.parquet")
        with ledger.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    return record, result


def load_trials(track: str, registry: Path | None = None) -> pd.DataFrame:
    """The registry of a track as a table (one row per logged trial)."""
    ledger = (registry or registry_root()) / track / "trials.jsonl"
    if not ledger.exists():
        return pd.DataFrame()
    rows = []
    for line in ledger.open():
        record = json.loads(line)
        row = {"trial_id": record["trial_id"], "name": record["name"], "venue": record["venue"]["name"], "params": json.dumps(record["params"], sort_keys=True)}
        for segment, metrics in record["metrics"].items():
            for key in ("sharpe", "cagr", "max_drawdown", "turnover_per_year", "cost_per_year", "time_in_market"):
                row[f"{segment}_{key}"] = metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows).set_index("trial_id")


def trial_returns(track: str, segment: str | None = "design", venue: str | None = None, registry: Path | None = None) -> pd.DataFrame:
    """Per-bar returns of every logged trial (columns = trial ids), cut to a segment."""
    folder = (registry or registry_root()) / track
    table = load_trials(track, registry)
    if table.empty:
        return pd.DataFrame()
    if venue is not None:
        table = table[table["venue"] == venue]
    columns = {}
    for trial_id in table.index:
        columns[trial_id] = pd.read_parquet(folder / "returns" / f"{trial_id}.parquet")["return"]
    frame = pd.DataFrame(columns).dropna(how="any")
    if segment is not None and len(frame):
        frame = frame[segments(frame.index)[segment]]
    return frame


def overfitting_report(track: str, chosen: str, segment: str = "design", venue: str | None = None, registry: Path | None = None) -> dict:
    """Trial count, PBO and deflated Sharpe for ``chosen`` among a track's logged trials.

    Only trials on the same venue and bar index are comparable; pass ``venue`` to
    select them. Duplicate configurations are kept: they were tried.
    """
    matrix = trial_returns(track, segment, venue, registry)
    if chosen not in matrix.columns:
        raise KeyError(f"{chosen} is not among the comparable trials of {track}")
    usable = matrix.loc[:, matrix.std() > 0]
    sharpes = stats.trial_sharpes(usable)
    sharpes = sharpes[np.isfinite(sharpes)]
    n_trials = int(matrix.shape[1])
    report = {
        "track": track,
        "segment": segment,
        "trials": n_trials,
        "bars": int(len(matrix)),
        "chosen": chosen,
        "chosen_sharpe": stats.sharpe(matrix[chosen], stats.periods_per_year(matrix.index)),
        "psr": stats.probabilistic_sharpe(matrix[chosen]),
    }
    if len(sharpes) >= 2:
        variance = float(np.var(sharpes, ddof=1))
        report["deflated_sharpe"] = stats.deflated_sharpe(matrix[chosen], n_trials, variance)
        report["expected_max_sharpe_annualised"] = stats.expected_max_sharpe(n_trials, variance) * np.sqrt(stats.periods_per_year(matrix.index))
    if usable.shape[1] >= 2 and len(usable) >= 40:
        blocks = 10 if len(usable) >= 200 else 4
        report.update({f"pbo_{k}": v for k, v in stats.pbo_cscv(usable, n_blocks=blocks).items()})
    return _jsonable(report)


def liquid_membership(market: str, top_n: int, start=None, end=None, **universe_kwargs) -> pd.DataFrame:
    """Point-in-time membership of the ``top_n`` most liquid coins (daily rows x symbols).

    Ranks are always taken over the FULL daily universe of the market, so the
    result does not depend on which symbols you later choose to load. Columns are
    the symbols that are members on at least one day. Row t is known at the CLOSE
    of daily bar t; to use it on intraday bars, move it first with
    :func:`algotrade.backtest.decisions_on` (bar "1D") and forward-fill.
    """
    from . import universe

    names = universe.coin_symbols(store.symbols(market, "1d"))
    volume = store.load_panel(market, "1d", names, fields=("quote_volume",), start=start, end=end)["quote_volume"]
    member = universe.top_liquid(volume, top_n, **universe_kwargs)
    return member.loc[:, member.any()]


def ever_liquid(market: str, top_n: int, start=None, end=None, **universe_kwargs) -> list[str]:
    """Coins that were among the ``top_n`` most liquid at any time in the window.

    Use it to choose which symbols to load at finer intervals (1h, 1m) without
    pulling the whole exchange into memory. For the membership on a given day
    use :func:`liquid_membership`; do not recompute ranks on the loaded subset,
    which would let fallen coins back in.
    """
    return sorted(liquid_membership(market, top_n, start, end, **universe_kwargs).columns)


def perp_funding(index: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Funding rates of USDT perpetuals summed into the bars of ``index`` (ready for ``simulate``).

    Cells are NaN where the archive has no funding record for the symbol (before
    its first record, after its last, or no file at all). The simulator charges
    nothing there and counts those position-bars in ``SimResult.funding_missing``.
    """
    from .backtest import align_funding

    raw = store.load_funding(symbols)
    return align_funding(raw, index).reindex(columns=symbols)


def buy_and_hold(open_: pd.DataFrame, close: pd.DataFrame, symbol: str, venue: Venue, initial_equity: float = 2_000.0) -> SimResult:
    """Benchmark: buy ``symbol`` at the first possible open and never trade again."""
    weights = pd.DataFrame(np.nan, index=close.index, columns=[symbol])
    first = close[symbol].first_valid_index()
    weights.loc[first, symbol] = 1.0
    return simulate(open_, close, weights, venue.sim_config(initial_equity=initial_equity, allow_short=False, max_gross=1.0))
