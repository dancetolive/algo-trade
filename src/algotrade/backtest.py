"""Bar-based portfolio simulator.

Timing convention (the single most important thing in this file)
---------------------------------------------------------------
Bars are indexed by their *open* time. ``target_weights.loc[t]`` is a decision
taken at the CLOSE of bar ``t`` and may therefore use every value in rows
``<= t`` (including bar t's own close) and nothing later. The simulator fills
that decision at the OPEN of bar ``t + exec_delay`` (default: the next bar), pays
costs on the traded notional, and earns the open-to-close move of that bar.

Accounting
----------
Equity moves by ``units * price change``, minus trading costs, minus funding.
This one identity covers unlevered spot (weights >= 0, gross <= 1) and
USDT-margined perpetuals (shorts and leverage allowed, funding applies); what
differs between them is only the constraints in :class:`SimConfig`.

Rows of ``target_weights`` that are entirely NaN mean "no decision": positions
are left to drift and nothing trades. In any other row a NaN cell means 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    """Per-side trading costs in basis points of traded notional."""

    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    @property
    def rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 1e4


@dataclass(frozen=True)
class SimConfig:
    initial_equity: float = 1_000.0
    costs: Costs = Costs()
    allow_short: bool = False
    max_gross: float = 1.0  # cap on sum(|weights|); targets above it are scaled down
    rebalance_band: float = 0.0  # skip trades that change a weight by less than this
    min_trade_notional: float = 0.0  # skip trades smaller than this (exchange minimum)
    exec_delay: int = 1  # bars between the decision and its fill; must be >= 1
    delist_slippage_bps: float = 100.0  # extra cost when a held market disappears
    maintenance_margin: float | None = None  # e.g. 0.05; None disables the liquidation check
    liquidation_fee_bps: float = 100.0


@dataclass
class SimResult:
    equity: pd.Series  # at each bar's close
    returns: pd.Series  # per-bar net return
    weights: pd.DataFrame  # realised weights at each bar's close
    turnover: pd.Series  # traded notional / equity, per bar (one-way)
    costs: pd.Series  # fees + slippage paid, in account currency
    funding: pd.Series  # funding paid (positive) or received (negative)
    liquidations: int
    ruined: bool
    config: SimConfig

    @property
    def gross(self) -> pd.Series:
        return self.weights.abs().sum(axis=1)

    @property
    def net(self) -> pd.Series:
        return self.weights.sum(axis=1)


def align_funding(funding: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sum funding rates into the bar during which the position was carried.

    A payment stamped at time ``s`` is charged to whoever holds the position at
    ``s``. It is assigned to the bar with ``open < s <= open + bar length``, i.e.
    a payment exactly on a bar boundary belongs to the bar that just ended,
    because the simulator's position during that bar is the one held at ``s``.
    """
    out = pd.DataFrame(0.0, index=index, columns=funding.columns)
    if funding.empty or len(index) < 2:
        return out
    step = pd.Series(index[1:] - index[:-1]).median()
    stamps = funding.index
    position = index.searchsorted(stamps, side="left") - 1
    valid = position >= 0
    valid[valid] &= stamps[valid] <= index[position[valid]] + step
    values = np.nan_to_num(funding.to_numpy(dtype=float), nan=0.0)
    accumulator = np.zeros(out.shape)
    np.add.at(accumulator, position[valid], values[valid])
    return pd.DataFrame(accumulator, index=index, columns=funding.columns)


def simulate(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: SimConfig = SimConfig(),
    funding: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> SimResult:
    """Simulate ``target_weights`` on the given bars. See the module docstring.

    ``funding`` must already be aligned to the bar index (see :func:`align_funding`):
    cell (t, s) is the sum of funding rates charged while holding s during bar t.
    ``high`` / ``low`` are only needed for the optional liquidation check.
    """
    if config.exec_delay < 1:
        raise ValueError("exec_delay must be >= 1; a decision cannot fill before it is made")
    index = close.index
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("bar index must be strictly increasing")
    columns = list(target_weights.columns)
    missing = [c for c in columns if c not in close.columns or c not in open_.columns]
    if missing:
        raise KeyError(f"no price data for: {missing[:5]}")
    extra_rows = target_weights.index.difference(index)
    if len(extra_rows):
        raise ValueError(f"{len(extra_rows)} weight rows have no matching bar (first: {extra_rows[0]})")

    O = open_.reindex(index=index, columns=columns).to_numpy(dtype=float)
    C = close.reindex(index=index, columns=columns).to_numpy(dtype=float)
    W = target_weights.reindex(index=index).to_numpy(dtype=float)
    decided = ~np.all(np.isnan(W), axis=1)
    W = np.nan_to_num(W, nan=0.0)
    if not config.allow_short and (W < 0).any():
        raise ValueError("negative target weights but allow_short is False")
    F = None
    if funding is not None:
        F = funding.reindex(index=index, columns=columns).fillna(0.0).to_numpy(dtype=float)
    check_liq = config.maintenance_margin is not None
    if check_liq and (high is None or low is None):
        raise ValueError("the liquidation check needs high and low")
    if check_liq:
        H = high.reindex(index=index, columns=columns).to_numpy(dtype=float)
        L = low.reindex(index=index, columns=columns).to_numpy(dtype=float)

    n, m = C.shape
    cost_rate = config.costs.rate
    delist_rate = config.delist_slippage_bps / 1e4
    liq_rate = config.liquidation_fee_bps / 1e4

    units = np.zeros(m)
    last_close = np.full(m, np.nan)
    equity = float(config.initial_equity)
    ruined = False
    liquidations = 0

    equity_path = np.empty(n)
    weight_path = np.zeros((n, m))
    turnover_path = np.zeros(n)
    cost_path = np.zeros(n)
    funding_path = np.zeros(n)

    for i in range(n):
        o, c = O[i], C[i]
        live = np.isfinite(o) & np.isfinite(c)
        cost_i = 0.0
        traded_i = 0.0

        # 1. A held market that has no bar any more is closed at its last close, at a penalty.
        gone = (units != 0) & ~live
        if gone.any():
            notional = float(np.sum(np.abs(units[gone]) * last_close[gone]))
            cost = notional * (cost_rate + delist_rate)
            equity -= cost
            cost_i += cost
            traded_i += notional
            units[gone] = 0.0

        # 2. Overnight gap: previous close -> this open.
        held = units != 0
        if held.any():
            equity += float(np.sum(units[held] * (o[held] - last_close[held])))
        equity_at_open = equity

        # 3. Fill the decision taken exec_delay bars ago.
        j = i - config.exec_delay
        if j >= 0 and decided[j] and equity_at_open > 0 and not ruined:
            target = np.where(live, W[j], 0.0)
            gross = float(np.abs(target).sum())
            if gross > config.max_gross:
                target = target * (config.max_gross / gross)
            safe_open = np.where(live, o, 1.0)
            current = np.where(live, units * safe_open, 0.0) / equity_at_open
            diff = target - current
            closing = (target == 0.0) & (units != 0) & live
            trade = live & (diff != 0.0)
            if config.rebalance_band > 0:
                trade &= (np.abs(diff) >= config.rebalance_band) | closing
            delta = np.where(trade, diff * equity_at_open, 0.0)
            if config.min_trade_notional > 0:
                delta[(np.abs(delta) < config.min_trade_notional) & ~closing] = 0.0
            traded = float(np.abs(delta).sum())
            if traded > 0:
                units = units + delta / safe_open
                units[closing] = 0.0
                cost = traded * cost_rate
                equity -= cost
                cost_i += cost
                traded_i += traded

        # 4. Intrabar liquidation check on worst-case prices (conservative).
        held = units != 0
        if check_liq and held.any():
            adverse = np.where(units > 0, L[i], H[i])
            adverse = np.where(np.isfinite(adverse), adverse, o)
            worst_equity = equity + float(np.sum(units[held] * (adverse[held] - o[held])))
            notional = float(np.sum(np.abs(units[held]) * adverse[held]))
            if worst_equity <= config.maintenance_margin * notional:
                fee = notional * liq_rate
                equity = max(worst_equity - fee, 0.0)
                cost_i += fee
                traded_i += notional
                units[:] = 0.0
                held = units != 0
                liquidations += 1

        # 5. Open -> close move, then funding on what was carried through the bar.
        if held.any():
            equity += float(np.sum(units[held] * (c[held] - o[held])))
            if F is not None:
                paid = float(np.sum(units[held] * c[held] * F[i][held]))
                equity -= paid
                funding_path[i] = paid

        if equity <= 0:
            equity = 0.0
            units[:] = 0.0
            ruined = True

        last_close = np.where(np.isfinite(c), c, last_close)
        equity_path[i] = equity
        if equity > 0:
            weight_path[i] = np.where(units != 0, units * np.where(live, c, 0.0), 0.0) / equity
        cost_path[i] = cost_i
        turnover_path[i] = traded_i / equity_at_open if equity_at_open > 0 else 0.0

    equity_series = pd.Series(equity_path, index=index, name="equity")
    previous = np.concatenate([[config.initial_equity], equity_path[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(previous > 0, equity_path / previous - 1.0, 0.0)
    return SimResult(
        equity=equity_series,
        returns=pd.Series(ret, index=index, name="return"),
        weights=pd.DataFrame(weight_path, index=index, columns=columns),
        turnover=pd.Series(turnover_path, index=index, name="turnover"),
        costs=pd.Series(cost_path, index=index, name="costs"),
        funding=pd.Series(funding_path, index=index, name="funding"),
        liquidations=liquidations,
        ruined=ruined,
        config=config,
    )
