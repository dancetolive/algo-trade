"""Bar-based portfolio simulator.

Timing convention (the single most important thing in this file)
---------------------------------------------------------------
Bars are indexed by their *open* time. ``target_weights.loc[t]`` is a decision
taken at the CLOSE of bar ``t`` and may therefore use every value in rows
``<= t`` (including bar t's own close) and nothing later. The simulator fills
that decision at the OPEN of bar ``t + exec_delay`` (default: the next bar), pays
costs on the traded notional, and earns the open-to-close move of that bar.

Decisions made on coarser bars
------------------------------
A daily bar labelled 2024-01-01 is complete only at 2024-01-02 00:00. If weights
computed on daily bars are passed as they are to a simulation on hourly bars,
the label is read as "close of the 00:00 hourly bar" and the decision is filled
23 hours before the information existed. Always move such weights with
:func:`decisions_on` first.

Accounting
----------
Equity moves by ``units * price change``, minus trading costs, minus funding.
This one identity covers unlevered spot (weights >= 0, gross <= 1) and
USDT-margined perpetuals (shorts and leverage allowed, funding applies); what
differs between them is only the constraints in :class:`SimConfig`.

Rows of ``target_weights`` that are entirely NaN mean "no decision": positions
are left to drift and nothing trades. In any other row a NaN cell means 0. (So
in a single-column frame NaN always reads as "hold", never as "flat".)

Markets without a bar
---------------------
A market with no bar cannot be traded. What happens to a position held in it
depends on what the market did next, which is a fact about the world and not a
choice of the strategy:

* it comes back within ``max_halt_days``: a halt. The position is carried at
  its last close, cannot be changed, pays no funding, still counts towards the
  gross-exposure cap, and the whole move across the halt is booked when the
  market reopens;
* it never comes back, comes back only after a longer absence (a relisting under
  the same ticker), or comes back at a price more than ``max_reopen_ratio`` times
  away (a redenomination or a reused ticker): a delisting. The position is
  settled at the last close at an extra ``delist_slippage_bps`` and stays closed.

Solvency
--------
Shorts or gross exposure above 1 can lose more than the account holds, and can
do so inside a bar. Such configurations therefore require ``high`` and ``low``:
each bar is tested at its adverse extreme (lows for longs, highs for shorts, all
at once, which is conservative), against ``maintenance_margin`` times notional,
or against zero when no maintenance margin is configured.
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
    delist_slippage_bps: float = 100.0  # extra cost when a held market disappears for good
    max_halt_days: float = 7.0  # a longer absence (or one that never ends) is a delisting
    max_reopen_ratio: float = 5.0  # reopening further away than this is a different instrument
    maintenance_margin: float | None = None  # e.g. 0.025; None tests for bankruptcy only
    liquidation_fee_bps: float = 100.0
    lot_step: dict | None = None  # symbol -> order size step in units; holdings round towards zero


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
    funding_missing: int = 0  # position-bars carried without any funding data (charged nothing)

    @property
    def gross(self) -> pd.Series:
        return self.weights.abs().sum(axis=1)

    @property
    def net(self) -> pd.Series:
        return self.weights.sum(axis=1)


def decisions_on(weights: pd.DataFrame, bar, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Move decisions made on coarser bars onto a finer bar index, without look-ahead.

    ``weights`` is indexed by the open times of bars of length ``bar`` (for
    example "1D"), so row t is known only at ``t + bar``. It is re-labelled to the
    last bar of ``index`` that opens before ``t + bar``; that bar closes at
    ``t + bar``, and the simulator fills the decision at the following open.
    Rows that would land before the first fine bar are dropped.
    """
    known_at = weights.index + pd.Timedelta(bar)
    position = index.searchsorted(known_at, side="left") - 1
    keep = position >= 0
    out = weights[keep].set_axis(index[position[keep]])
    return out[~out.index.duplicated(keep="last")]


def align_funding(funding: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sum funding rates into the bar during which the position was carried.

    A payment stamped at time ``s`` is charged to whoever holds the position at
    ``s``. It is assigned to the bar with ``open < s <= open + bar length``, i.e.
    a payment exactly on a bar boundary belongs to the bar that just ended,
    because the simulator's position during that bar is the one held at ``s``.
    A payment stamped inside a hole of the bar index goes to the bar before the
    hole (that bar's position is the one carried through it).

    Bars before a symbol's first record and after its last one are NaN ("no
    data"), which the simulator reports instead of silently charging nothing;
    bars inside the covered span with no payment due are 0.
    """
    out = pd.DataFrame(np.nan, index=index, columns=funding.columns)
    if funding.empty or len(index) < 2:
        return out
    step = pd.Series(index[1:] - index[:-1]).median()
    stamps = funding.index
    position = index.searchsorted(stamps, side="left") - 1
    valid = position >= 0
    last_bar = position == len(index) - 1
    valid[last_bar] &= np.asarray(stamps[last_bar] <= index[-1] + step)
    values = funding.to_numpy(dtype=float)
    result = np.full(out.shape, np.nan)
    for k in range(values.shape[1]):
        usable = valid & np.isfinite(values[:, k])
        if not usable.any():
            continue
        rows = position[usable]
        first, last = rows.min(), rows.max()
        column = np.zeros(last - first + 1)
        np.add.at(column, rows - first, values[usable, k])
        result[first : last + 1, k] = column
    return pd.DataFrame(result, index=index, columns=funding.columns)


def _terminal_gaps(
    live: np.ndarray,
    index: pd.DatetimeIndex,
    max_halt_days: float,
    open_: np.ndarray | None = None,
    close: np.ndarray | None = None,
    max_reopen_ratio: float = 5.0,
) -> np.ndarray:
    """True where a market has no bar and will not resume as the same instrument."""
    n, m = live.shape
    terminal = np.zeros((n, m), dtype=bool)
    stamps = (index.tz_convert(None) if index.tz is not None else index).to_numpy()
    limit = np.timedelta64(int(round(max_halt_days * 86_400_000)), "ms")
    for j in range(m):
        dead = ~live[:, j]
        if not dead.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([False], dead, [False])).astype(np.int8)))
        for start, stop in zip(edges[0::2], edges[1::2]):
            ends = stop >= n or stamps[stop] - stamps[start] > limit
            if not ends and start > 0 and open_ is not None and close is not None:
                ratio = open_[stop, j] / close[start - 1, j]
                ends = not (1.0 / max_reopen_ratio <= ratio <= max_reopen_ratio)
            if ends:
                terminal[start:stop, j] = True
    return terminal


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

    ``funding`` must be the output of :func:`align_funding` on this bar index:
    cell (t, s) is the sum of funding rates charged while holding s during bar t,
    NaN where no funding data exists. ``high`` / ``low`` are required whenever
    shorts or leverage are allowed.
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
    live_all = np.isfinite(O) & np.isfinite(C)

    F = None
    if funding is not None:
        absent = [c for c in columns if c not in funding.columns]
        if absent:
            raise KeyError(f"funding frame has no column for: {absent[:5]}")
        if not funding.index.equals(index):
            raise ValueError("funding is not on the bar index; pass the output of align_funding(funding, bars.index)")
        F = funding.reindex(columns=columns).to_numpy(dtype=float)

    can_go_broke = config.allow_short or config.max_gross > 1.0
    check_extremes = high is not None and low is not None
    if can_go_broke and not check_extremes:
        raise ValueError("shorts or leverage need high and low for the intrabar solvency test")
    if check_extremes:
        H = high.reindex(index=index, columns=columns).to_numpy(dtype=float)
        L = low.reindex(index=index, columns=columns).to_numpy(dtype=float)
        if (live_all & ~(np.isfinite(H) & np.isfinite(L))).any():
            raise ValueError("high / low are missing on bars that have an open and a close")
    margin = config.maintenance_margin or 0.0

    n, m = C.shape
    cost_rate = config.costs.rate
    delist_rate = config.delist_slippage_bps / 1e4
    liq_rate = config.liquidation_fee_bps / 1e4
    terminal = _terminal_gaps(live_all, index, config.max_halt_days, O, C, config.max_reopen_ratio)
    lot = np.zeros(m)
    if config.lot_step:
        for k, name in enumerate(columns):
            lot[k] = float(config.lot_step.get(name, 0.0))
    has_lot = lot > 0
    safe_lot = np.where(has_lot, lot, 1.0)

    units = np.zeros(m)
    last_close = np.full(m, np.nan)
    equity = float(config.initial_equity)
    ruined = equity <= 0
    liquidations = 0
    funding_missing = 0

    equity_path = np.empty(n)
    weight_path = np.zeros((n, m))
    turnover_path = np.zeros(n)
    cost_path = np.zeros(n)
    funding_path = np.zeros(n)

    for i in range(n):
        o, c = O[i], C[i]
        live = live_all[i]
        cost_i = 0.0
        traded_i = 0.0

        # 1. A held market that has stopped for good is settled at its last close, at a penalty.
        gone = (units != 0) & terminal[i]
        if gone.any():
            notional = float(np.sum(np.abs(units[gone]) * last_close[gone]))
            cost = notional * (cost_rate + delist_rate)
            equity -= cost
            cost_i += cost
            traded_i += notional
            units[gone] = 0.0

        # 2. Gap from the previous close to this open (halted markets keep their last price).
        active = (units != 0) & live
        if active.any():
            equity += float(np.sum(units[active] * (o[active] - last_close[active])))
        if equity <= 0 and not ruined:
            equity, ruined = 0.0, True
            units[:] = 0.0
        equity_at_open = equity
        frozen = (units != 0) & ~live

        # 3. Fill the decision taken exec_delay bars ago.
        j = i - config.exec_delay
        if j >= 0 and decided[j] and equity_at_open > 0 and not ruined:
            target = np.where(live, W[j], 0.0)
            frozen_gross = float(np.sum(np.abs(units[frozen]) * last_close[frozen])) / equity_at_open
            room = max(config.max_gross - frozen_gross, 0.0)
            gross = float(np.abs(target).sum())
            if gross > room:
                target = target * (room / gross)
            safe_open = np.where(live, o, 1.0)
            current = np.where(live, units * safe_open, 0.0) / equity_at_open
            diff = target - current
            closing = (target == 0.0) & (units != 0) & live
            trade = live & (diff != 0.0)
            if config.rebalance_band > 0:
                trade &= (np.abs(diff) >= config.rebalance_band) | closing
            wanted = np.where(trade, units + diff * equity_at_open / safe_open, units)
            if has_lot.any():
                rounded = np.trunc(wanted / safe_lot + np.sign(wanted) * 1e-9) * lot
                wanted = np.where(has_lot & trade, rounded, wanted)
            wanted[closing] = 0.0
            delta = np.where(live, (wanted - units) * safe_open, 0.0)
            if config.min_trade_notional > 0:
                dust = (np.abs(delta) < config.min_trade_notional) & ~closing
                wanted[dust] = units[dust]
                delta[dust] = 0.0
            traded = float(np.abs(delta).sum())
            if traded > 0:
                units = wanted
                cost = traded * cost_rate
                equity -= cost
                cost_i += cost
                traded_i += traded

        # 4. Solvency at the bar's adverse extreme (conservative: every position at its worst at once).
        active = (units != 0) & live
        if check_extremes and active.any() and not ruined:
            adverse = np.where(units > 0, L[i], H[i])
            worst_equity = equity + float(np.sum(units[active] * (adverse[active] - o[active])))
            notional = float(np.sum(np.abs(units[active]) * adverse[active]))
            notional += float(np.sum(np.abs(units[frozen]) * last_close[frozen]))
            if worst_equity <= margin * notional:
                fee = notional * liq_rate
                equity = max(worst_equity - fee, 0.0)
                cost_i += fee
                traded_i += notional
                units[:] = 0.0
                active = units != 0
                liquidations += 1

        # 5. Open -> close move, then funding on what was carried through the bar.
        if active.any():
            equity += float(np.sum(units[active] * (c[active] - o[active])))
            if F is not None:
                rates = F[i][active]
                known = np.isfinite(rates)
                paid = float(np.sum(units[active][known] * c[active][known] * rates[known]))
                equity -= paid
                funding_path[i] = paid
                funding_missing += int((~known).sum())

        if equity <= 0 and not ruined:
            ruined = True
        if ruined:
            equity = 0.0
            units[:] = 0.0

        last_close = np.where(live, c, last_close)
        equity_path[i] = equity
        if equity > 0:
            weight_path[i] = np.where(units != 0, units * np.where(units != 0, last_close, 0.0), 0.0) / equity
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
        ruined=bool(ruined),
        config=config,
        funding_missing=funding_missing,
    )
