"""Part B: does waiting for a 1-minute structure improve the entry price of the daily trend trade?

    python -I part_b.py design
    python -I part_b.py validation        (needs SELECTION.json; uses only the selected structures and their mirrors)

Rule (research/baseline/trend_baseline.py): target weight of BTC and ETH = blend of the 50 / 100 / 200-day
moving-average filters on spot daily closes; a decision at the daily close is filled at the next daily
open, 00:00 UTC. For every day on which the target weight of a coin goes UP the reference fill is the
00:00 open of that day. The alternative waits, for up to 4 hours or up to 24 hours, for the first bullish
occurrence of a structure flagged at the close of a 1-minute bar at or after 00:00, and buys at the open
of the next minute; if none occurs it buys at the open of the bar at the end of the window.

    improvement (buy)  = (P_open - P_wait) / P_open        positive = bought cheaper
    improvement (sell) = (P_wait - P_open) / P_open        positive = sold higher     (weight goes DOWN, bearish mirror)

Benchmarks on the same days, to separate "the structure" from "waiting at all":
    delay   buy at the end of the window, whatever happened
    twap    buy in equal parts at every minute's open over the window
Also reported for power: the same comparison on EVERY day of the segment (not only rule-change days).

Inference: one observation per (coin, day); t-statistics clustered by day (BTC and ETH share the day).
Fees do not enter: both alternatives make one purchase.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "baseline"))

import definitions as defs  # noqa: E402
import eventstudy as es  # noqa: E402
import run_events as runner  # noqa: E402

sys.dont_write_bytecode = True  # the baseline script lives outside this directory: leave no cache file next to it
import trend_baseline as tb  # noqa: E402

sys.dont_write_bytecode = False
from algotrade.data import store  # noqa: E402

WINDOWS = (240, 1440)
SYMBOLS = tb.SYMBOLS
# structure -> (definition whose flag is the bullish occurrence, definition whose flag is the bearish mirror)
STRUCTURES = {
    "move5_abs2pct": ("move5_abs2pct_dn", "move5_abs2pct_up"),
    "move1_z3": ("move1_z3_dn", "move1_z3_up"),
    "fund_post5_big": ("fund_post5_big", "fund_post5_big"),
}
SELECTED_TO_STRUCTURE = {"move5_abs2pct_dn": "move5_abs2pct", "move1_z3_dn": "move1_z3", "fund_post5_big": "fund_post5_big"}


def rule_changes(segment: str) -> pd.DataFrame:
    """Days on which the daily rule raises or lowers a coin's target weight, by fill day (00:00 UTC)."""
    _, load_end, first_day, end_day = runner.SEGMENTS[segment]
    panel = store.load_panel("spot", "1d", SYMBOLS, end=load_end)
    weights = tb.blend_weights(panel)
    change = weights.diff()
    rows = []
    for symbol in SYMBOLS:
        for decided, delta in change[symbol].dropna().items():
            if delta == 0:
                continue
            fill = decided + pd.Timedelta(days=1)  # decided at the close of this daily bar, filled at the next open
            if first_day + pd.Timedelta(days=2) <= fill < end_day - pd.Timedelta(days=1):
                rows.append({"symbol": symbol, "fill": fill, "kind": "increase" if delta > 0 else "decrease",
                             "weight_before": weights[symbol].loc[decided] - delta, "weight_after": weights[symbol].loc[decided],
                             "daily_open": panel["open"][symbol].get(fill, np.nan)})
    return pd.DataFrame(rows)


def structure_flags(grid, feats, funding, names) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    wanted = set()
    for name in names:
        wanted.update(STRUCTURES[name])
    found = {d.name: d for d in defs.generate(grid, feats, funding, only=wanted)}
    out = {}
    for name in names:
        bull_name, bear_name = STRUCTURES[name]
        if name == "fund_post5_big":
            d = found[bull_name]
            out[name] = (d.mask & (d.direction > 0), d.mask & (d.direction < 0))
        else:
            out[name] = (found[bull_name].mask, found[bear_name].mask)
    return out


def next_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True at or after each position (len(mask) if none)."""
    n = len(mask)
    position = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(position[::-1])[::-1]


def timing_table(market: str, segment: str, names: list[str], trades: pd.DataFrame, all_days: bool) -> pd.DataFrame:
    """One row per (coin, day, side, structure, window) with the reference price and the waited price."""
    _, load_end, first_day, end_day = runner.SEGMENTS[segment]
    rows = []
    for symbol in SYMBOLS:
        grid, feats, _, inside, _, _ = runner.load_segment(market, symbol, segment)
        load_start = runner.SEGMENTS[segment][0]
        wide = store.load_funding([symbol], start=None if load_start is None else load_start - pd.Timedelta(days=2), end=load_end)
        funding = wide[symbol] if symbol in wide.columns else None
        flags = structure_flags(grid, feats, funding, names)
        n = len(grid)
        tradable = grid.exists & ~grid.outage & (grid.raw_qv > 0)
        price = np.where(tradable, grid.raw_open, np.nan)
        cum = np.concatenate([[0.0], np.nancumsum(price)])
        cnt = np.concatenate([[0], np.cumsum(np.isfinite(price))])
        if all_days:
            days = pd.date_range(first_day + pd.Timedelta(days=2), end_day - pd.Timedelta(days=2), freq="1D")
            todo = [(day, kind) for day in days for kind in ("increase", "decrease")]
        else:
            mine = trades[trades.symbol == symbol]
            todo = list(zip(mine.fill, mine.kind))
        index = grid.index
        firsts = {name: (next_true(flags[name][0]), next_true(flags[name][1])) for name in names}
        for fill, kind in todo:
            p0 = int(index.searchsorted(fill))
            if p0 >= n or index[p0] != fill or not np.isfinite(price[p0]):
                continue
            sign = 1.0 if kind == "increase" else -1.0
            for name in names:
                first = firsts[name][0 if kind == "increase" else 1]
                for window in WINDOWS:
                    end = p0 + window
                    if end >= n or not np.isfinite(price[end]):
                        continue
                    event = int(first[p0])
                    triggered = event <= end - 1
                    at = event + 1 if triggered else end
                    if not np.isfinite(price[at]):
                        continue
                    twap = (cum[end + 1] - cum[p0 + 1]) / max(cnt[end + 1] - cnt[p0 + 1], 1)
                    rows.append({
                        "market": market, "symbol": symbol, "fill": fill, "kind": kind, "structure": name, "window": window,
                        "triggered": triggered, "waited_min": at - p0,
                        "p_open": price[p0], "p_wait": price[at], "p_delay": price[end], "p_twap": twap,
                        "imp_wait_bp": sign * (price[p0] - price[at]) / price[p0] * 1e4,
                        "imp_delay_bp": sign * (price[p0] - price[end]) / price[p0] * 1e4,
                        "imp_twap_bp": sign * (price[p0] - twap) / price[p0] * 1e4,
                    })
    return pd.DataFrame(rows)


def day_t(values: pd.Series, days: pd.Series) -> dict:
    ok = values.notna()
    values, days = values[ok], days[ok]
    if len(values) < 3:
        return {"n": int(len(values)), "days": 0, "mean": float("nan"), "t": float("nan")}
    codes, _ = pd.factorize(days)
    n_day = np.bincount(codes).astype(float)
    s_day = np.bincount(codes, weights=values.to_numpy())
    # factorised days are not consecutive calendar days: plain day clustering (no Bartlett terms; windows of different days do not overlap)
    stat = es.cluster_mean(n_day, s_day, 0)
    return {"n": stat["n"], "days": stat["days"], "mean": stat["mean"], "t": stat["t"]}


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, name, window), part in table.groupby(["kind", "structure", "window"], sort=False):
        wait = day_t(part.imp_wait_bp, part.fill)
        delay = day_t(part.imp_delay_bp, part.fill)
        twap = day_t(part.imp_twap_bp, part.fill)
        versus_delay = day_t(part.imp_wait_bp - part.imp_delay_bp, part.fill)
        versus_twap = day_t(part.imp_wait_bp - part.imp_twap_bp, part.fill)
        rows.append({
            "side": "entry (buy)" if kind == "increase" else "exit (sell)",
            "structure": name,
            "window_min": window,
            "trades": wait["n"],
            "days": wait["days"],
            "triggered": part.triggered.mean(),
            "median_wait_min": float(part.waited_min.median()),
            "mean_bp": wait["mean"],
            "median_bp": float(part.imp_wait_bp.median()),
            "t_day": wait["t"],
            "worse": float((part.imp_wait_bp < 0).mean()),
            "sd_bp": float(part.imp_wait_bp.std()),
            "delay_mean_bp": delay["mean"],
            "delay_t": delay["t"],
            "twap_mean_bp": twap["mean"],
            "twap_t": twap["t"],
            "vs_delay_bp": versus_delay["mean"],
            "vs_delay_t": versus_delay["t"],
            "vs_twap_bp": versus_twap["mean"],
            "vs_twap_t": versus_twap["t"],
        })
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    segment = argv[0]
    markets = argv[1:] or ["spot", "um"]
    selection = HERE / "SELECTION.json"
    if not selection.exists():
        raise SystemExit("write SELECTION.json first: Part B is run for the selected structures")
    chosen = json.loads(selection.read_text())["definition_names_to_evaluate"]
    names = [SELECTED_TO_STRUCTURE[name] for name in chosen]
    trades = rule_changes(segment)
    folder = runner.OUT / "part_b"
    folder.mkdir(parents=True, exist_ok=True)
    trades.to_csv(folder / f"rule_changes_{segment}.csv", index=False)
    print(segment, "rule changes:", trades.groupby(["symbol", "kind"]).size().to_dict())
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    for market in markets:
        for label, all_days in (("rule_days", False), ("all_days", True)):
            table = timing_table(market, segment, names, trades, all_days)
            table.to_parquet(folder / f"timing_{segment}_{market}_{label}.parquet")
            summary = summarise(table)
            summary.to_csv(folder / f"summary_{segment}_{market}_{label}.csv", index=False)
            print(f"\n=== {segment} / {market} / {label} ===")
            print(summary.round(2).to_string(index=False))
            if not all_days and market == "spot" and len(table):
                both = table.merge(trades, on=["symbol", "fill", "kind"])
                print("max |1m open at 00:00 / daily open - 1|:", float((both.p_open / both.daily_open - 1).abs().max()))


if __name__ == "__main__":
    main(sys.argv[1:])
