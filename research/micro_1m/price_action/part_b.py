"""Part B: does waiting for a 1-minute structure improve the fill of a daily trend trade?

    python -I -B part_b.py design
    python -I -B part_b.py validation      # only after selection.json exists; protocol is read from it

The trend rule is blend_weights() from research/baseline/trend_baseline.py
(mean of the 50 / 100 / 200-day moving-average filters on BTC and ETH spot
daily closes). A change decided at the close of day D is executed at 00:00 UTC
of day D+1. For every such trade the price at the 00:00 open is compared with
the price obtained by waiting, for up to 4 hours and for up to 24 hours, for
the first occurrence of a structure in the direction of the trade (bullish for
a buy, bearish for a sell): fill at the open of the minute after the event, or
at the open of the bar at 00:00 + window if the structure never shows up.

improvement (bp) = side * (P_open - P_wait) / P_open * 1e4, positive = better price.

Controls
--------
delay_end : wait until the end of the window regardless of structures.
placebo   : the same distribution of waiting times, applied without looking at
            the structure (each trade is filled after the waiting times of all
            the OTHER trades and the results are averaged). "info" = improvement
            - placebo is what the structure itself adds over simply waiting.
oracle    : the best price available in the window (perfect hindsight), for scale.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import eventstudy as es  # noqa: E402
import structures as st  # noqa: E402
from run_segment import OUT, load_segment_bars, segment_window  # noqa: E402

LOOKBACK = 1500


def trend_rule():
    spec = importlib.util.spec_from_file_location("trend_baseline", os.path.join(REPO, "research", "baseline", "trend_baseline.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trend_trades(segment: str) -> pd.DataFrame:
    """One row per (coin, day) on which the blended target weight changes. ``minute`` is the fill time (00:00 UTC of D+1)."""
    from algotrade.data import store

    rule = trend_rule()
    panel = store.load_panel("spot", "1d", rule.SYMBOLS, end=segment_window(segment)[1])
    weights = rule.blend_weights(panel)
    change = weights.diff()
    rows = []
    for symbol in rule.SYMBOLS:
        for day, delta in change[symbol].dropna().items():
            if abs(delta) > 1e-12:
                fill = day + pd.Timedelta(days=1)
                rows.append({"symbol": symbol, "minute": int(fill.timestamp() // 60), "side": 1 if delta > 0 else -1,
                             "size": abs(float(delta)), "weight_after": float(weights.loc[day, symbol])})
    return pd.DataFrame(rows)


def all_days(bars: es.Bars) -> np.ndarray:
    """Every UTC midnight inside the loaded bars."""
    first = bars.minute[0] + (-bars.minute[0]) % es.MINUTES_PER_DAY
    return np.arange(first, bars.minute[-1] + 1, es.MINUTES_PER_DAY, dtype=np.int64)


def next_index(mask: np.ndarray) -> np.ndarray:
    """Smallest u >= t with mask[u]; len(mask) if none."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def evaluate(bars: es.Bars, signals: dict[str, np.ndarray], minute: np.ndarray, side: np.ndarray, windows) -> pd.DataFrame:
    """Per trade and (structure, window): fill delay, improvement and controls. Trades touching a data gap are dropped."""
    n, o = bars.n, bars.o
    miss = np.concatenate([[0], np.cumsum(~bars.valid)])
    i0 = minute - bars.minute[0]
    frames = []
    for window in windows:
        ok = (i0 - LOOKBACK >= 0) & (i0 + window < n)
        j0 = np.where(ok, i0, LOOKBACK)
        ok &= (miss[np.minimum(j0 + window + 1, n)] - miss[j0 - LOOKBACK]) == 0
        t0, sd = i0[ok], side[ok].astype(float)
        p0 = o[t0]
        end_px = o[t0 + window]
        # perfect hindsight: best price in the window (lows for a buyer, highs for a seller), including the open itself
        lo = pd.Series(bars.l).rolling(window, min_periods=window).min().to_numpy()
        hi = pd.Series(bars.h).rolling(window, min_periods=window).max().to_numpy()
        best = np.where(sd > 0, lo[t0 + window - 1], hi[t0 + window - 1])
        oracle = sd * (p0 - best) / p0 * 1e4
        delay_end = sd * (p0 - end_px) / p0 * 1e4
        for name, sig in signals.items():
            nxt_up, nxt_dn = next_index(sig > 0), next_index(sig < 0)
            first = np.where(sd > 0, nxt_up[t0], nxt_dn[t0])
            triggered = first <= t0 + window - 1
            fill = np.where(triggered, first + 1, t0 + window)
            delay = fill - t0
            improvement = sd * (p0 - o[fill]) / p0 * 1e4
            # placebo: every trade filled after the waiting times of all the other trades
            grid, counts = np.unique(delay, return_counts=True)
            counts = counts.astype(float)
            at_delay = sd[:, None] * (p0[:, None] - o[t0[:, None] + grid[None, :]]) / p0[:, None] * 1e4
            total = at_delay @ counts
            own = at_delay[np.arange(len(t0)), np.searchsorted(grid, delay)]
            placebo = (total - own) / max(len(t0) - 1, 1)
            frames.append(pd.DataFrame({
                "structure": name, "window": window, "minute": minute[ok], "side": side[ok], "triggered": triggered,
                "delay": delay, "improvement": improvement, "delay_end": delay_end, "placebo": placebo,
                "info": improvement - placebo, "oracle": oracle,
            }))
    return pd.concat(frames, ignore_index=True)


def summarise(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per (set, structure, window): the numbers asked for in the brief, clustered by fill day."""
    rows = []
    for (label, name, window), part in trades.groupby(["set", "structure", "window"], sort=False):
        day = part["minute"].to_numpy() // es.MINUTES_PER_DAY
        lag = 1 if window >= es.MINUTES_PER_DAY else 0

        def clustered(column):
            by = part.groupby(day)[column].agg(["sum", "count"])
            return es.cluster_mean_t(by["sum"].to_numpy(), by["count"].to_numpy(), by.index.to_numpy(), lag)

        mean, se, t = clustered("improvement")
        _, _, t_end = clustered("delay_end")
        info, _, t_info = clustered("info")
        imp = part["improvement"]
        rows.append({
            "set": label, "structure": name, "window": window, "trades": len(part), "days": int(len(np.unique(day))),
            "triggered": part["triggered"].mean(), "mean_wait_min": part["delay"].mean(), "median_wait_min": part["delay"].median(),
            "mean_bp": mean, "median_bp": imp.median(), "se_bp": se, "t": t,
            "worse": (imp < 0).mean(), "better": (imp > 0).mean(),
            "mean_if_better": imp[imp > 0].mean(), "mean_if_worse": imp[imp < 0].mean(),
            "delay_end_bp": part["delay_end"].mean(), "t_delay_end": t_end,
            "placebo_bp": part["placebo"].mean(), "info_bp": info, "t_info": t_info,
            "oracle_bp": part["oracle"].mean(), "oracle_median_bp": part["oracle"].median(),
        })
    return pd.DataFrame(rows)


def with_pooled_sets(trades: pd.DataFrame) -> pd.DataFrame:
    """Add the pooled sets: all trend trades (entries and exits together) and all days (buys and sells together)."""
    both = trades[trades["set"].isin(["trend_entry", "trend_exit"])].assign(set="trend_both")
    every = trades[trades["set"].isin(["all_days_buy", "all_days_sell"])].assign(set="all_days_both")
    return pd.concat([trades, both, every], ignore_index=True)


def main() -> None:
    segment = sys.argv[1]
    assert segment in ("design", "validation", "dryrun")
    with open(os.path.join(HERE, "selection.json")) as handle:
        protocol = json.load(handle)["protocol"]["part_b"]
    structures = protocol["structures_selected"] + protocol["structures_companion"]
    windows = protocol["windows_minutes"]
    rule_trades = trend_trades(segment)
    pieces = []
    first_minute = None
    for market in ("spot", "um"):
        for symbol in sorted(rule_trades["symbol"].unique()):
            bars, first_minute = load_segment_bars(segment, market, symbol)
            events = st.detect(bars, structures)
            signals = {name: st.signal_array(events[name], bars.n) for name in structures}
            mine = rule_trades[rule_trades["symbol"] == symbol]
            sets = {
                "trend_entry": (mine.loc[mine["side"] > 0, "minute"].to_numpy(), 1),
                "trend_exit": (mine.loc[mine["side"] < 0, "minute"].to_numpy(), -1),
                "all_days_buy": (all_days(bars), 1),
                "all_days_sell": (all_days(bars), -1),
            }
            for label, (minute, side) in sets.items():
                if first_minute is not None:
                    minute = minute[minute >= first_minute]
                minute = minute[(minute >= bars.minute[0]) & (minute <= bars.minute[-1])]
                if len(minute) == 0:
                    continue
                part = evaluate(bars, signals, minute, np.full(len(minute), side), windows)
                part.insert(0, "set", label)
                part.insert(0, "symbol", symbol)
                part.insert(0, "market", market)
                pieces.append(part)
    trades = pd.concat(pieces, ignore_index=True)
    trades.insert(0, "segment", segment)
    trades.to_parquet(os.path.join(OUT, f"part_b_trades_{segment}.parquet"), index=False)
    tables = []
    for market, part in trades.groupby("market", sort=False):
        table = summarise(with_pooled_sets(part))
        table.insert(0, "market", market)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table.insert(0, "segment", segment)
    table.to_csv(os.path.join(OUT, f"part_b_{segment}.csv"), index=False)
    counts = rule_trades.assign(year=pd.to_datetime(rule_trades["minute"] * 60, unit="s", utc=True).dt.year)
    counts = counts[counts["year"] >= 2020]  # 1-minute data start in 2020
    if first_minute is not None:
        counts = counts[counts["minute"] >= first_minute]
    counts.groupby(["year", "side"]).size().unstack(fill_value=0).to_csv(os.path.join(OUT, f"part_b_trade_counts_{segment}.csv"))
    print("trend-rule trades by year and side (both coins):")
    print(counts.groupby(["year", "side"]).size().unstack(fill_value=0).to_string())
    print("mean trade size (share of equity):", round(counts["size"].mean(), 4))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    cols = ["market", "set", "structure", "window", "trades", "days", "triggered", "mean_wait_min", "mean_bp", "median_bp", "t",
            "worse", "mean_if_better", "mean_if_worse", "delay_end_bp", "t_delay_end", "placebo_bp", "info_bp", "t_info", "oracle_median_bp"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
