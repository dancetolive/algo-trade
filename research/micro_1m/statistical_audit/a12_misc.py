"""Audit, small checks:
   A. bar labelling: 1-minute bars aggregated by clock hour must equal the hourly bars (open of :00, close of :59,
      max high, min low, summed volume) if and only if both files are indexed by OPEN time;
   B. the headline pair's best / busiest / worst days and the drawdown of the per-unit P&L (design, validation).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auditlib as al
from algotrade.data import store

OUT = HERE / "out"
res = {}
print("A. 1-minute bars aggregated to hours vs the hourly file (design + validation, through the loader)")
for market, symbol in (("um", "BTCUSDT"), ("um", "DOGEUSDT"), ("um", "SOLUSDT"), ("spot", "ETHUSDT")):
    m = store.load_klines(market, "1m", symbol, columns=["open", "high", "low", "close", "quote_volume"])
    h = store.load_klines(market, "1h", symbol, columns=["open", "high", "low", "close", "quote_volume"])
    g = m.groupby(m.index.floor("h")).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), quote_volume=("quote_volume", "sum"), n=("open", "size"))
    full = g[g.n == 60]
    both = full.join(h, how="inner", rsuffix="_h")
    d = {c: float((both[c] / both[c + "_h"] - 1).abs().max()) for c in ("open", "high", "low", "close")}
    exact = {c: float((both[c] == both[c + "_h"]).mean()) for c in ("open", "high", "low", "close")}
    # the same comparison if the minute bars were (wrongly) taken as labelled by their close time: shift labels by one minute
    m2 = m.copy(); m2.index = m2.index - pd.Timedelta(minutes=1)
    g2 = m2.groupby(m2.index.floor("h")).agg(open=("open", "first"), close=("close", "last"), n=("open", "size"))
    b2 = g2[g2.n == 60].join(h, how="inner", rsuffix="_h")
    wrong = float((b2["open"] == b2["open_h"]).mean())
    res[f"labelling/{market}/{symbol}"] = {"hours_compared": int(len(both)), "share_exactly_equal": exact, "max_rel_diff": d, "share_equal_open_if_minute_labels_were_close_times": wrong}
    print(f"  {market} {symbol}: {len(both)} complete hours; exactly equal: {exact}; max rel diff {d}; open equal if labels were close times: {wrong:.3f}")

print("\nB. days of the headline pair")
for segment in ("design", "validation"):
    e = pd.read_parquet(OUT / f"events_{segment}_um.parquet")
    e = e[(e.name == "move5_abs2pct_dn") & np.isfinite(e.f15)].copy()
    e["time"] = e["time"].dt.as_unit("ns")
    e["date"] = e["time"].dt.floor("D").dt.strftime("%Y-%m-%d")
    e["net"] = e.f15 - e.symbol.map(al.taker_round_trip_bp) / 1e4
    by = e.groupby("date").agg(events=("net", "size"), coins=("symbol", "nunique"), gross_mean_bp=("f15", lambda s: s.mean() * 1e4), net_sum_pct=("net", lambda s: s.sum() * 100))
    total = by.net_sum_pct.sum()
    cum = by.net_sum_pct.cumsum()
    dd = float((cum - cum.cummax()).min())
    print(f"\n  {segment}: {len(e)} events on {len(by)} days; sum of net = {total:.0f}% of one unit; worst day {by.net_sum_pct.min():.0f}% of one unit; max drawdown of the cumulative sum {dd:.0f}% of one unit")
    print("  busiest days:\n", by.sort_values("events", ascending=False).head(8).round(1).to_string())
    print("  best days:\n", by.sort_values("net_sum_pct", ascending=False).head(6).round(1).to_string())
    print("  worst days:\n", by.sort_values("net_sum_pct").head(6).round(1).to_string())
    res[f"days/{segment}"] = {"total_net_pct_of_unit": float(total), "worst_day_pct_of_unit": float(by.net_sum_pct.min()), "max_drawdown_pct_of_unit": dd,
                              "busiest": by.sort_values("events", ascending=False).head(8).round(2).reset_index().to_dict(orient="records"),
                              "worst": by.sort_values("net_sum_pct").head(6).round(2).reset_index().to_dict(orient="records")}
(OUT / "misc.json").write_text(json.dumps(res, indent=1, default=float))
