"""Audit step 2 (check 3): reproduce the headline numbers of every selected pair from my own event tables
and put them next to the researcher's tables (read as CSV only; none of their functions is imported).

    python -I a02_reproduce.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al  # noqa: E402

THEIRS = os.path.join(os.path.dirname(al.HERE), "price_action", "out")
PAIRS = [("sweep_N240_x10_k1", 240), ("sweep_N240_x10_k1", 60), ("sweep_N30_x2_k3", 5)]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def day_of(ts: pd.Series) -> np.ndarray:
    return ((ts - EPOCH) // pd.Timedelta(days=1)).to_numpy()


def headline(ev: pd.DataFrame, h: int, market: str = "um") -> dict:
    part = ev[np.isfinite(ev[f"g{h}"]) & np.isfinite(ev[f"b{h}"])]
    day = day_of(part["ts"])
    g, b = part[f"g{h}"].to_numpy(), part[f"b{h}"].to_numpy()
    adj = h >= 240
    sg = al.cluster_stats(g, day, adjacent=adj)
    se = al.cluster_stats(g - b, day, adjacent=adj)
    cost = float(np.mean([al.taker_round_trip_bp(s, market) for s in part["symbol"]]))
    return dict(events=sg["n"], days=sg["clusters"], gross=sg["mean"], base=float(b.mean()), excess=se["mean"],
                t_excess=se["t"], t_gross=sg["t"], t_naive=se["t_naive"], hit=float((g > 0).mean()), cost_taker=cost,
                net_taker=sg["mean"] - cost, t_net_taker=(sg["mean"] - cost) / sg["se"], net_maker_bound=sg["mean"] - 4.0)


def main() -> None:
    rows = []
    for segment in ("design", "validation"):
        ev = pd.read_parquet(os.path.join(al.OUT, f"events_{segment}_um.parquet"))
        theirs = pd.read_csv(os.path.join(THEIRS, f"table_{segment}.csv")).set_index(["def_id", "h"])
        for def_id, h in PAIRS:
            mine = headline(ev[ev["def_id"] == def_id], h)
            ref = theirs.loc[(def_id, h)]
            for k in ("events", "days", "gross", "base", "excess", "t_excess", "t_gross", "t_naive", "hit", "cost_taker", "net_taker",
                      "net_maker_bound"):
                rows.append(dict(segment=segment, def_id=def_id, h=h, stat=k, audit=mine[k], researcher=float(ref[k]),
                                 diff=mine[k] - float(ref[k])))
            rows.append(dict(segment=segment, def_id=def_id, h=h, stat="t_net_taker", audit=mine["t_net_taker"], researcher=np.nan, diff=np.nan))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(al.OUT, "reproduce_headline.csv"), index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    wide = table.pivot_table(index=["segment", "def_id", "h"], columns="stat", values=["audit", "researcher"], sort=False)
    for (segment, def_id, h), part in table.groupby(["segment", "def_id", "h"], sort=False):
        print(f"\n{segment}  {def_id} @ {h}")
        print(part[["stat", "audit", "researcher", "diff"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    worst = table.dropna(subset=["diff"]).assign(a=lambda d: d["diff"].abs()).sort_values("a", ascending=False).head(8)
    print("\nlargest absolute differences:")
    print(worst[["segment", "def_id", "h", "stat", "audit", "researcher", "diff"]].to_string(index=False))

    # by coin, year, side: compare with their breakdown tables
    out = []
    for segment in ("design", "validation"):
        ev = pd.read_parquet(os.path.join(al.OUT, f"events_{segment}_um.parquet"))
        ev["year"] = ev["ts"].dt.year
        bd = pd.read_csv(os.path.join(THEIRS, f"breakdown_{segment}.csv"))
        for def_id, h in PAIRS:
            sub = ev[ev["def_id"] == def_id]
            for label, col in (("year", "year"), ("symbol", "symbol"), ("direction", "dir")):
                for value, chunk in sub.groupby(col):
                    mine = headline(chunk, h)
                    name = str(value) if label != "direction" else ("long" if value > 0 else "short")
                    ref = bd[(bd["def_id"] == def_id) & (bd["h"] == h) & (bd["by"] == label) & (bd["value"].astype(str) == name)]
                    r = ref.iloc[0] if len(ref) else None
                    out.append(dict(segment=segment, def_id=def_id, h=h, by=label, value=name, events=mine["events"], gross=mine["gross"],
                                    excess=mine["excess"], t_excess=mine["t_excess"], net_taker=mine["net_taker"],
                                    r_events=None if r is None else int(r["events"]), r_gross=None if r is None else r["gross"],
                                    r_excess=None if r is None else r["excess"], r_t_excess=None if r is None else r["t_excess"]))
    bdt = pd.DataFrame(out)
    bdt.to_csv(os.path.join(al.OUT, "reproduce_breakdown.csv"), index=False)
    bdt["d_gross"] = bdt["gross"] - bdt["r_gross"]
    bdt["d_t"] = bdt["t_excess"] - bdt["r_t_excess"]
    print("\nbreakdown rows:", len(bdt), " max |gross diff|:", bdt["d_gross"].abs().max(), " max |t diff|:", bdt["d_t"].abs().max(),
          " events mismatches:", int((bdt["events"] != bdt["r_events"]).sum()))
    print(bdt.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
