"""Audit check 5b: the selected definitions on symbols / venues the researcher did not use for selection.

    python -I a07_other_symbols.py

* spot BTC + ETH: the researcher's own cross-check (reproduced here from my event tables);
* spot SOL, BNB, XRP, DOGE: 1-minute sets the researcher never loaded (same coins as the perpetuals, other venue);
* spot PAXG (tokenised gold): researcher looked at design only; validation shown here;
* PAXG perpetual (2025-03-27 .. 2025-06-30): never loaded by the researcher; three months only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al  # noqa: E402

PAIRS = [("sweep_N240_x10_k1", 240), ("sweep_N240_x10_k1", 60), ("sweep_N30_x2_k3", 5)]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
THEIRS = os.path.join(os.path.dirname(al.HERE), "price_action", "out")


def stats(ev: pd.DataFrame, h: int, market: str) -> dict:
    p = ev[np.isfinite(ev[f"g{h}"]) & np.isfinite(ev[f"b{h}"])]
    if len(p) == 0:
        return {"events": 0}
    day = ((p["ts"] - EPOCH) // pd.Timedelta(days=1)).to_numpy()
    adj = h >= 240
    g = al.cluster_stats(p[f"g{h}"].to_numpy(), day, adjacent=adj)
    e = al.cluster_stats((p[f"g{h}"] - p[f"b{h}"]).to_numpy(), day, adjacent=adj)
    w = al.cluster_stats((p[f"g{h}"] - p[f"b{h}"]).to_numpy(), day // 7)
    cost = float(np.mean([al.taker_round_trip_bp(s, market) for s in p["symbol"]]))
    long_ = p[p["dir"] > 0]
    short = p[p["dir"] < 0]
    return dict(events=g["n"], days=g["clusters"], gross=g["mean"], base=float(p[f"b{h}"].mean()), excess=e["mean"], t_excess=e["t"],
                t_excess_week=w["t"], t_gross=g["t"], cost_taker=cost, net_taker=g["mean"] - cost,
                long_gross=float(long_[f"g{h}"].mean()) if len(long_) else np.nan, long_excess=float((long_[f"g{h}"] - long_[f"b{h}"]).mean()) if len(long_) else np.nan,
                short_gross=float(short[f"g{h}"].mean()) if len(short) else np.nan,
                short_excess=float((short[f"g{h}"] - short[f"b{h}"]).mean()) if len(short) else np.nan)


def main() -> None:
    rows = []
    sets = [
        ("um six coins (selection sample)", "um", "", lambda e: e),
        ("spot BTC+ETH (researcher's cross-check)", "spot", "", lambda e: e[e["symbol"].isin(["BTCUSDT", "ETHUSDT"])]),
        ("spot SOL+BNB+XRP+DOGE (unused by researcher)", "spot", "", lambda e: e[~e["symbol"].isin(["BTCUSDT", "ETHUSDT"])]),
        ("spot six coins", "spot", "", lambda e: e),
        ("spot PAXG", "spot", "_paxg", lambda e: e),
        ("perp PAXG (2025-03-27..)", "um", "_paxgperp", lambda e: e),
    ]
    for label, market, tag, pick in sets:
        for segment in ("design", "validation"):
            path = os.path.join(al.OUT, f"events_{segment}_{market}{tag}.parquet")
            if not os.path.exists(path):
                continue
            ev = pick(pd.read_parquet(path))
            for def_id, h in PAIRS:
                r = stats(ev[ev["def_id"] == def_id], h, market)
                r.update(sample=label, segment=segment, def_id=def_id, h=h)
                rows.append(r)
    table = pd.DataFrame(rows)
    # per coin on spot, selected pair only
    per = []
    for segment in ("design", "validation"):
        ev = pd.read_parquet(os.path.join(al.OUT, f"events_{segment}_spot.parquet"))
        for sym, chunk in ev[ev["def_id"] == "sweep_N240_x10_k1"].groupby("symbol"):
            for h in (240, 60):
                r = stats(chunk, h, "spot")
                r.update(sample=f"spot {sym}", segment=segment, def_id="sweep_N240_x10_k1", h=h)
                per.append(r)
    per = pd.DataFrame(per)
    full = pd.concat([table, per], ignore_index=True)
    full.to_csv(os.path.join(al.OUT, "other_symbols.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    cols = ["sample", "segment", "def_id", "h", "events", "days", "gross", "base", "excess", "t_excess", "t_excess_week", "cost_taker", "net_taker",
            "long_gross", "long_excess", "short_gross", "short_excess"]
    print(full[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    # researcher's spot table for comparison
    for segment in ("design", "validation"):
        ref = pd.read_csv(os.path.join(THEIRS, f"table_{segment}_spot.csv")).set_index(["def_id", "h"])
        for def_id, h in PAIRS:
            r = ref.loc[(def_id, h)]
            mine = table[(table["sample"].str.startswith("spot BTC+ETH")) & (table["segment"] == segment) & (table["def_id"] == def_id) & (table["h"] == h)].iloc[0]
            print(f"researcher spot BTC+ETH {segment} {def_id}@{h}: events {int(r['events'])} gross {r['gross']:.4f} t {r['t_excess']:.4f} | audit: "
                  f"events {int(mine['events'])} gross {mine['gross']:.4f} t {mine['t_excess']:.4f}")


if __name__ == "__main__":
    main()
