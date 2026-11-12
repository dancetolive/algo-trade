"""Audit step 5b: the selected definitions on markets the researcher did not use for the selection.

    python -I a10_unused_symbols.py

* spot SOL, BNB, XRP, DOGE (the study only ran BTC and ETH spot as a cross-check): design and validation;
* spot BTC + ETH, to compare with the study's own cross-check figures;
* the PAXG perpetual (listed 2025-03-27, three months inside the validation window, never used by the study).
Spot prices follow the same crashes as the perpetuals, so this is a test of "not an artefact of one data file
or one venue", not independent evidence. Spot taker round trip: 22 bp (BTC, ETH) or 24 bp.
Writes out/unused_symbols.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auditlib as al  # noqa: E402

OUT = HERE / "out"


def stat(e: pd.DataFrame, h: int, cost_fn) -> dict:
    x = e[f"f{h}"].to_numpy() * e["side"].to_numpy()
    ok = np.isfinite(x)
    e, x = e[ok], x[ok]
    if len(x) < 2:
        return {"events": int(len(x))}
    day = (e["time"].dt.as_unit("ns").astype("int64") // al.DAY_NS).to_numpy()
    cost = e["symbol"].map(cost_fn).to_numpy() / 1e4
    base = e[f"b{h}"].to_numpy() * e["side"].to_numpy()
    g, ex, net = al.cluster_stat(x, day), al.cluster_stat(x - base, day), al.cluster_stat(x - cost, day)
    lag2 = e[f"lag2_{h}"].to_numpy() * e["side"].to_numpy()
    return {"events": int(len(x)), "days": g["clusters"], "gross_bp": g["mean"] * 1e4, "t_gross": g["t"], "excess_bp": ex["mean"] * 1e4, "t_day": ex["t"],
            "net_bp": net["mean"] * 1e4, "t_net": net["t"], "hit": float((x > 0).mean()), "median_bp": float(np.median(x)) * 1e4,
            "one_minute_late_bp": float(np.nanmean(lag2)) * 1e4}


def main():
    res = {}
    for segment in ("design", "validation"):
        spot = pd.read_parquet(OUT / f"events_{segment}_spot.parquet")
        for name, h in (("move5_abs2pct_dn", 15), ("move1_z3_dn", 15)):
            e = spot[spot.name == name]
            groups = {"spot SOL+BNB+XRP+DOGE (never run by the study)": e[e.symbol.isin(["SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"])],
                      "spot BTC+ETH (the study's cross-check)": e[e.symbol.isin(["BTCUSDT", "ETHUSDT"])]}
            for sym in ("SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"):
                groups[f"spot {sym[:-4]}"] = e[e.symbol == sym]
            print(f"\n=== {segment}: {name} @ {h} on spot (spot taker costs) ===")
            for label, part in groups.items():
                s = stat(part, h, al.spot_round_trip_bp)
                res[f"{segment}/{name}/{label}"] = s
                if s["events"] < 2:
                    print(f"  {label:48s} events {s['events']}")
                    continue
                print(f"  {label:48s} events {s['events']:6d} days {s['days']:4d} gross {s['gross_bp']:7.2f} (t {s['t_gross']:5.2f}) excess {s['excess_bp']:7.2f} (t {s['t_day']:5.2f}) "
                      f"net spot taker {s['net_bp']:7.2f} (t {s['t_net']:5.2f}) hit {s['hit']:.2f} median {s['median_bp']:6.1f} one minute late {s['one_minute_late_bp']:6.2f}")
    paxg = pd.read_parquet(OUT / "events_validation_um_PAXG.parquet")
    print("\n=== validation: PAXG perpetual (2025-03-27 .. 2025-06-30) ===")
    print("  events by definition:", paxg.groupby("name").size().to_dict(), "(no fall of 2% in 5 minutes occurred)")
    s = stat(paxg[paxg.name == "move1_z3_dn"], 15, al.taker_round_trip_bp)
    res["validation/move1_z3_dn/um PAXG"] = s
    print(f"  move1_z3_dn @ 15: events {s['events']}, days {s['days']}, gross {s['gross_bp']:.2f} bp (t {s['t_gross']:.2f}), net taker {s['net_bp']:.2f} (t {s['t_net']:.2f}), hit {s['hit']:.2f}")
    res["validation/move5_abs2pct_dn/um PAXG"] = {"events": int((paxg.name == "move5_abs2pct_dn").sum())}
    (OUT / "unused_symbols.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
