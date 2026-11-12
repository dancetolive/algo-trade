"""Audit step 3 (+2, 6, 7): headline numbers of the three selected pairs from the auditor's own event tables.

    python -I a03_headline.py

Prints, per segment and pair: events, days, gross, matched baseline, excess, day-clustered t (own formula),
net of taker costs, by year, by coin, later fills; then an event-by-event comparison with the researcher's
event-level tables (same bars? same forward return?), which is the concrete test of the fill timing.
Writes out/headline.json.
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
THEIRS = HERE.parent / "statistical" / "out"
PAIRS = [("move5_abs2pct_dn", 15), ("move1_z3_dn", 15), ("fund_post5_big", 5)]
REPORTED = {  # (segment, name): events, days, gross, baseline, t_day, net taker   -- from the researcher's results table
    ("design", "move5_abs2pct_dn"): (7965, 697, 40.42, 1.16, 9.9, 26.7),
    ("design", "move1_z3_dn"): (84676, 1460, 7.87, 0.64, 7.4, -5.4),
    ("design", "fund_post5_big"): (4270, 766, 8.63, 0.21, 5.0, -4.8),
    ("validation", "move5_abs2pct_dn"): (1004, 157, 47.12, 0.12, 5.0, 33.4),
    ("validation", "move1_z3_dn"): (35439, 547, 3.68, 0.01, 3.2, -9.6),
    ("validation", "fund_post5_big"): (700, 149, 2.18, 0.07, 0.9, -11.4),
}


def stats(e: pd.DataFrame, h: int, col: str | None = None) -> dict:
    col = col or f"f{h}"
    x = (e[col] * e["side"]).to_numpy()
    base = (e[f"b{h}"] * e["side"]).to_numpy()
    ok = np.isfinite(x)
    x, base, sub = x[ok], base[ok], e[ok]
    day = (sub["time"].dt.floor("D").astype("int64") // al.DAY_NS).to_numpy() if str(sub["time"].dtype).endswith("ns, UTC]") else \
        (sub["time"].dt.floor("D").dt.as_unit("ns").astype("int64") // al.DAY_NS).to_numpy()
    cost = sub["symbol"].map(al.taker_round_trip_bp).to_numpy() / 1e4
    spot_cost = sub["symbol"].map(al.spot_round_trip_bp).to_numpy() / 1e4
    g = al.cluster_stat(x, day)
    ex = al.cluster_stat(x - base, day)
    net = al.cluster_stat(x - cost, day)
    return {"events": int(len(x)), "days": int(g["clusters"]), "gross_bp": g["mean"] * 1e4, "t_gross": g["t"], "base_bp": float(base.mean()) * 1e4,
            "excess_bp": ex["mean"] * 1e4, "t_day": ex["t"], "t_naive": al.naive_t(x - base), "hit": float((x > 0).mean()),
            "median_bp": float(np.median(x)) * 1e4, "cost_bp": float(cost.mean()) * 1e4, "net_taker_bp": net["mean"] * 1e4, "t_net": net["t"],
            "net_maker_bound_bp": g["mean"] * 1e4 - al.MAKER_ROUND_TRIP_BP, "net_spot_taker_bp": float((x - spot_cost).mean()) * 1e4,
            "sd_bp": float(x.std(ddof=1)) * 1e4}


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    out = {}
    for segment in ("design", "validation"):
        events = pd.read_parquet(OUT / f"events_{segment}_um.parquet")
        events["time"] = events["time"].dt.as_unit("ns")
        for name, h in PAIRS:
            e = events[events.name == name]
            s = stats(e, h)
            rep = REPORTED[(segment, name)]
            print(f"\n=== {segment}: {name} @ {h} min (trade side) ===")
            print(f"  own     : events {s['events']}, days {s['days']}, gross {s['gross_bp']:.2f} bp, baseline {s['base_bp']:.2f}, excess {s['excess_bp']:.2f}, "
                  f"t_day {s['t_day']:.2f} (gross t {s['t_gross']:.2f}, naive {s['t_naive']:.1f}), hit {s['hit']:.3f}, median {s['median_bp']:.1f}, sd {s['sd_bp']:.0f}, "
                  f"cost {s['cost_bp']:.2f}, net taker {s['net_taker_bp']:.2f} (t {s['t_net']:.2f}), maker bound {s['net_maker_bound_bp']:.2f}, spot taker {s['net_spot_taker_bp']:.2f}")
            print(f"  reported: events {rep[0]}, days {rep[1]}, gross {rep[2]:.2f} bp, baseline {rep[3]:.2f}, t_day {rep[4]:.1f}, net taker {rep[5]:.1f}")
            entry = {"own": s, "reported": dict(zip(("events", "days", "gross_bp", "base_bp", "t_day", "net_taker_bp"), rep))}
            # by year / by coin
            years = {}
            for year, part in e.groupby(e["time"].dt.year):
                r = stats(part, h)
                years[int(year)] = {k: r[k] for k in ("events", "days", "gross_bp", "t_gross", "excess_bp", "t_day", "net_taker_bp", "hit")}
            coins = {}
            for sym, part in e.groupby("symbol", sort=False):
                r = stats(part, h)
                coins[sym] = {k: r[k] for k in ("events", "days", "gross_bp", "t_gross", "excess_bp", "t_day", "net_taker_bp", "hit")}
            print("  by year:", {k: (v["events"], round(v["gross_bp"], 1), round(v["t_day"], 1)) for k, v in years.items()})
            print("  by coin:", {k[:-4]: (v["events"], round(v["gross_bp"], 1), round(v["t_day"], 1)) for k, v in coins.items()})
            # fills
            fills = {}
            for label, col in (("next open (convention)", f"f{h}"), ("one minute late", f"lag2_{h}"), ("two minutes late", f"lag3_{h}"),
                               ("minute vwap -> minute vwap", f"vwap_{h}"), ("worst print of first minute", f"worst_{h}"),
                               ("WRONG: event bar's own close", f"cc_{h}"), ("no halt rule (missing bars only)", f"nohalt_{h}")):
                r = stats(e, h, col)
                fills[label] = {k: r[k] for k in ("events", "gross_bp", "t_gross", "median_bp", "hit", "net_taker_bp")}
                print(f"  fill {label:34s}: n {r['events']:6d}, mean {r['gross_bp']:7.2f} bp (t {r['t_gross']:5.2f}), median {r['median_bp']:6.1f}, net taker {r['net_taker_bp']:7.2f}")
            entry.update({"by_year": years, "by_coin": coins, "fills": fills})
            out[f"{segment}/{name}"] = entry

            # event-by-event comparison with the researcher's table
            theirs = pd.concat([pd.read_parquet(THEIRS / f"events_{segment}" / f"um_{sym}.parquet") for sym in al.PERPS], ignore_index=True)
            theirs = theirs[theirs.definition == name][["symbol", "time", "side", f"f{h}", "c0", "o1"]].rename(columns={f"f{h}": "f_theirs", "side": "side_hyp", "c0": "c0_t", "o1": "o1_t"})
            theirs["time"] = theirs["time"].dt.as_unit("ns")
            both = e.merge(theirs, on=["symbol", "time"], how="outer", indicator=True)
            only_mine = both[both["_merge"] == "left_only"]
            only_theirs = both[both["_merge"] == "right_only"]
            common = both[both["_merge"] == "both"]
            fin = common[np.isfinite(common[f"f{h}"]) & np.isfinite(common["f_theirs"])]
            diff = (fin[f"f{h}"] - fin["f_theirs"]).abs().max() if len(fin) else float("nan")
            nan_mismatch = int((np.isfinite(common[f"f{h}"]) != np.isfinite(common["f_theirs"])).sum())
            # the entry price in their table (o1) against the raw open of bar t+1 from my grid
            o1_diff = float((common["o1"] / common["o1_t"] - 1).abs().max())
            print(f"  event match: common {len(common)}, only mine {len(only_mine)}, only theirs {len(only_theirs)}; max |f own - f theirs| {diff:.2e}; "
                  f"NaN mismatches {nan_mismatch}; max |o1 own / o1 theirs - 1| {o1_diff:.1e}")
            if len(only_mine) and len(only_mine) <= 100:
                print("    only mine, finite f:", int(np.isfinite(only_mine[f"f{h}"]).sum()), "per coin", only_mine.groupby("symbol").size().to_dict())
            if len(only_theirs):
                print("    only theirs:", only_theirs.groupby("symbol").size().to_dict())
            entry["match"] = {"common": int(len(common)), "only_mine": int(len(only_mine)), "only_theirs": int(len(only_theirs)), "max_abs_f_diff": float(diff),
                              "nan_mismatch": nan_mismatch, "max_rel_entry_price_diff": o1_diff}
    (OUT / "headline.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
