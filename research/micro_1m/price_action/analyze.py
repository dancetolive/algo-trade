"""Tables from the per-day sufficient statistics written by run_segment.py.

    python -I -B analyze.py design        # every definition x horizon, pooled over the six perpetuals
    python -I -B analyze.py validation    # the pre-registered definitions only

Writes out/table_<segment>.csv (all rows), out/breakdown_<segment>.csv (by
year, by symbol, by direction for the rows worth a closer look) and prints the
tables used in REPORT.md.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eventstudy as es  # noqa: E402

OUT = os.path.join(HERE, "out")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def load_daily(segment: str, market: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(OUT, f"daily_{segment}_{market}_*.parquet")))
    table = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    table["year"] = (EPOCH + pd.to_timedelta(table["day"], unit="D")).dt.year.to_numpy()
    return table


def costs(market: str, symbols) -> dict[str, dict[str, float]]:
    venue = "perp_taker" if market == "um" else "spot_taker"
    return {"taker": {s: es.round_trip_cost_bp(s, venue) for s in symbols},
            "maker": {s: es.round_trip_cost_bp(s, "perp_maker") for s in symbols}}


def full_table(daily: pd.DataFrame, market: str, horizons=es.HORIZONS) -> pd.DataFrame:
    """One row per definition x horizon."""
    cost = costs(market, daily["symbol"].unique())
    rows = []
    for def_id, part in daily.groupby("def_id", sort=False):
        stats = es.pooled_stats(part, horizons, cost_bp=cost["taker"])
        for h, row in stats.iterrows():
            if row.get("events", 0) == 0 or "gross" not in row or not np.isfinite(row["gross"]):
                continue
            record = {"def_id": def_id, "h": int(h)}
            record.update(row.to_dict())
            record["net_taker"] = record.pop("net")
            record["cost_taker"] = record.pop("cost")
            record["net_maker_bound"] = record["gross"] - 4.0 if market == "um" else np.nan
            rows.append(record)
    table = pd.DataFrame(rows)
    table["events"] = table["events"].astype(int)
    table["days"] = table["days"].astype(int)
    return table


def breakdown(daily: pd.DataFrame, def_id: str, h: int, market: str) -> pd.DataFrame:
    """The same statistics by calendar year, by symbol and by direction."""
    part = daily[daily["def_id"] == def_id]
    cost = costs(market, part["symbol"].unique())["taker"]
    rows = []
    for label, column in (("year", "year"), ("symbol", "symbol"), ("direction", "dir")):
        for value, chunk in part.groupby(column, sort=True, observed=True):
            stats = es.pooled_stats(chunk, (h,), cost_bp=cost)
            if stats.loc[h].get("events", 0) == 0:
                continue
            row = stats.loc[h].to_dict()
            row.update(def_id=def_id, h=h, by=label, value=str(value) if label != "direction" else ("long" if value > 0 else "short"))
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(table: pd.DataFrame, columns) -> str:
    return table[columns].to_string(index=False, float_format=lambda v: f"{v:.2f}")


def main() -> None:
    segment = sys.argv[1]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 2000)
    daily = load_daily(segment, "um")
    table = full_table(daily, "um")
    with open(os.path.join(OUT, f"run_{segment}.json")) as handle:
        meta = json.load(handle)["definitions"]
    table.insert(1, "family", table["def_id"].map(lambda d: meta[d]["family"]))
    table.to_csv(os.path.join(OUT, f"table_{segment}.csv"), index=False)

    spot = load_daily(segment, "spot")
    spot_table = full_table(spot, "spot")
    spot_table.to_csv(os.path.join(OUT, f"table_{segment}_spot.csv"), index=False)

    # the same statistics on BTC and ETH perpetuals only, to compare like with like against spot
    major = full_table(daily[daily["symbol"].isin(["BTCUSDT", "ETHUSDT"])], "um")
    major.to_csv(os.path.join(OUT, f"table_{segment}_um_btc_eth.csv"), index=False)

    real = table[table["family"] != "control"]
    if segment != "design":
        with open(os.path.join(HERE, "selection.json")) as handle:
            pairs = {(item["def_id"], int(item["h"])) for item in json.load(handle)["selected"]}
        assert set(zip(real["def_id"], real["h"])) == pairs, "the validation tables must hold the pre-registered pairs and nothing else"
        both = real.merge(spot_table, on=["def_id", "h"], suffixes=("", "_spot")).merge(major, on=["def_id", "h"], suffixes=("", "_perp_btc_eth"))
        print(f"segment {segment}: the {len(real)} pre-registered pairs, six perpetuals pooled")
        print(fmt(real, ["def_id", "h", "events", "days", "gross", "base", "excess", "se_excess", "t_excess", "t_gross", "t_naive", "hit",
                         "cost_taker", "net_taker", "net_maker_bound"]))
        print("\nBTC and ETH only: perpetuals versus spot")
        print(fmt(both, ["def_id", "h", "events_perp_btc_eth", "gross_perp_btc_eth", "t_excess_perp_btc_eth", "net_taker_perp_btc_eth",
                         "events_spot", "gross_spot", "t_excess_spot", "net_taker_spot"]))
        frames = [breakdown(daily, d, h, "um") for d, h in sorted(pairs)]
        detail = pd.concat(frames, ignore_index=True)
        detail.to_csv(os.path.join(OUT, f"breakdown_{segment}.csv"), index=False)
        print("\nby year, coin and direction")
        print(fmt(detail, ["def_id", "h", "by", "value", "events", "days", "gross", "base", "excess", "t_excess", "hit", "net"]))
        return
    print(f"segment {segment}: {real['def_id'].nunique()} definitions x {real['h'].nunique()} horizons = {len(real)} looks")
    cols = ["def_id", "h", "events", "days", "gross", "base", "excess", "t_excess", "t_naive", "hit", "net_taker", "net_maker_bound"]
    print("\nCONTROLS (machinery check on real data)")
    print(fmt(table[table["family"] == "control"], cols))
    print("\nALL LOOKS")
    print(fmt(real, cols))
    print("\n|t_excess| > 3:", int((real["t_excess"].abs() > 3).sum()), "of", len(real),
          "   |t_excess| > 3.9 (Bonferroni 5% for this many looks):", int((real["t_excess"].abs() > 3.9).sum()))
    print("net_taker > 0:", int((real["net_taker"] > 0).sum()), "   gross - 4 bp (maker bound) > 0:", int((real["net_maker_bound"] > 0).sum()))

    frames = []
    for def_id in real["def_id"].unique():
        for h in es.HORIZONS:
            frames.append(breakdown(daily, def_id, h, "um"))
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(OUT, f"breakdown_{segment}.csv"), index=False)


if __name__ == "__main__":
    main()
