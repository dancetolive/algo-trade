"""Freeze the selection made on DESIGN evidence, before any validation bar is read.

    python -I -B write_selection.py

Writes selection.json once and refuses to overwrite it. Everything the
validation run is allowed to do is listed in that file.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "selection.json")

SELECTED = [
    {
        "def_id": "sweep_N240_x10_k1", "h": 240, "direction": "reversal (hypothesis sign as defined)",
        "why": "Of the 420 looks, the only one with day-clustered |t| >= 3.9 (Bonferroni 5%) AND a positive mean after a taker "
               "round trip. Weakness known at selection time: the effect is on the long side (after sweeps of lows) and is "
               "concentrated in 2020-2021; 2023 is about zero.",
    },
    {
        "def_id": "sweep_N240_x10_k1", "h": 60, "direction": "reversal (hypothesis sign as defined)",
        "why": "The same structure at the horizon where the evidence is strongest (t = 6.2, positive in all four design years "
               "and all six coins). Gross about equals the taker cost, so it can only matter for a maker.",
    },
    {
        "def_id": "sweep_N30_x2_k3", "h": 5, "direction": "reversal (hypothesis sign as defined)",
        "why": "The most significant of all 420 looks (t = 12.2), same sign in every design year and every coin. Its size "
               "(about 1 bp) is an order of magnitude below any cost; it is carried forward to show whether the most robust "
               "statistical effect persists, not because it could be traded.",
    },
]

PROTOCOL = {
    "segment_rules": "design = bars before 2024-01-01; validation = 2024-01-01 .. 2025-06-30; hold-out never loaded",
    "part_a_validation": {
        "what": "exactly the three (definition, horizon) pairs above, six perpetuals pooled, same code, same baseline "
                "construction, same clustering; spot BTC/ETH as cross-check; breakdown by year, coin and direction",
        "replicates_if": "same sign as on design and day-clustered t of the excess over baseline >= 2.4 "
                         "(two-sided 5% with a Bonferroni factor of 3)",
        "pays_taker_costs_if": "mean gross minus the taker round trip > 0 with (gross - cost) / clustered s.e. >= 2, "
                               "on design and on validation",
        "maker_diagnostic": "maker_sim.py on the same three pairs: limit order at the event bar's close, resting 5 minutes, "
                            "filled only on a strict trade-through; reported with a 4 bp (maker both sides) and a maker-in / "
                            "taker-out cost. Counts as 'edge only with maker fills' if the 4 bp figure is > 0 with t >= 2 "
                            "on design and on validation while the taker test fails.",
    },
    "part_b": {
        "rule": "blend_weights() of research/baseline/trend_baseline.py on Binance spot daily closes of BTCUSDT and ETHUSDT; "
                "a change decided at the close of day D is executed at 00:00 UTC of day D+1",
        "trades": "every (coin, day) on which the target weight rises (entry: wait for the first BULLISH occurrence) or "
                  "falls (exit: wait for the first BEARISH occurrence)",
        "structures_selected": ["sweep_N240_x10_k1", "sweep_N30_x2_k3"],
        "structures_companion": ["bos_s5_all", "fvgfill_m0p5_W60"],
        "companion_note": "not selected in Part A (no usable edge on design); included because 'wait for a 1-minute break of "
                          "structure' and 'wait for the return into a fair value gap' are the entry triggers discretionary "
                          "traders actually use. Reported separately from the selected ones.",
        "windows_minutes": [240, 1440],
        "fill": "open of the minute after the first occurrence detected at or after the 00:00 bar; if none occurs, the open "
                "of the bar at 00:00 + window",
        "measures": "improvement in bp versus the 00:00 open (positive = better price), mean, median, day-clustered t, "
                    "share of trades made worse, share triggered, mean wait; controls: waiting to the end of the window "
                    "regardless, and the same waiting times applied without the structure (time-matched placebo)",
        "markets": "spot 1-minute bars (primary: the rule trades spot) and perpetual 1-minute bars (secondary)",
        "supplement": "the same comparison on EVERY calendar day (both sides), as a higher-powered check of whether waiting "
                      "for the structure beats the open at all",
    },
}


def main() -> None:
    if os.path.exists(PATH):
        print("selection.json already exists; it is frozen and will not be overwritten")
        sys.exit(1)
    table = pd.read_csv(os.path.join(HERE, "out", "table_design.csv")).set_index(["def_id", "h"])
    looks = int((table["family"] != "control").sum())
    selected = []
    for item in SELECTED:
        row = table.loc[(item["def_id"], item["h"])]
        record = dict(item)
        record["design"] = {k: (int(row[k]) if k in ("events", "days") else round(float(row[k]), 4))
                            for k in ("events", "days", "gross", "base", "excess", "t_excess", "t_gross", "t_naive", "hit",
                                      "cost_taker", "net_taker", "net_maker_bound")}
        selected.append(record)
    payload = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "statement": "Written after the design run and before any price, return or event dated 2024-01-01 or later was "
                     "computed or looked at by this study. (Disclosure: a data-availability printout of the full files - "
                     "row counts, first and last timestamps, number of gaps - was seen at the very start; it contains no prices.)",
        "design_looks": looks,
        "selected": selected,
        "protocol": PROTOCOL,
    }
    with open(PATH, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
