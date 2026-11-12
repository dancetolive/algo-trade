"""Design-segment tables: every look, pooled over the six perpetuals, plus breakdowns and the spot cross-check.

    python -I design_tables.py        -> out/tables/*.csv and a printed summary

Reads only out/design/*.npz (written by run_events.py with end = 2024-01-01).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analysis as an  # noqa: E402
from run_events import PERPS  # noqa: E402

TABLES = an.OUT / "tables"
#: (definition, horizon) pairs that get a by-year / by-coin breakdown in the report
BREAKDOWNS = [
    ("move5_abs2pct_dn", 5), ("move5_abs2pct_dn", 15), ("move5_abs2pct_dn", 60), ("move5_abs2pct_dn", 240),
    ("move1_z8_dn", 15), ("move1_z5_dn", 15), ("move1_z3_dn", 15), ("move5_z3_dn", 15),
    ("move1_z8_up", 15), ("move1_z5_up", 15), ("move5_abs2pct_up", 15),
    ("trdspike10_cont_up", 5), ("trdspike10_cont_dn", 5), ("volspike5_exh_dn", 5), ("volspike5_cont_dn", 5), ("trdspike5_exh_dn", 15),
    ("ofi5_z2_both", 5), ("ofi1_z2_both", 5), ("sign1", 1), ("sign5", 5), ("sign15", 15), ("sign60", 60),
    ("minute00", 1), ("minute15", 1), ("minute30", 1), ("minute45", 1),
    ("fund_post5_big", 5), ("fund_post15_big", 15), ("fund_pre5_big", 5), ("hour05", 60), ("hour14", 60),
]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    perps = an.load("design", "um", PERPS)
    pooled = an.table(perps)
    pooled.to_csv(TABLES / "design_perp_pooled.csv", index=False)
    full = an.table(perps, all_horizons=True)
    full.to_csv(TABLES / "design_perp_pooled_all_horizons.csv", index=False)

    spot = an.load("design", "spot", ["BTCUSDT", "ETHUSDT"])
    perp2 = an.load("design", "um", ["BTCUSDT", "ETHUSDT"])
    cross = an.table(spot).merge(an.table(perp2), on=["definition", "family", "h", "hypothesis"], suffixes=("_spot", "_perp"))
    cross.to_csv(TABLES / "design_spot_vs_perp_btc_eth.csv", index=False)

    rows = []
    for name, h in BREAKDOWNS:
        for kind, frame in (("year", an.by_year(perps, name, h)), ("coin", an.by_symbol(perps, name, h)), ("half", an.halves(perps, name, h))):
            for key, row in frame.iterrows():
                rows.append({"definition": name, "h": h, "split": kind, "key": str(key), **row.to_dict()})
    pd.DataFrame(rows).to_csv(TABLES / "design_breakdowns.csv", index=False)

    print("looks (definition x horizon):", len(pooled), pooled.groupby("family").size().to_dict())
    for bar in (3, 4):
        print(f"|t_day| > {bar}:", int((pooled.t_day.abs() > bar).sum()), "by family", pooled[pooled.t_day.abs() > bar].groupby("family").size().to_dict())
    pooled["net_best_side_taker"] = pooled.gross_bp.abs() - pooled.cost_perp_taker_bp
    pays = pooled[(pooled.t_day.abs() > 3) & (pooled.net_best_side_taker > 0)]
    print("|t| > 3 and |gross| above a taker round trip:", len(pays), "by family", pays.groupby("family").size().to_dict())
    print("spot vs perp (BTC+ETH) correlation of excess bp over", len(cross), "looks:", round(float(np.corrcoef(cross.excess_bp_spot, cross.excess_bp_perp)[0, 1]), 3))


if __name__ == "__main__":
    main()
