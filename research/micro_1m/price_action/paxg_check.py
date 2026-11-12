"""Side check on tokenised gold (PAXGUSDT spot), DESIGN segment only, reported separately and never pooled.

    python -I -B paxg_check.py

PAXG trades thinly on a 1-minute scale (about a third of its design-segment
bars have no trade at all), so its bars are a poor picture of an executable
price and the 24 bp spot round trip is an under-estimate of the real cost.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze  # noqa: E402
import run_segment  # noqa: E402


def main() -> None:
    result = run_segment.run_symbol(("design", "spot", "PAXGUSDT", None))
    result.pop("meta")
    print(result)
    source = os.path.join(run_segment.OUT, "daily_design_spot_PAXGUSDT.parquet")
    target = os.path.join(run_segment.OUT, "daily_design_paxg_PAXGUSDT.parquet")  # keep it out of the BTC / ETH spot pool
    os.replace(source, target)
    daily = analyze.load_daily("design", "paxg")
    table = analyze.full_table(daily, "spot")
    table.to_csv(os.path.join(run_segment.OUT, "table_design_paxg.csv"), index=False)
    real = table[~table["def_id"].str.startswith("ctl")]
    pd.set_option("display.width", 250)
    print(f"{len(real)} looks; |t_excess| > 3: {(real['t_excess'].abs() > 3).sum()}; net of spot taker costs > 0: {(real['net_taker'] > 0).sum()}")
    cols = ["def_id", "h", "events", "days", "gross", "base", "excess", "t_excess", "net_taker"]
    print(real.reindex(real["t_excess"].abs().sort_values(ascending=False).index)[cols].head(25).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(real.sort_values("net_taker", ascending=False)[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
