"""Write the list of listed instruments that are not crypto coins.

Since late 2025 the exchange lists perpetuals on stocks, ETFs, commodities and
currencies (TSLAUSDT, SPYUSDT, XAGUSDT, CLUSDT, USDBRLUSDT, ...) and spot tokens
on stocks (AAPLBUSDT, ...), under names that look like any other symbol. Left
alone they would enter the "liquid coins" universe.

The archive carries no instrument metadata, so they are recognised by what sets
them apart: their underlying market is closed at the weekend, so their weekend
bars are tiny. The rule below is applied to instruments first traded from
October 2025, and a short hand-checked list covers the ones the rule misses
(too little history, or commodities that also move at weekends).

This is a METADATA job: it reads listing dates and bar ranges straight from the
parquet files (including the hold-out period) and writes names only. It computes
no return and no strategy statistic. Re-run it after refreshing the data, and
review new listings: in live trading the exchange's own instrument metadata is
the authoritative source.

    python scripts/classify_instruments.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "parquet"
OUT = ROOT / "src" / "algotrade" / "data" / "instrument_kinds.json"
SINCE = pd.Timestamp("2025-10-01", tz="UTC")

# Not caught by the weekend rule; checked by name (stocks, ETFs, commodities, a currency pair, pre-IPO contracts).
MANUAL_TRADFI_UM = [
    "ABBVUSDT", "AMGNUSDT", "ANTHROPICUSDT", "BACUSDT", "BRKBUSDT", "BZUSDT", "CEGUSDT", "CLUSDT", "EIDOUSDT",
    "EQTUSDT", "EWZSUSDT", "EWZUSDT", "IBKRUSDT", "IWMUSDT", "KOUSDT", "MDTUSDT", "MELIUSDT", "NVOUSDT",
    "OPENAIUSDT", "OXYUSDT", "PFEUSDT", "SMRUSDT", "STMUSDT", "USDBRLUSDT", "XLEUSDT",
]
INDEX_UM = ["BLUEBIRDUSDT", "BTCDOMUSDT", "DEFIUSDT", "DOTECOUSDT", "FOOTBALLUSDT"]
GOLD = {"XAUUSDT", "XAUTUSDT", "PAXGUSDT"}  # classified as gold elsewhere


def weekend_ratios(market: str) -> pd.DataFrame:
    rows = []
    for path in sorted((STORE / market / "1d").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["high", "low", "close", "quote_volume"])
        if frame.empty:
            continue
        span = (frame["high"] - frame["low"]) / frame["close"]
        weekend = frame.index.day_of_week >= 5
        both = weekend.any() and (~weekend).any()
        rows.append(
            {
                "symbol": path.stem,
                "first": frame.index[0],
                "bars": len(frame),
                "range_ratio": span[weekend].mean() / span[~weekend].mean() if both else np.nan,
                "volume_ratio": frame["quote_volume"][weekend].mean() / frame["quote_volume"][~weekend].mean() if both else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def main() -> None:
    um, spot = weekend_ratios("um"), weekend_ratios("spot")
    late_um = um[um["first"] >= SINCE]
    quiet = (late_um["range_ratio"] < 0.55) | ((late_um["volume_ratio"] < 0.45) & (late_um["range_ratio"] < 0.65))
    tradfi_um = (set(late_um.index[quiet]) | (set(MANUAL_TRADFI_UM) & set(um.index))) - GOLD
    late_spot = spot[spot["first"] >= SINCE]
    stock_tokens = late_spot.index.str.match(r"^[A-Z0-9]+BUSDT$") & (late_spot["range_ratio"] < 0.55)
    tradfi_spot = set(late_spot.index[stock_tokens])
    kinds = {
        "generated_from": "listing dates and weekend/weekday bar ranges in data/parquet (names only)",
        "tradfi": {"um": sorted(tradfi_um), "spot": sorted(tradfi_spot)},
        "index": {"um": sorted(set(INDEX_UM) & set(um.index)), "spot": []},
    }
    OUT.write_text(json.dumps(kinds, indent=1) + "\n")
    print(f"tradfi: {len(tradfi_um)} perpetuals, {len(tradfi_spot)} spot tokens; index: {len(kinds['index']['um'])} perpetuals")
    missed = sorted(set(MANUAL_TRADFI_UM) - set(um.index))
    if missed:
        print("manual names not in the store:", missed)
    borderline = late_um[~late_um.index.isin(tradfi_um) & (late_um["range_ratio"] < 0.7)]
    print("late listings kept as coins although their weekends are quiet (review):")
    print(borderline[["bars", "range_ratio", "volume_ratio"]].round(2).to_string())


if __name__ == "__main__":
    main()
