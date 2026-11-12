"""Audit step 0: what is in the 1-minute store (index, gaps, zero-volume bars). Reads through the loader only."""
import os
assert "ALGOTRADE_ALLOW_HOLDOUT" not in os.environ
import numpy as np, pandas as pd
from algotrade.data import store

for market, syms in (("um", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "PAXGUSDT", "XAUUSDT"]),
                     ("spot", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "PAXGUSDT"])):
    for s in syms:
        try:
            df = store.load_klines(market, "1m", s, columns=["open", "close", "volume", "quote_volume"])
        except Exception as e:
            print(market, s, "ERR", e); continue
        if len(df) == 0:
            print(market, s, "EMPTY (after hold-out lock)"); continue
        idx = df.index
        d = np.diff(idx.asi8) // (60 * 10**9) if idx.unit == "ns" else np.diff(idx.as_unit("ns").asi8) // (60 * 10**9)
        gaps = d[d > 1]
        zero = (df["quote_volume"] <= 0).to_numpy()
        print(f"{market:4s} {s:9s} rows {len(df):8d}  {idx[0]} .. {idx[-1]}  tz={idx.tz} dup={idx.has_duplicates} mono={idx.is_monotonic_increasing} "
              f"gaps={len(gaps)} missing_bars={int((gaps-1).sum())} max_gap={int(gaps.max()) if len(gaps) else 0} zero_vol_bars={int(zero.sum())}")
        del df
