"""Audit step 0: inventory of the 1-minute sets through the locked loader (no hold-out)."""
import os, sys
assert os.environ.get("ALGOTRADE_ALLOW_HOLDOUT") is None
import numpy as np, pandas as pd
from algotrade.data import store

rows = []
for market in ("um", "spot"):
    for sym in store.symbols(market, "1m"):
        df = store.load_klines(market, "1m", sym, columns=["open"])
        if len(df) == 0:
            rows.append(dict(market=market, symbol=sym, rows=0))
            continue
        idx = df.index
        mins = (idx.as_unit("s").asi8 // 60)
        d = np.diff(mins)
        rows.append(dict(market=market, symbol=sym, rows=len(df), first=str(idx[0]), last=str(idx[-1]),
                         gaps=int((d > 1).sum()), missing=int((d - 1)[d > 1].sum()), dups=int((d <= 0).sum()),
                         nan_open=int(df["open"].isna().sum())))
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.to_string(index=False))
print("HOLDOUT_START", store.HOLDOUT_START, "max last:", t["last"].dropna().max())
