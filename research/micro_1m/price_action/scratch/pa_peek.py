import time
import numpy as np, pandas as pd
from algotrade.data import store
t0=time.time()
for mkt, syms in (("um", ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"]), ("spot", ["BTCUSDT","ETHUSDT","PAXGUSDT"])):
    for s in syms:
        df = store.load_klines(mkt, "1m", s, end="2024-01-01")
        idx = df.index
        step = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
        gaps = step[step > pd.Timedelta("1min")]
        print(mkt, s, df.shape, idx[0], idx[-1], "idx dtype", idx.dtype,
              "n_gaps", len(gaps), "max gap", gaps.max() if len(gaps) else None, "missing min", int((gaps.dt.total_seconds()/60 - 1).sum()) if len(gaps) else 0,
              "dups", int(idx.duplicated().sum()), "mono", idx.is_monotonic_increasing, f"{time.time()-t0:.1f}s")
        if len(gaps):
            print("   gaps:", [(str(k), str(v)) for k, v in gaps.sort_values(ascending=False).head(8).items()])
        print("   zero-volume bars:", int((df.volume==0).sum()), "nan rows:", int(df.isna().any(axis=1).sum()),
              "bars with high==low:", int((df.high==df.low).sum()), "ohlc violations:", int(((df.high < df[['open','close']].max(axis=1)) | (df.low > df[['open','close']].min(axis=1))).sum()))
        if s == "BTCUSDT":
            print(df.head(3).to_string()); print(df.dtypes.to_string()); print(df.memory_usage(deep=True).sum()/1e6, "MB")
