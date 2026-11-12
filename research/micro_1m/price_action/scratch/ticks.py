import numpy as np, pandas as pd
from algotrade.data import store
for mkt, s in [("spot","PAXGUSDT"),("um","BTCUSDT"),("um","ETHUSDT"),("um","SOLUSDT"),("um","BNBUSDT"),("um","XRPUSDT"),("um","DOGEUSDT")]:
    df = store.load_klines(mkt, "1m", s, start="2023-01-01", end="2024-01-01", columns=["open","high","low","close","volume"])
    px = np.unique(np.round(df[["open","high","low","close"]].to_numpy().ravel(), 10))
    tick = np.min(np.diff(px))
    rng = ((df.high - df.low) / df.close * 1e4)
    print(f"{mkt} {s}: tick ~{tick:.6g} = {tick / df.close.median() * 1e4:.2f} bp of price; median 1m range {rng.median():.1f} bp, mean {rng.mean():.1f} bp; zero-range bars {(rng == 0).mean():.1%}; median |1m ret| {((df.close/df.open-1).abs()*1e4).median():.2f} bp")
